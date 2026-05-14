from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch

from graph_features import ProteinMPNNFeaturizer
from protein_mpnn_utils import ALPHABET


AA_TO_IDX = {aa: i for i, aa in enumerate(ALPHABET)}


def _split_from_cluster(cluster_id: int, valid_clusters: set[int], test_clusters: set[int]) -> str:
    if cluster_id in test_clusters:
        return "test"
    if cluster_id in valid_clusters:
        return "valid"
    return "train"


def _length_stats(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {"count": 0}
    arr = np.asarray(lengths, dtype=np.float32)
    return {
        "count": int(arr.shape[0]),
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _read_cluster_file(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return {int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _safe_name(chain_id: str) -> str:
    return chain_id.replace("/", "_")


def _encode_sequence(raw_sequence: str) -> tuple[np.ndarray, int]:
    replaced = 0
    tokens = []
    for aa in raw_sequence.upper():
        if aa in AA_TO_IDX:
            tokens.append(AA_TO_IDX[aa])
        else:
            replaced += 1
            tokens.append(AA_TO_IDX["X"])
    return np.asarray(tokens, dtype=np.int32), replaced


def _build_local_neighbor_index(length: int, k_neighbors: int) -> np.ndarray:
    positions = np.arange(length, dtype=np.int32)
    neighbor_idx = np.zeros((length, k_neighbors), dtype=np.int32)
    for i in range(length):
        order = np.argsort(np.abs(positions - i), kind="stable")
        top = order[: min(length, k_neighbors)]
        if top.shape[0] < k_neighbors:
            top = np.concatenate([top, np.full((k_neighbors - top.shape[0],), i, dtype=np.int32)])
        neighbor_idx[i] = top
    return neighbor_idx


def _mock_edge_features(length: int, edge_dim: int, k_neighbors: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    edge_features = rng.normal(0.0, 1.0, size=(length, k_neighbors, edge_dim)).astype(np.float32)
    neighbor_idx = _build_local_neighbor_index(length, k_neighbors)
    return edge_features, neighbor_idx


def _load_chain_pt_coords(data_path: Path, chain_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    if "_" not in chain_id:
        return None
    pdb_id, chain = chain_id.split("_", 1)
    pt_path = data_path / "pdb" / pdb_id[1:3] / f"{pdb_id}_{chain}.pt"
    if not pt_path.exists():
        return None

    chain_data = torch.load(pt_path, map_location="cpu")
    if "xyz" not in chain_data:
        return None
    xyz = np.asarray(chain_data["xyz"], dtype=np.float32)  # [L, 14, 3]
    if xyz.ndim != 3 or xyz.shape[1] < 4 or xyz.shape[2] != 3:
        return None
    atom_coords = xyz[:, :4, :]
    residue_mask = np.all(np.isfinite(atom_coords), axis=(1, 2)).astype(np.float32)
    atom_coords = np.nan_to_num(atom_coords, nan=0.0)
    return atom_coords, residue_mask


def _compute_graph_from_coords(
    featurizer: ProteinMPNNFeaturizer,
    atom_coords: np.ndarray,
    residue_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    length = atom_coords.shape[0]
    residue_idx = jnp.arange(length, dtype=jnp.int32)[None, :]
    chain_ids = jnp.zeros((1, length), dtype=jnp.int32)
    edge_features, neighbor_idx = featurizer(
        atom_coords=jnp.asarray(atom_coords[None, ...], dtype=jnp.float32),
        residue_mask=jnp.asarray(residue_mask[None, ...], dtype=jnp.float32),
        residue_idx=residue_idx,
        chain_ids=chain_ids,
    )
    edge_features = np.asarray(jax.device_get(edge_features[0]), dtype=np.float32)
    neighbor_idx = np.asarray(jax.device_get(neighbor_idx[0]), dtype=np.int32)
    return edge_features, neighbor_idx


@dataclass
class SplitStats:
    lengths: list[int] = field(default_factory=list)
    token_count: int = 0
    examples: int = 0
    from_structure: int = 0
    from_mock: int = 0
    aa_counter: Counter = field(default_factory=Counter)
    cluster_counter: Counter = field(default_factory=Counter)


def _print_report(report: dict[str, Any]) -> None:
    print("\n=== Dataset Build Report ===")
    print(f"backend={report['jax_backend']} devices={report['jax_device_count']}")
    print(f"input={report['input_data_path']}")
    print(f"output={report['output_path']}")
    print(f"kept_examples={report['kept_examples']} skipped_examples={report['skipped_examples']}")
    print(f"unknown_AA_replaced={report['unknown_aa_replaced']}")
    print(f"limits: max_length={report['max_length']} max_examples={report['max_examples']}")
    for split_name, split in report["splits"].items():
        print(
            f"\n[{split_name}] examples={split['examples']} tokens={split['token_count']} "
            f"from_structure={split['from_structure']} from_mock={split['from_mock']}"
        )
        print(f"  length_stats={split['length_stats']}")
        print(f"  top_clusters={split['top_clusters']}")
        print(f"  top_AAs={split['top_aas']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/Users/thom/Projects/references/ProteinMPNN/training/small_dataset/pdb_2021aug02_sample",
    )
    parser.add_argument("--output_path", type=str, default="./precomputed_dataset")
    parser.add_argument("--edge_dim", type=int, default=128)
    parser.add_argument("--k_neighbors", type=int, default=48)
    parser.add_argument("--max_length", type=int, default=1000)
    parser.add_argument("--max_examples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_mock_missing_structure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--report_json", type=str, default="")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        (output_path / split).mkdir(parents=True, exist_ok=True)

    list_csv = data_path / "list.csv"
    if not list_csv.exists():
        raise FileNotFoundError(f"Missing {list_csv}")

    valid_clusters = _read_cluster_file(data_path / "valid_clusters.txt")
    test_clusters = _read_cluster_file(data_path / "test_clusters.txt")

    featurizer = ProteinMPNNFeaturizer(
        node_features=args.edge_dim,
        edge_features=args.edge_dim,
        rngs=nnx.Rngs(args.seed),
        k_neighbors=args.k_neighbors,
        augment_eps=0.0,
        ca_only=False,
    )

    split_stats = {"train": SplitStats(), "valid": SplitStats(), "test": SplitStats()}
    skipped_examples = 0
    unknown_aa_replaced = 0
    kept_examples = 0

    with list_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if kept_examples >= args.max_examples:
                break
            chain_id = row["CHAINID"]
            cluster_id = int(row["CLUSTER"])
            split = _split_from_cluster(cluster_id, valid_clusters, test_clusters)
            sequence_tokens, replaced = _encode_sequence(row["SEQUENCE"])
            unknown_aa_replaced += replaced

            length = int(sequence_tokens.shape[0])
            if length == 0 or length > args.max_length:
                skipped_examples += 1
                continue

            struct = _load_chain_pt_coords(data_path, chain_id)
            if struct is None:
                if not args.allow_mock_missing_structure:
                    skipped_examples += 1
                    continue
                edge_features, neighbor_idx = _mock_edge_features(
                    length=length,
                    edge_dim=args.edge_dim,
                    k_neighbors=args.k_neighbors,
                    seed=(abs(hash(chain_id)) + args.seed) % (2**31 - 1),
                )
                residue_mask = np.ones((length,), dtype=np.float32)
                from_structure = False
            else:
                atom_coords, residue_mask = struct
                if atom_coords.shape[0] != length:
                    # Keep sequence from list.csv and truncate/pad coords to match length conservatively.
                    min_len = min(atom_coords.shape[0], length)
                    atom_coords = atom_coords[:min_len]
                    residue_mask = residue_mask[:min_len]
                    sequence_tokens = sequence_tokens[:min_len]
                    length = min_len
                    if length == 0:
                        skipped_examples += 1
                        continue
                edge_features, neighbor_idx = _compute_graph_from_coords(
                    featurizer=featurizer,
                    atom_coords=atom_coords,
                    residue_mask=residue_mask,
                )
                from_structure = True

            decode_mask = residue_mask.copy()
            output_file = output_path / split / f"{_safe_name(chain_id)}.npz"
            np.savez_compressed(
                output_file,
                edge_features=edge_features.astype(np.float32),
                neighbor_idx=neighbor_idx.astype(np.int32),
                sequence_tokens=sequence_tokens.astype(np.int32),
                residue_mask=residue_mask.astype(np.float32),
                decode_mask=decode_mask.astype(np.float32),
                chain_id=np.asarray(chain_id),
                cluster_id=np.asarray(cluster_id, dtype=np.int32),
            )

            stats = split_stats[split]
            stats.examples += 1
            stats.lengths.append(length)
            stats.token_count += int(np.sum(residue_mask > 0.5))
            stats.cluster_counter[cluster_id] += 1
            if from_structure:
                stats.from_structure += 1
            else:
                stats.from_mock += 1
            for aa_idx in sequence_tokens.tolist():
                stats.aa_counter[int(aa_idx)] += 1
            kept_examples += 1

    if kept_examples == 0:
        raise ValueError("No examples were written.")

    report_splits: dict[str, Any] = {}
    for split, stats in split_stats.items():
        top_clusters = stats.cluster_counter.most_common(10)
        aa_total = max(1, sum(stats.aa_counter.values()))
        top_aas = []
        for aa_idx, count in stats.aa_counter.most_common(10):
            top_aas.append(
                {
                    "aa": ALPHABET[aa_idx],
                    "count": int(count),
                    "fraction": float(count / aa_total),
                }
            )
        report_splits[split] = {
            "examples": stats.examples,
            "token_count": stats.token_count,
            "from_structure": stats.from_structure,
            "from_mock": stats.from_mock,
            "length_stats": _length_stats(stats.lengths),
            "top_clusters": [{"cluster": int(cluster), "count": int(count)} for cluster, count in top_clusters],
            "top_aas": top_aas,
        }

    report = {
        "input_data_path": str(data_path),
        "output_path": str(output_path),
        "jax_backend": jax.default_backend(),
        "jax_device_count": int(jax.device_count()),
        "k_neighbors": args.k_neighbors,
        "edge_dim": args.edge_dim,
        "max_length": args.max_length,
        "max_examples": args.max_examples,
        "allow_mock_missing_structure": bool(args.allow_mock_missing_structure),
        "kept_examples": kept_examples,
        "skipped_examples": skipped_examples,
        "unknown_aa_replaced": unknown_aa_replaced,
        "splits": report_splits,
    }

    report_path = Path(args.report_json) if args.report_json else output_path / "dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_report(report)
    print(f"\nreport_json={report_path}")


if __name__ == "__main__":
    main()
