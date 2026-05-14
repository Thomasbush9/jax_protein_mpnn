from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from losses import loss_nll_jax, loss_smoothed_jax
from model import ProteinMPNN
from protein_mpnn_utils import ALPHABET


AA_TO_IDX = {aa: i for i, aa in enumerate(ALPHABET)}


@dataclass
class GraphExample:
    edge_features: np.ndarray  # [L, K, D]
    neighbor_idx: np.ndarray  # [L, K]
    sequence_tokens: np.ndarray  # [L]
    residue_mask: np.ndarray  # [L]
    decode_mask: np.ndarray  # [L]
    name: str


def _pad_axis0(arr: np.ndarray, target: int, pad_value: float | int) -> np.ndarray:
    pad_len = target - arr.shape[0]
    if pad_len <= 0:
        return arr
    pad_width = [(0, 0)] * arr.ndim
    pad_width[0] = (0, pad_len)
    return np.pad(arr, pad_width=pad_width, mode="constant", constant_values=pad_value)


def _collate_batch(
    examples: list[GraphExample],
    edge_dim: int,
    k_neighbors: int,
    dtype: jnp.dtype,
) -> dict[str, jnp.ndarray]:
    max_len = max(example.sequence_tokens.shape[0] for example in examples)
    edge_batch = []
    idx_batch = []
    seq_batch = []
    residue_mask_batch = []
    decode_mask_batch = []
    for ex in examples:
        length = ex.sequence_tokens.shape[0]
        if ex.edge_features.shape != (length, k_neighbors, edge_dim):
            raise ValueError(
                f"{ex.name}: edge_features shape {ex.edge_features.shape} must be ({length}, {k_neighbors}, {edge_dim})"
            )
        if ex.neighbor_idx.shape != (length, k_neighbors):
            raise ValueError(
                f"{ex.name}: neighbor_idx shape {ex.neighbor_idx.shape} must be ({length}, {k_neighbors})"
            )
        edge_batch.append(_pad_axis0(ex.edge_features, max_len, 0.0))
        idx_batch.append(_pad_axis0(ex.neighbor_idx, max_len, 0))
        seq_batch.append(_pad_axis0(ex.sequence_tokens, max_len, 0))
        residue_mask_batch.append(_pad_axis0(ex.residue_mask, max_len, 0.0))
        decode_mask_batch.append(_pad_axis0(ex.decode_mask, max_len, 0.0))
    return {
        "edge_features": jnp.asarray(np.stack(edge_batch, axis=0), dtype=dtype),
        "neighbor_idx": jnp.asarray(np.stack(idx_batch, axis=0), dtype=jnp.int32),
        "sequence_tokens": jnp.asarray(np.stack(seq_batch, axis=0), dtype=jnp.int32),
        "residue_mask": jnp.asarray(np.stack(residue_mask_batch, axis=0), dtype=jnp.float32),
        "decode_mask": jnp.asarray(np.stack(decode_mask_batch, axis=0), dtype=jnp.float32),
    }


def _token_batches(
    examples: list[GraphExample],
    tokens_per_batch: int,
    rng: np.random.Generator,
    shuffle: bool = True,
) -> list[list[GraphExample]]:
    if not examples:
        return []
    indices = np.arange(len(examples))
    lengths = np.array([examples[i].sequence_tokens.shape[0] for i in indices], dtype=np.int32)
    sorted_order = indices[np.argsort(lengths)]
    batches: list[list[int]] = []
    current: list[int] = []
    current_max_len = 0
    for idx in sorted_order:
        length = examples[idx].sequence_tokens.shape[0]
        proposal_max_len = max(current_max_len, length)
        proposal_tokens = proposal_max_len * (len(current) + 1)
        if current and proposal_tokens > tokens_per_batch:
            batches.append(current)
            current = [idx]
            current_max_len = length
        else:
            current.append(idx)
            current_max_len = proposal_max_len
    if current:
        batches.append(current)
    if shuffle:
        rng.shuffle(batches)
    return [[examples[i] for i in batch] for batch in batches]


def _build_local_neighbor_index(length: int, k_neighbors: int) -> np.ndarray:
    positions = np.arange(length, dtype=np.int32)
    neighbor_idx = np.zeros((length, k_neighbors), dtype=np.int32)
    for i in range(length):
        order = np.argsort(np.abs(positions - i), kind="stable")
        top = order[: min(k_neighbors, length)]
        if top.shape[0] < k_neighbors:
            fill = np.full((k_neighbors - top.shape[0],), i, dtype=np.int32)
            top = np.concatenate([top, fill], axis=0)
        neighbor_idx[i] = top
    return neighbor_idx


def _mock_edges_from_sequence(
    sequence: str,
    edge_dim: int,
    k_neighbors: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    length = len(sequence)
    neighbor_idx = _build_local_neighbor_index(length, k_neighbors)
    rng = np.random.default_rng(seed)
    edge_features = rng.normal(loc=0.0, scale=1.0, size=(length, k_neighbors, edge_dim)).astype(np.float32)
    return edge_features, neighbor_idx


def _load_list_csv_examples(
    dataset_dir: Path,
    edge_dim: int,
    k_neighbors: int,
    max_length: int,
    max_examples: int,
    seed: int,
) -> tuple[list[GraphExample], list[GraphExample]]:
    list_path = dataset_dir / "list.csv"
    valid_clusters_path = dataset_dir / "valid_clusters.txt"
    test_clusters_path = dataset_dir / "test_clusters.txt"
    if not list_path.exists():
        raise FileNotFoundError(f"Missing {list_path}")

    valid_clusters = set()
    test_clusters = set()
    if valid_clusters_path.exists():
        valid_clusters = {int(line.strip()) for line in valid_clusters_path.read_text().splitlines() if line.strip()}
    if test_clusters_path.exists():
        test_clusters = {int(line.strip()) for line in test_clusters_path.read_text().splitlines() if line.strip()}

    train_examples: list[GraphExample] = []
    valid_examples: list[GraphExample] = []
    with list_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = row["CHAINID"]
            sequence = row["SEQUENCE"].strip().upper()
            cluster_id = int(row["CLUSTER"])
            if len(sequence) == 0 or len(sequence) > max_length:
                continue
            sequence_tokens = np.array([AA_TO_IDX.get(aa, AA_TO_IDX["X"]) for aa in sequence], dtype=np.int32)
            residue_mask = np.ones((len(sequence),), dtype=np.float32)
            decode_mask = np.ones((len(sequence),), dtype=np.float32)
            edge_features, neighbor_idx = _mock_edges_from_sequence(
                sequence=sequence,
                edge_dim=edge_dim,
                k_neighbors=k_neighbors,
                seed=abs(hash(name)) + seed,
            )
            example = GraphExample(
                edge_features=edge_features,
                neighbor_idx=neighbor_idx,
                sequence_tokens=sequence_tokens,
                residue_mask=residue_mask,
                decode_mask=decode_mask,
                name=name,
            )
            if cluster_id in test_clusters:
                continue
            if cluster_id in valid_clusters:
                valid_examples.append(example)
            else:
                train_examples.append(example)
            if len(train_examples) >= max_examples and len(valid_examples) >= max(1, max_examples // 10):
                break
    return train_examples[:max_examples], valid_examples[: max(1, max_examples // 10)]


def _load_npz_examples(
    dataset_dir: Path,
    max_length: int,
    max_examples: int,
) -> tuple[list[GraphExample], list[GraphExample]]:
    train_dir = dataset_dir / "train"
    valid_dir = dataset_dir / "valid"
    if not train_dir.exists():
        raise FileNotFoundError(
            f"Expected precomputed tensors at {train_dir}. "
            "Format: train/*.npz and valid/*.npz with keys "
            "edge_features, neighbor_idx, sequence_tokens, residue_mask, decode_mask(optional)."
        )

    def _read_split(split_dir: Path, limit: int) -> list[GraphExample]:
        examples: list[GraphExample] = []
        for npz_path in sorted(split_dir.glob("*.npz")):
            with np.load(npz_path) as data:
                sequence_tokens = data["sequence_tokens"].astype(np.int32)
                if sequence_tokens.shape[0] > max_length:
                    continue
                residue_mask = data["residue_mask"].astype(np.float32)
                decode_mask = data["decode_mask"].astype(np.float32) if "decode_mask" in data else residue_mask.copy()
                examples.append(
                    GraphExample(
                        edge_features=data["edge_features"].astype(np.float32),
                        neighbor_idx=data["neighbor_idx"].astype(np.int32),
                        sequence_tokens=sequence_tokens,
                        residue_mask=residue_mask,
                        decode_mask=decode_mask,
                        name=npz_path.stem,
                    )
                )
            if len(examples) >= limit:
                break
        return examples

    train_examples = _read_split(train_dir, max_examples)
    valid_examples = _read_split(valid_dir, max(1, max_examples // 10)) if valid_dir.exists() else train_examples[:8]
    return train_examples, valid_examples


def load_examples(
    data_path: Path,
    edge_dim: int,
    k_neighbors: int,
    max_length: int,
    max_examples: int,
    seed: int,
) -> tuple[list[GraphExample], list[GraphExample], str]:
    if (data_path / "list.csv").exists():
        train_examples, valid_examples = _load_list_csv_examples(
            dataset_dir=data_path,
            edge_dim=edge_dim,
            k_neighbors=k_neighbors,
            max_length=max_length,
            max_examples=max_examples,
            seed=seed,
        )
        mode = "mock_from_list_csv"
    else:
        train_examples, valid_examples = _load_npz_examples(
            dataset_dir=data_path,
            max_length=max_length,
            max_examples=max_examples,
        )
        mode = "precomputed_npz"
    if not train_examples:
        raise ValueError("No training examples were loaded.")
    if not valid_examples:
        valid_examples = train_examples[: min(len(train_examples), 8)]
    return train_examples, valid_examples, mode


def build_optimizer(
    model: ProteinMPNN,
    hidden_dim: int,
    warmup_steps: int,
    clip_grad_norm: float,
) -> nnx.Optimizer:
    def noam_lr(step: jnp.ndarray) -> jnp.ndarray:
        step = step.astype(jnp.float32) + 1.0
        return 2.0 * (hidden_dim ** -0.5) * jnp.minimum(step ** -0.5, step * (warmup_steps ** -1.5))

    tx_parts = []
    if clip_grad_norm > 0:
        tx_parts.append(optax.clip_by_global_norm(clip_grad_norm))
    tx_parts.append(optax.adam(learning_rate=noam_lr, b1=0.9, b2=0.98, eps=1e-9))
    return nnx.Optimizer(model, optax.chain(*tx_parts), wrt=nnx.Param)


@nnx.jit
def train_step(
    model: ProteinMPNN,
    optimizer: nnx.Optimizer,
    edge_features: jnp.ndarray,
    neighbor_idx: jnp.ndarray,
    sequence_tokens: jnp.ndarray,
    residue_mask: jnp.ndarray,
    decode_mask: jnp.ndarray,
    decode_noise: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    def _loss_fn(m: ProteinMPNN):
        log_probs = m(
            edge_features=edge_features,
            neighbor_idx=neighbor_idx,
            sequence_tokens=sequence_tokens,
            residue_mask=residue_mask,
            decode_mask=decode_mask,
            decode_noise=decode_noise,
        )
        mask_for_loss = residue_mask * decode_mask
        _, loss_smoothed = loss_smoothed_jax(sequence_tokens, log_probs, mask_for_loss)
        nll, _ = loss_nll_jax(sequence_tokens, log_probs, mask_for_loss)
        pred = jnp.argmax(log_probs, axis=-1)
        correct = (pred == sequence_tokens).astype(jnp.float32) * mask_for_loss
        token_count = jnp.maximum(jnp.sum(mask_for_loss), 1.0)
        nll_sum = jnp.sum(nll * mask_for_loss)
        correct_sum = jnp.sum(correct)
        return loss_smoothed, (nll_sum, correct_sum, token_count)

    grad_fn = nnx.value_and_grad(_loss_fn, has_aux=True, argnums=nnx.DiffState(0, nnx.Param))
    (loss, metrics), grads = grad_fn(model)
    optimizer.update(model, grads)
    del loss
    return metrics


@nnx.jit
def eval_step(
    model: ProteinMPNN,
    edge_features: jnp.ndarray,
    neighbor_idx: jnp.ndarray,
    sequence_tokens: jnp.ndarray,
    residue_mask: jnp.ndarray,
    decode_mask: jnp.ndarray,
    decode_noise: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    log_probs = model(
        edge_features=edge_features,
        neighbor_idx=neighbor_idx,
        sequence_tokens=sequence_tokens,
        residue_mask=residue_mask,
        decode_mask=decode_mask,
        decode_noise=decode_noise,
    )
    mask_for_loss = residue_mask * decode_mask
    nll, _ = loss_nll_jax(sequence_tokens, log_probs, mask_for_loss)
    pred = jnp.argmax(log_probs, axis=-1)
    correct = (pred == sequence_tokens).astype(jnp.float32) * mask_for_loss
    token_count = jnp.maximum(jnp.sum(mask_for_loss), 1.0)
    return jnp.sum(nll * mask_for_loss), jnp.sum(correct), token_count


def _metric_from_sums(nll_sum: float, correct_sum: float, token_count: float) -> dict[str, float]:
    nll = nll_sum / max(token_count, 1e-8)
    ppl = math.exp(min(20.0, nll))
    acc = correct_sum / max(token_count, 1e-8)
    return {"nll": nll, "perplexity": ppl, "accuracy": acc}


def _summarize_examples(tag: str, examples: list[GraphExample]) -> None:
    lengths = np.asarray([ex.sequence_tokens.shape[0] for ex in examples], dtype=np.float32)
    token_counts = np.asarray([float(np.sum(ex.residue_mask > 0.5)) for ex in examples], dtype=np.float32)
    if lengths.size == 0:
        print(f"{tag}: no examples")
        return
    print(
        f"{tag}: n={int(lengths.size)} "
        f"len[min/med/p90/max]={float(np.min(lengths)):.0f}/{float(np.median(lengths)):.0f}/"
        f"{float(np.percentile(lengths, 90)):.0f}/{float(np.max(lengths)):.0f} "
        f"tokens_total={int(np.sum(token_counts))}"
    )


def _summarize_examples_dict(examples: list[GraphExample]) -> dict[str, float]:
    lengths = np.asarray([ex.sequence_tokens.shape[0] for ex in examples], dtype=np.float32)
    token_counts = np.asarray([float(np.sum(ex.residue_mask > 0.5)) for ex in examples], dtype=np.float32)
    if lengths.size == 0:
        return {"n_examples": 0.0, "tokens_total": 0.0}
    return {
        "n_examples": float(lengths.size),
        "tokens_total": float(np.sum(token_counts)),
        "len_min": float(np.min(lengths)),
        "len_median": float(np.median(lengths)),
        "len_p90": float(np.percentile(lengths, 90)),
        "len_max": float(np.max(lengths)),
    }


def _merge_nested(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested(merged[key], value)
        else:
            merged[key] = value
    return merged


def _default_config() -> dict[str, Any]:
    return {
        "data": {
            "path": "/Users/thom/Projects/references/ProteinMPNN/training/small_dataset/pdb_2021aug02_sample",
            "max_examples": 64,
            "max_length": 256,
        },
        "training": {
            "output_dir": "./outputs_jax_train",
            "num_epochs": 1,
            "tokens_per_batch": 2048,
            "warmup_steps": 4000,
            "clip_grad_norm": 1.0,
            "seed": 0,
            "precision": "float32",
            "dropout": 0.1,
        },
        "model": {
            "hidden_dim": 128,
            "edge_dim": 128,
            "num_encoder_layers": 3,
            "num_decoder_layers": 3,
            "k_neighbors": 48,
        },
        "wandb": {
            "enabled": False,
            "entity": "",
            "project": "",
            "run_id": "",
            "run_name": "",
            "mode": "online",
            "tags": [],
        },
    }


def _load_config(path: Path) -> dict[str, Any]:
    defaults = _default_config()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError("Config must be a YAML mapping at top level.")
    config = _merge_nested(defaults, loaded)
    return config


def _init_wandb(config: dict[str, Any], output_dir: Path):
    wb = config["wandb"]
    if not bool(wb.get("enabled", False)):
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError("wandb is enabled in config but the package is not installed.") from exc

    project = wb.get("project", "").strip()
    if not project:
        raise ValueError("wandb.project must be set when wandb.enabled=true.")
    entity = wb.get("entity", "").strip() or None
    run_id = wb.get("run_id", "").strip() or None
    run_name = wb.get("run_name", "").strip() or None
    mode = wb.get("mode", "online")
    tags = wb.get("tags", [])
    return wandb.init(
        project=project,
        entity=entity,
        id=run_id,
        name=run_name,
        resume="allow" if run_id else None,
        mode=mode,
        tags=tags,
        config=config,
        dir=str(output_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train_jax.yaml")
    args = parser.parse_args()

    config = _load_config(Path(args.config))
    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]

    dtype = jnp.float32 if train_cfg["precision"] == "float32" else jnp.bfloat16
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    wandb_run = _init_wandb(config, output_dir)

    train_examples, valid_examples, dataset_mode = load_examples(
        data_path=Path(data_cfg["path"]),
        edge_dim=model_cfg["edge_dim"],
        k_neighbors=model_cfg["k_neighbors"],
        max_length=data_cfg["max_length"],
        max_examples=data_cfg["max_examples"],
        seed=train_cfg["seed"],
    )

    model = ProteinMPNN(
        num_letters=len(ALPHABET),
        node_features=model_cfg["hidden_dim"],
        edge_features=model_cfg["edge_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        rngs=nnx.Rngs(train_cfg["seed"]),
        num_encoder_layers=model_cfg["num_encoder_layers"],
        num_decoder_layers=model_cfg["num_decoder_layers"],
        k_neighbors=model_cfg["k_neighbors"],
        dropout=train_cfg["dropout"],
        augment_eps=0.0,
        ca_only=False,
    )
    optimizer = build_optimizer(
        model=model,
        hidden_dim=model_cfg["hidden_dim"],
        warmup_steps=train_cfg["warmup_steps"],
        clip_grad_norm=train_cfg["clip_grad_norm"],
    )

    rng = np.random.default_rng(train_cfg["seed"])
    jax_rng = jax.random.key(train_cfg["seed"])
    history = []
    print(f"JAX backend: {jax.default_backend()} | devices: {jax.device_count()}")
    print(f"Loaded train={len(train_examples)} valid={len(valid_examples)} examples via mode={dataset_mode}")
    _summarize_examples("train_dist", train_examples)
    _summarize_examples("valid_dist", valid_examples)
    if wandb_run is not None:
        wandb_run.summary.update(
            {
                "dataset_mode": dataset_mode,
                "train_examples": len(train_examples),
                "valid_examples": len(valid_examples),
                **{f"train_dist/{k}": v for k, v in _summarize_examples_dict(train_examples).items()},
                **{f"valid_dist/{k}": v for k, v in _summarize_examples_dict(valid_examples).items()},
            }
        )

    for epoch in range(1, train_cfg["num_epochs"] + 1):
        train_batches = _token_batches(train_examples, train_cfg["tokens_per_batch"], rng, shuffle=True)
        train_nll_sum = 0.0
        train_correct_sum = 0.0
        train_token_sum = 0.0
        for examples in train_batches:
            batch = _collate_batch(
                examples=examples,
                edge_dim=model_cfg["edge_dim"],
                k_neighbors=model_cfg["k_neighbors"],
                dtype=dtype,
            )
            jax_rng, subkey = jax.random.split(jax_rng)
            decode_noise = jnp.abs(jax.random.normal(subkey, shape=batch["decode_mask"].shape, dtype=jnp.float32))
            nll_sum, correct_sum, token_count = train_step(
                model=model,
                optimizer=optimizer,
                edge_features=batch["edge_features"],
                neighbor_idx=batch["neighbor_idx"],
                sequence_tokens=batch["sequence_tokens"],
                residue_mask=batch["residue_mask"],
                decode_mask=batch["decode_mask"],
                decode_noise=decode_noise,
            )
            train_nll_sum += float(nll_sum)
            train_correct_sum += float(correct_sum)
            train_token_sum += float(token_count)

        valid_batches = _token_batches(valid_examples, train_cfg["tokens_per_batch"], rng, shuffle=False)
        valid_nll_sum = 0.0
        valid_correct_sum = 0.0
        valid_token_sum = 0.0
        for examples in valid_batches:
            batch = _collate_batch(
                examples=examples,
                edge_dim=model_cfg["edge_dim"],
                k_neighbors=model_cfg["k_neighbors"],
                dtype=dtype,
            )
            jax_rng, subkey = jax.random.split(jax_rng)
            decode_noise = jnp.abs(jax.random.normal(subkey, shape=batch["decode_mask"].shape, dtype=jnp.float32))
            nll_sum, correct_sum, token_count = eval_step(
                model=model,
                edge_features=batch["edge_features"],
                neighbor_idx=batch["neighbor_idx"],
                sequence_tokens=batch["sequence_tokens"],
                residue_mask=batch["residue_mask"],
                decode_mask=batch["decode_mask"],
                decode_noise=decode_noise,
            )
            valid_nll_sum += float(nll_sum)
            valid_correct_sum += float(correct_sum)
            valid_token_sum += float(token_count)

        train_metrics = _metric_from_sums(train_nll_sum, train_correct_sum, train_token_sum)
        valid_metrics = _metric_from_sums(valid_nll_sum, valid_correct_sum, valid_token_sum)
        report = {
            "epoch": epoch,
            "train": train_metrics,
            "valid": valid_metrics,
        }
        history.append(report)
        print(
            f"epoch={epoch} "
            f"train_ppl={train_metrics['perplexity']:.3f} train_acc={train_metrics['accuracy']:.3f} "
            f"valid_ppl={valid_metrics['perplexity']:.3f} valid_acc={valid_metrics['accuracy']:.3f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/nll": train_metrics["nll"],
                    "train/perplexity": train_metrics["perplexity"],
                    "train/accuracy": train_metrics["accuracy"],
                    "valid/nll": valid_metrics["nll"],
                    "valid/perplexity": valid_metrics["perplexity"],
                    "valid/accuracy": valid_metrics["accuracy"],
                }
            )

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Wrote metrics to {history_path}")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
