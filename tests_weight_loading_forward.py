import jax
import jax.numpy as jnp
import flax.nnx as nnx
import numpy as np
from collections import OrderedDict

from graph_features import ProteinMPNNFeaturizer
from model import ProteinMPNN
from protein_mpnn_utils import ALPHABET
from weights import load_torch_vanilla_checkpoint


WEIGHTS_PATH = "/Users/thom/Projects/references/ProteinMPNN/vanilla_model_weights/v_48_020.pt"
PDB_PATH = "/Users/thom/tmp_data/tmp_data/data/predictions/run/sequences/Q9Y6G3/esmfold/structure.pdb"

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _extract_single_chain_backbone(path):
    residues = OrderedDict()
    chain_id = None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in {"N", "CA", "C", "O"}:
                continue
            current_chain = line[21:22]
            if chain_id is None:
                chain_id = current_chain
            if current_chain != chain_id:
                continue

            residue_name = line[17:20].strip()
            residue_number = int(line[22:26].strip())
            insertion_code = line[26:27]
            residue_key = (residue_number, insertion_code)
            if residue_key not in residues:
                residues[residue_key] = {
                    "residue_name": residue_name,
                    "atoms": {},
                }
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            residues[residue_key]["atoms"][atom_name] = np.array([x, y, z], dtype=np.float32)

    if not residues:
        raise ValueError(f"No ATOM backbone records found in {path}")

    atom_coords = []
    sequence_tokens = []
    residue_mask = []
    for residue in residues.values():
        coords = []
        atoms = residue["atoms"]
        has_all_backbone = all(atom in atoms for atom in ("N", "CA", "C", "O"))
        for atom in ("N", "CA", "C", "O"):
            coords.append(atoms.get(atom, np.zeros((3,), dtype=np.float32)))
        atom_coords.append(np.stack(coords, axis=0))
        aa1 = AA3_TO_1.get(residue["residue_name"], "X")
        sequence_tokens.append(ALPHABET.index(aa1))
        residue_mask.append(1.0 if has_all_backbone else 0.0)

    length = len(atom_coords)
    atom_coords = jnp.asarray(np.stack(atom_coords, axis=0)[None, ...], dtype=jnp.float32)
    sequence_tokens = jnp.asarray(np.array(sequence_tokens, dtype=np.int32)[None, ...], dtype=jnp.int32)
    residue_mask = jnp.asarray(np.array(residue_mask, dtype=np.float32)[None, ...], dtype=jnp.float32)
    chain_ids = jnp.zeros((1, length), dtype=jnp.int32)
    residue_idx = jnp.arange(length, dtype=jnp.int32)[None, :]
    return atom_coords, sequence_tokens, residue_mask, chain_ids, residue_idx


def test_weight_loading_and_forward_pass():
    atom_coords, sequence_tokens, residue_mask, chain_ids, residue_idx = _extract_single_chain_backbone(PDB_PATH)
    decode_mask = residue_mask
    decode_noise = jnp.abs(jax.random.normal(jax.random.key(7), shape=decode_mask.shape))

    model = ProteinMPNN(
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        rngs=nnx.Rngs(0),
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=48,
        augment_eps=0.0,
        dropout=0.0,
        ca_only=False,
    )
    featurizer = ProteinMPNNFeaturizer(
        node_features=128,
        edge_features=128,
        rngs=nnx.Rngs(1),
        k_neighbors=48,
        augment_eps=0.0,
        ca_only=False,
    )

    load_torch_vanilla_checkpoint(WEIGHTS_PATH, model=model, featurizer=featurizer)

    edge_features, neighbor_idx = featurizer(
        atom_coords=atom_coords,
        residue_mask=residue_mask,
        residue_idx=residue_idx,
        chain_ids=chain_ids,
    )
    log_probs = model(
        edge_features=edge_features,
        neighbor_idx=neighbor_idx,
        sequence_tokens=sequence_tokens,
        residue_mask=residue_mask,
        decode_mask=decode_mask,
        decode_noise=decode_noise,
    )

    assert log_probs.shape[:2] == sequence_tokens.shape
    assert log_probs.shape[-1] == 21
    assert jnp.all(jnp.isfinite(log_probs))
    probs = jnp.exp(log_probs)
    assert jnp.allclose(jnp.sum(probs, axis=-1), 1.0, atol=1e-4)

    print("weight loading + forward pass: OK")
    print("sequence length:", int(sequence_tokens.shape[1]))
    print("log_probs shape:", tuple(log_probs.shape))


if __name__ == "__main__":
    test_weight_loading_and_forward_pass()
