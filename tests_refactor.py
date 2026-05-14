import jax
import jax.numpy as jnp
import flax.nnx as nnx

from graph_features import ProteinMPNNFeaturizer
from model import ProteinMPNN


def _make_toy_batch(batch_size=1, num_residues=6, vocab=21):
    atom_coords = jax.random.normal(jax.random.key(0), (batch_size, num_residues, 4, 3))
    sequence_tokens = jax.random.randint(
        jax.random.key(1), (batch_size, num_residues), 0, vocab
    ).astype(jnp.int32)
    residue_mask = jnp.ones((batch_size, num_residues), dtype=jnp.float32)
    decode_mask = jnp.ones((batch_size, num_residues), dtype=jnp.float32)
    residue_idx = jnp.arange(num_residues, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0)
    chain_ids = jnp.zeros((batch_size, num_residues), dtype=jnp.int32)
    decode_noise = jnp.abs(jax.random.normal(jax.random.key(2), (batch_size, num_residues)))
    return {
        "atom_coords": atom_coords,
        "sequence_tokens": sequence_tokens,
        "residue_mask": residue_mask,
        "decode_mask": decode_mask,
        "residue_idx": residue_idx,
        "chain_ids": chain_ids,
        "decode_noise": decode_noise,
    }


def test_shapes_and_determinism():
    batch = _make_toy_batch()
    model = ProteinMPNN(
        num_letters=21,
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        rngs=nnx.Rngs(0),
        num_encoder_layers=2,
        num_decoder_layers=1,
        dropout=0.0,
        augment_eps=0.0,
        ca_only=False,
    )
    featurizer = ProteinMPNNFeaturizer(
        node_features=128,
        edge_features=128,
        rngs=nnx.Rngs(123),
        k_neighbors=64,
        ca_only=False,
    )
    edge_features, neighbor_idx = featurizer(
        atom_coords=batch["atom_coords"],
        residue_mask=batch["residue_mask"],
        residue_idx=batch["residue_idx"],
        chain_ids=batch["chain_ids"],
    )

    log_probs_1 = model(
        edge_features,
        neighbor_idx,
        batch["sequence_tokens"],
        batch["residue_mask"],
        batch["decode_mask"],
        batch["decode_noise"],
    )
    log_probs_2 = model(
        edge_features,
        neighbor_idx,
        batch["sequence_tokens"],
        batch["residue_mask"],
        batch["decode_mask"],
        batch["decode_noise"],
    )
    assert log_probs_1.shape == (1, 6, 21)
    assert jnp.allclose(log_probs_1, log_probs_2)

    sampled = model.sample(
        atom_coords=batch["atom_coords"],
        decode_noise=batch["decode_noise"],
        true_sequence=batch["sequence_tokens"],
        chain_mask=batch["decode_mask"],
        chain_ids=batch["chain_ids"],
        residue_idx=batch["residue_idx"],
        mask=batch["residue_mask"],
        sample_key=jax.random.key(3),
    )
    assert sampled["S"].shape == (1, 6)
    assert sampled["probs"].shape == (1, 6, 21)
    assert sampled["decoding_order"].shape == (1, 6)

    conditional = model.conditional_probs(
        atom_coords=batch["atom_coords"],
        sequence_tokens=batch["sequence_tokens"],
        residue_mask=batch["residue_mask"],
        decode_mask=batch["decode_mask"],
        residue_idx=batch["residue_idx"],
        chain_ids=batch["chain_ids"],
        decode_noise=batch["decode_noise"],
    )
    unconditional = model.unconditional_probs(
        atom_coords=batch["atom_coords"],
        residue_mask=batch["residue_mask"],
        residue_idx=batch["residue_idx"],
        chain_ids=batch["chain_ids"],
    )
    assert conditional.shape == (1, 6, 21)
    assert unconditional.shape == (1, 6, 21)


if __name__ == "__main__":
    test_shapes_and_determinism()
    print("tests_refactor: OK")
