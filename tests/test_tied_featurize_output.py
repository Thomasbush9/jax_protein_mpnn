import jax.numpy as jnp

from protein_mpnn_utils import TiedFeaturizeOutput, tied_featurize


def _toy_entry():
    return {
        "name": "toy",
        "seq": "AC",
        "seq_chain_A": "AC",
        "coords_chain_A": {
            "N_chain_A": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            "CA_chain_A": [[0.2, 0.0, 0.0], [1.2, 0.0, 0.0]],
            "C_chain_A": [[0.4, 0.0, 0.0], [1.4, 0.0, 0.0]],
            "O_chain_A": [[0.6, 0.0, 0.0], [1.6, 0.0, 0.0]],
        },
    }


def test_tied_featurize_dataclass_and_legacy_tuple_access():
    out = tied_featurize(batch=[_toy_entry()], chain_dict=None, ca_only=False)

    assert isinstance(out, TiedFeaturizeOutput)
    assert out.X.shape == (1, 2, 4, 3)
    assert out.S.shape == (1, 2)
    assert out.lengths.tolist() == [2]
    assert out.masked_list_list == [["A"]]
    assert out.visible_list_list == [[]]

    # Legacy compatibility: unpacking + indexing still works.
    X, S, mask, *_ = out
    assert jnp.array_equal(X, out.X)
    assert jnp.array_equal(S, out.S)
    assert jnp.array_equal(mask, out.mask)
    assert out[0].shape == out.X.shape
    assert out[13].shape == out.dihedral_mask.shape
    assert len(out) == 20
