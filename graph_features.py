import flax.nnx as nnx

from features import CA_ProteinFeatures, ProteinFeatures


class ProteinMPNNFeaturizer(nnx.Module):
    """Standalone graph featurizer for ProteinMPNN.

    This module is intended to be run outside the training step so that
    training forward passes can consume precomputed graph tensors.
    """

    def __init__(
        self,
        node_features,
        edge_features,
        rngs: nnx.Rngs,
        k_neighbors=64,
        augment_eps=0.05,
        ca_only=False,
    ):
        self.ca_only = ca_only
        if ca_only:
            self.features = CA_ProteinFeatures(
                node_features=node_features,
                edge_features=edge_features,
                rngs=rngs,
                top_k=k_neighbors,
                augment_eps=augment_eps,
            )
        else:
            self.features = ProteinFeatures(
                node_features=node_features,
                edge_features=edge_features,
                rngs=rngs,
                top_k=k_neighbors,
                augment_eps=augment_eps,
            )

    def __call__(self, atom_coords, residue_mask, residue_idx, chain_ids):
        return self.features(atom_coords, residue_mask, residue_idx, chain_ids)
