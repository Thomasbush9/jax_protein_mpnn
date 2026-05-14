import jax
import jax.numpy as jnp
import flax.nnx as nnx

from features import CA_ProteinFeatures, ProteinFeatures, gather_nodes
from layers import EncoderStack, DecoderStack, cat_neighbors_nodes


# ==== Loss functions
def loss_nll_jax(S, log_probs, mask):
    '''Neg Likelihood loss:
    Input:
        S: ground truth: (B, L)
        log_probs: predicted probs (B, L, C)
        mask: mask to positions that we care (B, L )
    '''
    loss = -jnp.take_along_axis(
        log_probs.reshape(-1, log_probs.shape[-1]),
        S.reshape(-1, 1),
        axis=-1
    ).reshape(S.shape)
    loss_av = jnp.sum(loss*mask) / jnp.sum(mask)
    return loss, loss_av

def loss_smoothed_jax(S, log_probs, mask, num_classes=21, weight=0.1):
    S_onehot = jax.nn.one_hot(S, num_classes=num_classes)

    S_onehot = S_onehot + weight / S_onehot.shape[-1]
    S_onehot = S_onehot / S_onehot.sum(-1, keepdims=True)

    loss = - (S_onehot*log_probs).sum(-1)
    loss_av = jnp.sum(loss * mask) / jnp.sum(mask)
    return loss, loss_av

class ProteinMPNN(nnx.Module):
    def __init__(self, num_letters, node_features, edge_features,
                 hidden_dim, rngs: nnx.Rngs, num_encoder_layers=30, 
                 num_decoder_layers=3, vocab=21, k_neighbors=64, augment_eps=0.05,
                 dropout=0.1, ca_only=False):
        kernel_init = jax.nn.initializers.glorot_uniform()
        bias_init = jax.nn.initializers.zeros

        self.node_features= node_features
        self.edge_features= edge_features
        self.hidden_dim= hidden_dim

        if ca_only:
            self.features = CA_ProteinFeatures(
                node_features=node_features,
                edge_features=edge_features,
                rngs=rngs,
                top_k=k_neighbors,
                augment_eps=augment_eps,
            )
            self.W_v = nnx.Linear(
                node_features, hidden_dim, rngs=rngs, use_bias=True,
                kernel_init=kernel_init, bias_init=bias_init
            )
        else:
            self.features = ProteinFeatures(
                node_features=node_features,
                edge_features=edge_features,
                rngs=rngs,
                top_k=k_neighbors,
                augment_eps=augment_eps,
            )

        self.W_e = nnx.Linear(
            edge_features, hidden_dim, use_bias=True, rngs=rngs,
            kernel_init=kernel_init, bias_init=bias_init
        )
        self.W_s = nnx.Embed(vocab, hidden_dim, rngs=rngs)

        self.encoder_layers = EncoderStack(num_encoder_layers, hidden_dim, dropout, rngs)
        self.decoder_layers = DecoderStack(num_decoder_layers, hidden_dim, dropout, rngs)

        self.W_out = nnx.Linear(
            hidden_dim, num_letters, use_bias=True, rngs=rngs,
            kernel_init=kernel_init, bias_init=bias_init
        )

    def __call__(
        self,
        atom_coords,
        sequence_tokens,
        residue_mask,
        decode_mask,
        residue_idx,
        chain_ids,
        decode_noise,
        use_input_decoding_order=False,
        decoding_order=None,
    ):
        '''Graph-conditioned sequence model'''
        #TODO: I think that we can move the feature creation to a pre-processing step
        edge_features, neighbor_idx = self.features(atom_coords, residue_mask, residue_idx, chain_ids)
        if edge_features.shape[-1] != self.hidden_dim:
            raise ValueError(f"Expected edge feature dim {self.hidden_dim}, got {edge_features.shape[-1]}")
        node_hidden = jnp.zeros((edge_features.shape[0], edge_features.shape[1], edge_features.shape[-1]), dtype=edge_features.dtype)
        edge_hidden = self.W_e(edge_features)

        # encoder -> unmasked self attention 
        attend_mask = gather_nodes(residue_mask[..., None], neighbor_idx).squeeze(axis=-1)
        attend_mask = residue_mask[..., None] * attend_mask
        
        node_hidden, edge_hidden = self.encoder_layers(node_hidden, edge_hidden, neighbor_idx, attend_mask, residue_mask)
        sequence_hidden = self.W_s(sequence_tokens)
        seq_edge_hidden = cat_neighbors_nodes(sequence_hidden, edge_hidden, neighbor_idx)

        encoder_edge_context = cat_neighbors_nodes(jnp.zeros_like(sequence_hidden), edge_hidden, neighbor_idx)
        encoder_node_edge_context = cat_neighbors_nodes(node_hidden, encoder_edge_context, neighbor_idx)

        decode_mask = decode_mask * residue_mask
        if not use_input_decoding_order:
            decoding_order = jnp.argsort((decode_mask + 0.000_1) * (jnp.abs(decode_noise)))
        mask_size = neighbor_idx.shape[1]
        permutation_matrix_reverse = jax.nn.one_hot(
            decoding_order.astype(jnp.int32), num_classes=mask_size, dtype=jnp.float32
        )
        lower_tri = 1.0 - jnp.triu(jnp.ones((mask_size, mask_size), dtype=permutation_matrix_reverse.dtype))
        order_mask_backward = jnp.einsum(
            "ij,biq,bjp->bqp",
            lower_tri,
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )  # [B, L, L]

        decoder_attend_mask = jnp.take_along_axis(order_mask_backward, neighbor_idx, axis=2)[..., None]  # [B, L, K, 1]
        node_mask_1d = residue_mask[..., None, None]
        decoder_backward_mask = node_mask_1d * decoder_attend_mask
        decoder_forward_mask = node_mask_1d * (1. - decoder_attend_mask)

        encoder_context_forward_masked = decoder_forward_mask * encoder_node_edge_context
        for layer in self.decoder_layers.decoder_layers:
            decoder_node_seq_edge_context = cat_neighbors_nodes(node_hidden, seq_edge_hidden, neighbor_idx)
            decoder_node_seq_edge_context = (
                decoder_backward_mask * decoder_node_seq_edge_context + encoder_context_forward_masked
            )
            node_hidden = layer(node_hidden=node_hidden, edge_hidden=decoder_node_seq_edge_context, node_mask=residue_mask)

        logits = self.W_out(node_hidden)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        return log_probs
