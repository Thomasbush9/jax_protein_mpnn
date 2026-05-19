import jax
import jax.numpy as jnp
import flax.nnx as nnx

from features import CA_ProteinFeatures, ProteinFeatures, gather_nodes
from layers import EncoderStack, DecoderStack, cat_neighbors_nodes
from losses import loss_nll_jax, loss_smoothed_jax
from decoding import (
    sample_decode,
    tied_sample_decode,
    conditional_probs_decode,
    unconditional_probs_decode,
)
from tqdm import tqdm
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
        edge_features,
        neighbor_idx,
        sequence_tokens,
        residue_mask,
        decode_mask,
        decode_noise,
        use_input_decoding_order=False,
        decoding_order=None,
    ):
        """Graph-conditioned sequence model training forward pass.

        Args:
            edge_features: [B, N, K, edge_features]
            neighbor_idx: [B, N, K]
            sequence_tokens: [B, N]
            residue_mask: [B, N]
            decode_mask: [B, N]
            decode_noise: [B, N]
        """
        if sequence_tokens.ndim != 2:
            raise ValueError("sequence_tokens must have shape [B, N]")
        if residue_mask.shape != sequence_tokens.shape:
            raise ValueError("residue_mask shape must match sequence_tokens shape")
        encoded_node_hidden, encoded_edge_hidden = self._encode_graph(
            edge_features=edge_features,
            neighbor_idx=neighbor_idx,
            residue_mask=residue_mask,
        )
        sequence_hidden = self.W_s(sequence_tokens)
        seq_edge_hidden = cat_neighbors_nodes(sequence_hidden, encoded_edge_hidden, neighbor_idx)
        encoder_node_edge_context = self._build_encoder_context(
            encoded_node_hidden=encoded_node_hidden,
            encoded_edge_hidden=encoded_edge_hidden,
            neighbor_idx=neighbor_idx,
            sequence_hidden_template=sequence_hidden,
        )
        decoding_order, decoder_backward_mask, decoder_forward_mask = self._build_decoder_masks(
            neighbor_idx=neighbor_idx,
            residue_mask=residue_mask,
            decode_mask=decode_mask,
            decode_noise=decode_noise,
            use_input_decoding_order=use_input_decoding_order,
            decoding_order=decoding_order,
        )
        del decoding_order
        encoder_context_forward_masked = decoder_forward_mask * encoder_node_edge_context
        decoded_node_hidden = self._run_decoder(
            node_hidden_init=encoded_node_hidden,
            seq_edge_hidden=seq_edge_hidden,
            neighbor_idx=neighbor_idx,
            decoder_backward_mask=decoder_backward_mask,
            encoder_context_forward_masked=encoder_context_forward_masked,
            residue_mask=residue_mask,
        )
        logits = self.W_out(decoded_node_hidden)
        return jax.nn.log_softmax(logits, axis=-1)

    def _encode_graph(self, edge_features, neighbor_idx, residue_mask):
        """Encode structure graph into node/edge hidden states."""
        if edge_features.shape[-1] != self.edge_features:
            raise ValueError(
                f"Expected edge feature dim {self.edge_features}, got {edge_features.shape[-1]}"
            )
        if neighbor_idx.ndim != 3:
            raise ValueError("neighbor_idx must have shape [B, N, K]")
        node_hidden = jnp.zeros(
            (edge_features.shape[0], edge_features.shape[1], self.hidden_dim),
            dtype=edge_features.dtype,
        )
        edge_hidden = self.W_e(edge_features)
        attend_mask = gather_nodes(residue_mask[..., None], neighbor_idx).squeeze(axis=-1)
        attend_mask = residue_mask[..., None] * attend_mask
        encoded_node_hidden, encoded_edge_hidden = self.encoder_layers(
            node_hidden, edge_hidden, neighbor_idx, attend_mask, residue_mask
        )
        return encoded_node_hidden, encoded_edge_hidden

    def _encode_graph_from_structure(self, atom_coords, residue_mask, residue_idx, chain_ids):
        """Compatibility path for decode APIs that still accept raw structure inputs."""
        edge_features, neighbor_idx = self.features(atom_coords, residue_mask, residue_idx, chain_ids)
        encoded_node_hidden, encoded_edge_hidden = self._encode_graph(
            edge_features=edge_features,
            neighbor_idx=neighbor_idx,
            residue_mask=residue_mask,
        )
        return neighbor_idx, encoded_node_hidden, encoded_edge_hidden

    def _build_decoder_masks(
        self,
        neighbor_idx,
        residue_mask,
        decode_mask,
        decode_noise,
        use_input_decoding_order=False,
        decoding_order=None,
    ):
        """Build autoregressive decoder masks and decode order."""
        decode_mask = decode_mask * residue_mask
        if (not use_input_decoding_order) or (decoding_order is None):
            decoding_order = jnp.argsort((decode_mask + 0.000_1) * jnp.abs(decode_noise), axis=-1)
        mask_size = neighbor_idx.shape[1]
        permutation_matrix_reverse = jax.nn.one_hot(
            decoding_order.astype(jnp.int32), num_classes=mask_size, dtype=jnp.float32
        )
        lower_tri = 1.0 - jnp.triu(jnp.ones((mask_size, mask_size), dtype=jnp.float32))
        order_mask_backward = jnp.einsum(
            "ij,biq,bjp->bqp",
            lower_tri,
            permutation_matrix_reverse,
            permutation_matrix_reverse,
        )
        decoder_attend_mask = jnp.take_along_axis(order_mask_backward, neighbor_idx, axis=2)[..., None]
        node_mask_1d = residue_mask[..., None, None]
        decoder_backward_mask = node_mask_1d * decoder_attend_mask
        decoder_forward_mask = node_mask_1d * (1.0 - decoder_attend_mask)
        return decoding_order, decoder_backward_mask, decoder_forward_mask

    def _build_encoder_context(
        self, encoded_node_hidden, encoded_edge_hidden, neighbor_idx, sequence_hidden_template
    ):
        """Build encoder node-edge context for decoder skip pathway."""
        zero_sequence_hidden = jnp.zeros_like(sequence_hidden_template)
        encoder_edge_context = cat_neighbors_nodes(
            zero_sequence_hidden, encoded_edge_hidden, neighbor_idx
        )
        return cat_neighbors_nodes(encoded_node_hidden, encoder_edge_context, neighbor_idx)

    def _run_decoder(
        self,
        node_hidden_init,
        seq_edge_hidden,
        neighbor_idx,
        decoder_backward_mask,
        encoder_context_forward_masked,
        residue_mask,
    ):
        """Run decoder stack over prepared sequence/edge context."""
        decoder_node_hidden = node_hidden_init
        for layer in self.decoder_layers.decoder_layers:
            decoder_context = cat_neighbors_nodes(decoder_node_hidden, seq_edge_hidden, neighbor_idx)
            decoder_context = decoder_backward_mask * decoder_context + encoder_context_forward_masked
            decoder_node_hidden = layer(
                node_hidden=decoder_node_hidden,
                edge_hidden=decoder_context,
                node_mask=residue_mask,
            )
        return decoder_node_hidden


    def sample(self, *args, **kwargs):
        """Autoregressive constrained sequence sampling."""
        return sample_decode(self, *args, **kwargs)




    def tied_sample(self, *args, **kwargs):
        """Autoregressive tied-position constrained sampling."""
        return tied_sample_decode(self, *args, **kwargs)

        

    def conditional_probs(self, *args, **kwargs):
        """Compute per-position conditional log-probabilities."""
        return conditional_probs_decode(self, *args, **kwargs)

    def unconditional_probs(self, *args, **kwargs):
        """Compute unconditional log-probabilities."""
        return unconditional_probs_decode(self, *args, **kwargs)
