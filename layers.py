import jax
import jax.numpy as jnp
import flax.nnx as nnx

from features import gather_nodes


def cat_neighbors_nodes(node_hidden, neighbor_hidden, neighbor_idx):
    gathered_node_hidden = gather_nodes(node_hidden, neighbor_idx)
    concatenated_hidden = jnp.concatenate([neighbor_hidden, gathered_node_hidden], axis=-1)
    return concatenated_hidden


class PositionWiseFeedForward(nnx.Module):
    def __init__(self, num_hidden, num_ff, rngs:nnx.Rngs):
        kernel_init = jax.nn.initializers.glorot_uniform()
        bias_init = jax.nn.initializers.zeros
        self.W_in = nnx.Linear(
            num_hidden, num_ff, use_bias=True, rngs=rngs,
            kernel_init=kernel_init, bias_init=bias_init
        )
        self.W_out = nnx.Linear(
            num_ff, num_hidden, use_bias=True, rngs=rngs,
            kernel_init=kernel_init, bias_init=bias_init
        )
        self.act = jax.nn.gelu

    def __call__(self, node_hidden):
        hidden = self.act(self.W_in(node_hidden))
        hidden = self.W_out(hidden)
        return hidden


#=== Encoder-Decoder Layers
class EncLayer(nnx.Module):
    def __init__(self, num_hidden, num_in, rngs: nnx.Rngs, dropout=.01, num_heads=None, scale=30) -> None:
        super().__init__()
        kernel_init = jax.nn.initializers.glorot_uniform()
        bias_init = jax.nn.initializers.zeros
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        self.dropout1 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout3 = nnx.Dropout(dropout, rngs=rngs)

        self.norm1 = nnx.LayerNorm(num_hidden, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_hidden, rngs=rngs)
        self.norm3 = nnx.LayerNorm(num_hidden, rngs=rngs)

        self.W1 = nnx.Linear(
            num_hidden + num_in, num_hidden, use_bias=True, rngs=rngs,
            kernel_init=kernel_init, bias_init=bias_init
        )
        self.W2 = nnx.Linear(
            num_hidden, num_hidden, use_bias=True, rngs=rngs,
            kernel_init=kernel_init, bias_init=bias_init
        )
        self.W3 = nnx.Linear(
            num_hidden, num_hidden, use_bias=True, rngs=rngs,
            kernel_init=kernel_init, bias_init=bias_init
        )

        self.W11 = nnx.Linear(
            num_hidden + num_in, num_hidden, use_bias=True, rngs=rngs,
            kernel_init=kernel_init, bias_init=bias_init
        )
        self.W12 = nnx.Linear(
            num_hidden, num_hidden, use_bias=True, rngs=rngs,
            kernel_init=kernel_init, bias_init=bias_init
        )
        self.W13 = nnx.Linear(
            num_hidden, num_hidden, use_bias=True, rngs=rngs,
            kernel_init=kernel_init, bias_init=bias_init
        )
        self.act = jax.nn.gelu
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden*4, rngs=rngs) #TODO: check which layer add


    def __call__(self, node_hidden, edge_hidden, neighbor_idx, node_mask=None, attend_mask=None):
        '''Parallel computation of full transformer layer'''

        node_edge_hidden = cat_neighbors_nodes(node_hidden, edge_hidden, neighbor_idx)
        node_edge_shape = node_edge_hidden.shape
        node_shape = node_hidden.shape
        expanded_node_shape = node_shape[:2] + (node_edge_shape[2],) + node_shape[2:]
        node_hidden_expanded = jnp.broadcast_to(node_hidden[:, :, None, ...], expanded_node_shape)
        node_edge_hidden = jnp.concatenate([node_hidden_expanded, node_edge_hidden], axis=-1)
        message_hidden = self.W3(self.act(self.W2(self.act(self.W1(node_edge_hidden)))))
        if attend_mask is not None:
            message_hidden = attend_mask[..., None] * message_hidden
        node_delta = jnp.sum(message_hidden, axis=-2) / self.scale
        node_hidden = self.norm1(node_hidden + self.dropout1(node_delta))

        node_delta = self.dense(node_hidden)
        node_hidden = self.norm2(node_hidden + self.dropout2(node_delta))

        if node_mask is not None:
            node_mask = node_mask[..., None]
            node_hidden = node_mask * node_hidden
        node_edge_hidden = cat_neighbors_nodes(node_hidden, edge_hidden, neighbor_idx)
        node_hidden_expanded = jnp.broadcast_to(node_hidden[..., None], expanded_node_shape)
        node_edge_hidden = jnp.concatenate([node_hidden_expanded, node_edge_hidden], axis=-1)
        message_hidden = self.W13(self.act(self.W12(self.act(self.W11(node_edge_hidden)))))
        edge_hidden = self.norm3(edge_hidden + self.dropout3(message_hidden))
        return node_hidden, edge_hidden

class DecLayer(nnx.Module):
    def __init__(self, num_hidden, num_in, rngs: nnx.Rngs, dropout=.01, num_heads=None, scale=30) -> None:
        kernel_init = jax.nn.initializers.glorot_uniform()
        bias_init = jax.nn.initializers.zeros
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        self.dropout1 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout, rngs=rngs)

        self.norm1 = nnx.LayerNorm(num_hidden, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_hidden, rngs=rngs)

        self.W1 = nnx.Linear(
            num_hidden + num_in, num_hidden, use_bias=True, rngs=rngs,
            kernel_init=kernel_init, bias_init=bias_init
        )
        self.W2 = nnx.Linear(
            num_hidden, num_hidden, use_bias=True, rngs=rngs,
            kernel_init=kernel_init, bias_init=bias_init
        )
        self.W3 = nnx.Linear(
            num_hidden, num_hidden, use_bias=True, rngs=rngs,
            kernel_init=kernel_init, bias_init=bias_init
        )
        self.act = jax.nn.gelu
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden*4, rngs=rngs)

    def __call__(self, node_hidden, edge_hidden, node_mask=None, attend_mask=None):
        '''Parallel computation of full transformer layer'''

        node_shape = node_hidden.shape
        edge_shape = edge_hidden.shape
        expanded_node_shape = node_shape[:2] + (edge_shape[2],) + node_shape[2:]

        node_hidden_expanded = jnp.broadcast_to(node_hidden[..., None], expanded_node_shape)

        node_edge_hidden = jnp.concatenate([node_hidden_expanded, edge_hidden], axis=-1)

        message_hidden = self.W3(self.act(self.W2(self.act(self.W1(node_edge_hidden)))))
        if attend_mask is not None:
            message_hidden = attend_mask[..., None] * message_hidden
        node_delta = jnp.sum(message_hidden, axis=-2) / self.scale

        node_hidden = self.norm1(node_hidden + self.dropout1(node_delta))

        # position-wise feed forward:
        node_delta = self.dense(node_hidden)
        node_hidden = self.norm2(node_hidden + self.dropout2(node_delta))

        if node_mask is not None:
            node_hidden = node_mask[..., None] * node_hidden
        return node_hidden


class EncoderStack(nnx.Module):
    def __init__(self, num_layers: int, hidden_dim: int, dropout: float, rngs: nnx.Rngs) -> None:
        self.encoder_stack = [
            EncLayer(hidden_dim, hidden_dim * 2, dropout=dropout, rngs=rngs)
            for _ in range(num_layers)
        ]

    def __call__(self, node_hidden, edge_hidden, neighbor_idx, attend_mask, node_mask):
        for layer in self.encoder_stack:
            node_hidden, edge_hidden = layer(
                edge_hidden=edge_hidden,
                node_hidden=node_hidden,
                neighbor_idx=neighbor_idx,
                attend_mask=attend_mask,
                node_mask=node_mask,
            )
        return node_hidden, edge_hidden


class DecoderStack(nnx.Module):
    def __init__(self, num_layers: int, hidden_dim: int, dropout: float, rngs: nnx.Rngs) -> None:
        self.decoder_layers = [
            DecLayer(hidden_dim, hidden_dim * 3, dropout=dropout, rngs=rngs)
            for _ in range(num_layers)
        ]

    def __call__(self, node_hidden, edge_hidden, node_mask=None):
        for layer in self.decoder_layers:
            node_hidden = layer(node_hidden=node_hidden, edge_hidden=edge_hidden, node_mask=node_mask)
        return node_hidden
