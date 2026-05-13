import jax
import jax.numpy as jnp
import flax.nnx as nnx

from features import gather_nodes


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = jnp.concatenate([h_neighbors, h_nodes], axis=-1)
    return h_nn


class PositionWiseFeedForward(nnx.Module):
    def __init__(self, num_hidden, num_ff, rngs:nnx.Rngs):
        self.W_in = nnx.Linear(num_hidden, num_ff, use_bias=True, rngs=rngs)
        self.W_out = nnx.Linear(num_ff, num_hidden, use_bias=True, rngs=rngs)
        self.act = jax.nn.gelu

    def __call__(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


#=== Encoder-Decoder Layers
class EncLayer(nnx.Module):
    def __init__(self, num_hidden, num_in, rngs: nnx.Rngs, dropout=.01, num_heads=None, scale=30) -> None:
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        self.dropout1 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout3 = nnx.Dropout(dropout, rngs=rngs)

        self.norm1 = nnx.LayerNorm(num_hidden, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_hidden, rngs=rngs)
        self.norm3 = nnx.LayerNorm(num_hidden, rngs=rngs)

        self.W1 = nnx.Linear(num_hidden + num_in, num_hidden, use_bias=True, rngs=rngs)
        self.W2 = nnx.Linear(num_hidden, num_hidden, use_bias=True, rngs=rngs)
        self.W3 = nnx.Linear(num_hidden, num_hidden, use_bias=True, rngs=rngs)

        self.W11 = nnx.Linear(num_hidden + num_in, num_hidden, use_bias=True, rngs=rngs)
        self.W12 = nnx.Linear(num_hidden, num_hidden, use_bias=True, rngs=rngs)
        self.W13 = nnx.Linear(num_hidden, num_hidden, use_bias=True, rngs=rngs)
        self.act = jax.nn.gelu
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden*4, rngs=rngs) #TODO: check which layer add


    def __call__(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        '''Parallel computation of full transformer layer'''

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_ev_shape = h_EV.shape
        h_v_shape = h_V.shape
        h_v_exp_shape = h_v_shape[:2] + (h_ev_shape[2],) + h_v_shape[2:]
        h_V_expand = jnp.broadcast_to(h_V[:, :, None, ...], h_v_exp_shape)
        h_EV = jnp.concatenate([h_V_expand, h_EV], axis=-1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend[..., None] * h_message
        dh = jnp.sum(h_message, axis=-2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))


        if mask_V is not None:
            mask_V = mask_V[..., None]
            h_V = mask_V * h_V
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = jnp.broadcast_to(h_V[..., None], h_v_exp_shape)
        h_EV = jnp.concatenate([h_V_expand, h_EV], axis=-1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E

class DecLayer(nnx.Module):
    def __init__(self, num_hidden, num_in, rngs: nnx.Rngs, dropout=.01, num_heads=None, scale=30) -> None:
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale

        self.dropout1 = nnx.Dropout(dropout, rngs=rngs)
        self.dropout2 = nnx.Dropout(dropout, rngs=rngs)

        self.norm1 = nnx.LayerNorm(num_hidden, rngs=rngs)
        self.norm2 = nnx.LayerNorm(num_hidden, rngs=rngs)

        self.W1 = nnx.Linear(num_hidden + num_in, num_hidden, use_bias=True, rngs=rngs)
        self.W2 = nnx.Linear(num_hidden, num_hidden, use_bias=True, rngs=rngs)
        self.W3 = nnx.Linear(num_hidden, num_hidden, use_bias=True, rngs=rngs)
        self.act = jax.nn.gelu
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden*4, rngs=rngs)

    def __call__(self, h_V, h_E, mask_V=None, mask_attend=None):
        '''Parallel computation of full transformer layer'''

        h_v_shape = h_V.shape
        h_e_shape = h_E.shape
        h_v_exp_shape = h_v_shape[:2] + (h_e_shape[2],) + h_v_shape[2:]

        h_V_expand = jnp.broadcast_to(h_V[..., None], h_v_exp_shape)

        h_EV = jnp.concatenate([h_V_expand, h_E], axis=-1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend[..., None] * h_message
        dh = jnp.sum(h_message, axis=-2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # position-wise feed forward:
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            h_V = mask_V[..., None] * h_V
        return h_V
