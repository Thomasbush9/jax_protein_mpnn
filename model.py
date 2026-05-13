import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx

from features import CA_ProteinFeatures, ProteinFeatures, PositionalEncodings
from layers import EncLayer, DecLayer, PositionWiseFeedForward, cat_neighbors_nodes


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
    def __init__(self):



        pass
    def __call__(self, x):
        pass
