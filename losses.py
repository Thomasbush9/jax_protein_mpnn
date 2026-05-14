import jax
import jax.numpy as jnp


def loss_nll_jax(sequence_tokens, log_probs, residue_mask):
    """Negative log-likelihood loss on masked residues."""
    loss = -jnp.take_along_axis(
        log_probs.reshape(-1, log_probs.shape[-1]),
        sequence_tokens.reshape(-1, 1),
        axis=-1,
    ).reshape(sequence_tokens.shape)
    loss_av = jnp.sum(loss * residue_mask) / jnp.sum(residue_mask)
    return loss, loss_av


def loss_smoothed_jax(sequence_tokens, log_probs, residue_mask, num_classes=21, weight=0.1):
    """Label-smoothed loss on masked residues."""
    sequence_onehot = jax.nn.one_hot(sequence_tokens, num_classes=num_classes)
    sequence_onehot = sequence_onehot + weight / sequence_onehot.shape[-1]
    sequence_onehot = sequence_onehot / sequence_onehot.sum(-1, keepdims=True)
    loss = -(sequence_onehot * log_probs).sum(-1)
    loss_av = jnp.sum(loss * residue_mask) / jnp.sum(residue_mask)
    return loss, loss_av
