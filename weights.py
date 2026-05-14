from __future__ import annotations

import jax.numpy as jnp
import torch


def _to_jax(array):
    return jnp.asarray(array.detach().cpu().numpy())


def _set_linear(linear_module, weight, bias=None):
    linear_module.kernel[...] = _to_jax(weight.T)
    if hasattr(linear_module, "bias") and bias is not None:
        linear_module.bias[...] = _to_jax(bias)


def _set_layer_norm(norm_module, weight, bias):
    norm_module.scale[...] = _to_jax(weight)
    norm_module.bias[...] = _to_jax(bias)


def _load_feature_weights(state_dict, feature_module):
    _set_linear(
        feature_module.embeddings.linear,
        state_dict["features.embeddings.linear.weight"],
        state_dict["features.embeddings.linear.bias"],
    )
    _set_linear(
        feature_module.edge_embedding,
        state_dict["features.edge_embedding.weight"],
        bias=None,
    )
    _set_layer_norm(
        feature_module.norm_edges,
        state_dict["features.norm_edges.weight"],
        state_dict["features.norm_edges.bias"],
    )


def load_torch_vanilla_checkpoint(checkpoint_path, model, featurizer=None):
    """Load a vanilla ProteinMPNN .pt checkpoint into NNX modules."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    if featurizer is not None:
        _load_feature_weights(state_dict, featurizer.features)
    _load_feature_weights(state_dict, model.features)

    _set_linear(model.W_e, state_dict["W_e.weight"], state_dict["W_e.bias"])
    model.W_s.embedding[...] = _to_jax(state_dict["W_s.weight"])
    _set_linear(model.W_out, state_dict["W_out.weight"], state_dict["W_out.bias"])

    for i, enc in enumerate(model.encoder_layers.encoder_stack):
        prefix = f"encoder_layers.{i}"
        _set_layer_norm(enc.norm1, state_dict[f"{prefix}.norm1.weight"], state_dict[f"{prefix}.norm1.bias"])
        _set_layer_norm(enc.norm2, state_dict[f"{prefix}.norm2.weight"], state_dict[f"{prefix}.norm2.bias"])
        _set_layer_norm(enc.norm3, state_dict[f"{prefix}.norm3.weight"], state_dict[f"{prefix}.norm3.bias"])
        _set_linear(enc.W1, state_dict[f"{prefix}.W1.weight"], state_dict[f"{prefix}.W1.bias"])
        _set_linear(enc.W2, state_dict[f"{prefix}.W2.weight"], state_dict[f"{prefix}.W2.bias"])
        _set_linear(enc.W3, state_dict[f"{prefix}.W3.weight"], state_dict[f"{prefix}.W3.bias"])
        _set_linear(enc.W11, state_dict[f"{prefix}.W11.weight"], state_dict[f"{prefix}.W11.bias"])
        _set_linear(enc.W12, state_dict[f"{prefix}.W12.weight"], state_dict[f"{prefix}.W12.bias"])
        _set_linear(enc.W13, state_dict[f"{prefix}.W13.weight"], state_dict[f"{prefix}.W13.bias"])
        _set_linear(
            enc.dense.W_in,
            state_dict[f"{prefix}.dense.W_in.weight"],
            state_dict[f"{prefix}.dense.W_in.bias"],
        )
        _set_linear(
            enc.dense.W_out,
            state_dict[f"{prefix}.dense.W_out.weight"],
            state_dict[f"{prefix}.dense.W_out.bias"],
        )

    for i, dec in enumerate(model.decoder_layers.decoder_layers):
        prefix = f"decoder_layers.{i}"
        _set_layer_norm(dec.norm1, state_dict[f"{prefix}.norm1.weight"], state_dict[f"{prefix}.norm1.bias"])
        _set_layer_norm(dec.norm2, state_dict[f"{prefix}.norm2.weight"], state_dict[f"{prefix}.norm2.bias"])
        _set_linear(dec.W1, state_dict[f"{prefix}.W1.weight"], state_dict[f"{prefix}.W1.bias"])
        _set_linear(dec.W2, state_dict[f"{prefix}.W2.weight"], state_dict[f"{prefix}.W2.bias"])
        _set_linear(dec.W3, state_dict[f"{prefix}.W3.weight"], state_dict[f"{prefix}.W3.bias"])
        _set_linear(
            dec.dense.W_in,
            state_dict[f"{prefix}.dense.W_in.weight"],
            state_dict[f"{prefix}.dense.W_in.bias"],
        )
        _set_linear(
            dec.dense.W_out,
            state_dict[f"{prefix}.dense.W_out.weight"],
            state_dict[f"{prefix}.dense.W_out.bias"],
        )

    return checkpoint
