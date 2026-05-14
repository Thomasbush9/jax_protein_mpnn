import jax
import jax.numpy as jnp

from layers import cat_neighbors_nodes


def sample_decode(
    model,
    atom_coords,
    decode_noise,
    true_sequence,
    chain_mask,
    chain_ids,
    residue_idx,
    mask=None,
    temperature=1.0,
    omit_AAs_np=None,
    chain_M_pos=None,
    omit_AA_mask=None,
    pssm_coef=None,
    pssm_bias=None,
    pssm_multi=None,
    pssm_log_odds_flag=None,
    pssm_log_odds_mask=None,
    pssm_bias_flag=None,
    bias_by_res=None,
    sample_key=None,
):
    """Autoregressive constrained decoding."""
    if mask is None:
        raise ValueError("mask must be provided for sampling")
    if chain_M_pos is None:
        chain_M_pos = jnp.ones_like(chain_mask)
    if pssm_multi is None:
        pssm_multi = 0.0
    if pssm_bias_flag is None:
        pssm_bias_flag = False
    if pssm_log_odds_flag is None:
        pssm_log_odds_flag = False

    neighbor_idx, encoded_node_hidden, encoded_edge_hidden = model._encode_graph_from_structure(
        atom_coords=atom_coords,
        residue_mask=mask,
        residue_idx=residue_idx,
        chain_ids=chain_ids,
    )
    design_mask = chain_mask * chain_M_pos * mask
    decoding_order, decoder_backward_mask, decoder_forward_mask = model._build_decoder_masks(
        neighbor_idx=neighbor_idx,
        residue_mask=mask,
        decode_mask=design_mask,
        decode_noise=decode_noise,
    )
    sequence_hidden_template = model.W_s(true_sequence.astype(jnp.int32))
    encoder_node_edge_context = model._build_encoder_context(
        encoded_node_hidden=encoded_node_hidden,
        encoded_edge_hidden=encoded_edge_hidden,
        neighbor_idx=neighbor_idx,
        sequence_hidden_template=sequence_hidden_template,
    )
    encoder_context_forward_masked = decoder_forward_mask * encoder_node_edge_context

    n_batch, n_nodes = atom_coords.shape[:2]
    num_letters = model.W_out.out_features
    sampled_sequence = true_sequence.astype(jnp.int32)
    all_probs = jnp.zeros((n_batch, n_nodes, num_letters), dtype=jnp.float32)

    omit_aa_mask = None
    if omit_AAs_np is not None:
        omit_aa_mask = jnp.asarray(omit_AAs_np, dtype=jnp.float32)

    def gather_2d(x, idx):
        return jnp.take_along_axis(x, idx[:, None], axis=1)[:, 0]

    def gather_3d(x, idx):
        return jnp.take_along_axis(x, idx[:, None, None], axis=1)[:, 0, :]

    for step in range(n_nodes):
        step_positions = decoding_order[:, step].astype(jnp.int32)
        decode_step_mask = gather_2d(design_mask, step_positions)
        true_step_tokens = gather_2d(true_sequence, step_positions).astype(jnp.int32)

        sequence_hidden = model.W_s(sampled_sequence)
        seq_edge_hidden = cat_neighbors_nodes(sequence_hidden, encoded_edge_hidden, neighbor_idx)
        decoder_node_hidden = model._run_decoder(
            node_hidden_init=encoded_node_hidden,
            seq_edge_hidden=seq_edge_hidden,
            neighbor_idx=neighbor_idx,
            decoder_backward_mask=decoder_backward_mask,
            encoder_context_forward_masked=encoder_context_forward_masked,
            residue_mask=mask,
        )
        logits = model.W_out(decoder_node_hidden) / temperature
        step_logits = gather_3d(logits, step_positions)

        if omit_aa_mask is not None:
            step_logits = step_logits - 1e8 * omit_aa_mask[None, :]
        if bias_by_res is not None:
            step_logits = step_logits + gather_3d(bias_by_res, step_positions) / temperature

        step_probs = jax.nn.softmax(step_logits, axis=-1)
        if pssm_bias_flag and pssm_coef is not None and pssm_bias is not None:
            pssm_coef_step = gather_2d(pssm_coef, step_positions)
            pssm_bias_step = gather_3d(pssm_bias, step_positions)
            step_probs = (
                (1.0 - pssm_multi * pssm_coef_step[:, None]) * step_probs
                + pssm_multi * pssm_coef_step[:, None] * pssm_bias_step
            )
        if pssm_log_odds_flag and pssm_log_odds_mask is not None:
            pssm_log_odds_step = gather_3d(pssm_log_odds_mask, step_positions)
            step_probs = step_probs * pssm_log_odds_step + step_probs * 0.001
            step_probs = step_probs / jnp.sum(step_probs, axis=-1, keepdims=True)
        if omit_AA_mask is not None:
            omit_step = gather_3d(omit_AA_mask, step_positions)
            step_probs = step_probs * (1.0 - omit_step)
            step_probs = step_probs / jnp.sum(step_probs, axis=-1, keepdims=True)

        if sample_key is None:
            sampled_step_tokens = jnp.argmax(step_probs, axis=-1).astype(jnp.int32)
        else:
            sample_key, subkey = jax.random.split(sample_key)
            sampled_step_tokens = jax.random.categorical(
                subkey, jnp.log(jnp.clip(step_probs, min=1e-9)), axis=-1
            ).astype(jnp.int32)

        sampled_step_tokens = jnp.where(decode_step_mask > 0.5, sampled_step_tokens, true_step_tokens)
        step_one_hot = jax.nn.one_hot(step_positions, num_classes=n_nodes, dtype=jnp.float32)
        sampled_sequence = jnp.where(step_one_hot.astype(bool), sampled_step_tokens[:, None], sampled_sequence)
        all_probs = all_probs * (1.0 - step_one_hot[..., None]) + (
            decode_step_mask[:, None, None] * step_probs[:, None, :]
        ) * step_one_hot[..., None]

    return {"S": sampled_sequence, "probs": all_probs, "decoding_order": decoding_order}


def tied_sample_decode(
    model,
    atom_coords,
    decode_noise,
    true_sequence,
    chain_mask,
    chain_ids,
    residue_idx,
    mask=None,
    temperature=1.0,
    omit_AAs_np=None,
    bias_AAs_np=None,
    chain_M_pos=None,
    omit_AA_mask=None,
    pssm_coef=None,
    pssm_bias=None,
    pssm_multi=None,
    pssm_log_odds_flag=None,
    pssm_log_odds_mask=None,
    pssm_bias_flag=None,
    tied_pos=None,
    tied_beta=None,
    bias_by_res=None,
    sample_key=None,
):
    """Autoregressive decoding with tied-position groups."""
    if mask is None:
        raise ValueError("mask must be provided for tied sampling")
    if chain_M_pos is None:
        chain_M_pos = jnp.ones_like(chain_mask)
    if pssm_multi is None:
        pssm_multi = 0.0
    if pssm_bias_flag is None:
        pssm_bias_flag = False
    if pssm_log_odds_flag is None:
        pssm_log_odds_flag = False
    if tied_beta is None:
        tied_beta = jnp.ones((atom_coords.shape[1],), dtype=jnp.float32)
    if not tied_pos:
        return sample_decode(
            model=model,
            atom_coords=atom_coords,
            decode_noise=decode_noise,
            true_sequence=true_sequence,
            chain_mask=chain_mask,
            chain_ids=chain_ids,
            residue_idx=residue_idx,
            mask=mask,
            temperature=temperature,
            omit_AAs_np=omit_AAs_np,
            chain_M_pos=chain_M_pos,
            omit_AA_mask=omit_AA_mask,
            pssm_coef=pssm_coef,
            pssm_bias=pssm_bias,
            pssm_multi=pssm_multi,
            pssm_log_odds_flag=pssm_log_odds_flag,
            pssm_log_odds_mask=pssm_log_odds_mask,
            pssm_bias_flag=pssm_bias_flag,
            bias_by_res=bias_by_res,
            sample_key=sample_key,
        )

    neighbor_idx, encoded_node_hidden, encoded_edge_hidden = model._encode_graph_from_structure(
        atom_coords=atom_coords,
        residue_mask=mask,
        residue_idx=residue_idx,
        chain_ids=chain_ids,
    )
    design_mask = chain_mask * chain_M_pos * mask
    base_order = jnp.argsort((design_mask + 0.0001) * jnp.abs(decode_noise), axis=-1)

    order_first = [int(x) for x in jax.device_get(base_order[0])]
    tied_groups = []
    seen = set()
    for position in order_first:
        if position in seen:
            continue
        group = None
        for candidate in tied_pos:
            if position in candidate:
                group = [int(g) for g in candidate]
                break
        if group is None:
            group = [position]
        tied_groups.append(group)
        seen.update(group)

    flattened_order = [p for group in tied_groups for p in group]
    decoding_order = jnp.broadcast_to(
        jnp.asarray(flattened_order, dtype=jnp.int32)[None, :],
        (atom_coords.shape[0], atom_coords.shape[1]),
    )
    _, decoder_backward_mask, decoder_forward_mask = model._build_decoder_masks(
        neighbor_idx=neighbor_idx,
        residue_mask=mask,
        decode_mask=design_mask,
        decode_noise=decode_noise,
        use_input_decoding_order=True,
        decoding_order=decoding_order,
    )

    sequence_hidden_template = model.W_s(true_sequence.astype(jnp.int32))
    encoder_node_edge_context = model._build_encoder_context(
        encoded_node_hidden=encoded_node_hidden,
        encoded_edge_hidden=encoded_edge_hidden,
        neighbor_idx=neighbor_idx,
        sequence_hidden_template=sequence_hidden_template,
    )
    encoder_context_forward_masked = decoder_forward_mask * encoder_node_edge_context

    n_batch, n_nodes = atom_coords.shape[:2]
    num_letters = model.W_out.out_features
    sampled_sequence = true_sequence.astype(jnp.int32)
    all_probs = jnp.zeros((n_batch, n_nodes, num_letters), dtype=jnp.float32)

    omit_aa_mask = jnp.asarray(omit_AAs_np, dtype=jnp.float32) if omit_AAs_np is not None else None
    bias_aa = jnp.asarray(bias_AAs_np, dtype=jnp.float32) if bias_AAs_np is not None else None
    tied_beta = jnp.asarray(tied_beta, dtype=jnp.float32)

    for group in tied_groups:
        group_is_padding = all(bool(jnp.all(mask[:, p] == 0.0)) for p in group)
        if group_is_padding:
            true_group_tokens = true_sequence[:, group[0]].astype(jnp.int32)
            for p in group:
                sampled_sequence = sampled_sequence.at[:, p].set(true_group_tokens)
            continue

        sequence_hidden = model.W_s(sampled_sequence)
        seq_edge_hidden = cat_neighbors_nodes(sequence_hidden, encoded_edge_hidden, neighbor_idx)
        decoder_node_hidden = model._run_decoder(
            node_hidden_init=encoded_node_hidden,
            seq_edge_hidden=seq_edge_hidden,
            neighbor_idx=neighbor_idx,
            decoder_backward_mask=decoder_backward_mask,
            encoder_context_forward_masked=encoder_context_forward_masked,
            residue_mask=mask,
        )
        logits_all = model.W_out(decoder_node_hidden) / temperature
        group_logits = jnp.zeros((n_batch, num_letters), dtype=logits_all.dtype)
        beta_sum = 0.0
        for p in group:
            beta = tied_beta[p]
            group_logits = group_logits + beta * logits_all[:, p, :]
            beta_sum += float(beta)
        group_logits = group_logits / max(beta_sum, 1e-8)

        if omit_aa_mask is not None:
            group_logits = group_logits - 1e8 * omit_aa_mask[None, :]
        if bias_aa is not None:
            group_logits = group_logits + bias_aa[None, :] / temperature
        if bias_by_res is not None:
            group_bias = jnp.zeros_like(group_logits)
            for p in group:
                group_bias = group_bias + bias_by_res[:, p, :]
            group_logits = group_logits + (group_bias / max(len(group), 1)) / temperature

        group_probs = jax.nn.softmax(group_logits, axis=-1)
        if pssm_bias_flag and pssm_coef is not None and pssm_bias is not None:
            pssm_coef_group = jnp.zeros((n_batch,), dtype=jnp.float32)
            pssm_bias_group = jnp.zeros((n_batch, num_letters), dtype=jnp.float32)
            for p in group:
                pssm_coef_group = pssm_coef_group + pssm_coef[:, p]
                pssm_bias_group = pssm_bias_group + pssm_bias[:, p, :]
            pssm_coef_group = pssm_coef_group / max(len(group), 1)
            pssm_bias_group = pssm_bias_group / max(len(group), 1)
            group_probs = (
                (1.0 - pssm_multi * pssm_coef_group[:, None]) * group_probs
                + pssm_multi * pssm_coef_group[:, None] * pssm_bias_group
            )
        if pssm_log_odds_flag and pssm_log_odds_mask is not None:
            pssm_log_odds_group = jnp.zeros((n_batch, num_letters), dtype=jnp.float32)
            for p in group:
                pssm_log_odds_group = pssm_log_odds_group + pssm_log_odds_mask[:, p, :]
            pssm_log_odds_group = pssm_log_odds_group / max(len(group), 1)
            group_probs = group_probs * pssm_log_odds_group + group_probs * 0.001
            group_probs = group_probs / jnp.sum(group_probs, axis=-1, keepdims=True)
        if omit_AA_mask is not None:
            omit_group = jnp.zeros((n_batch, num_letters), dtype=jnp.float32)
            for p in group:
                omit_group = omit_group + omit_AA_mask[:, p, :]
            omit_group = jnp.clip(omit_group, min=0.0, max=1.0)
            group_probs = group_probs * (1.0 - omit_group)
            group_probs = group_probs / jnp.sum(group_probs, axis=-1, keepdims=True)

        if sample_key is None:
            sampled_group_token = jnp.argmax(group_probs, axis=-1).astype(jnp.int32)
        else:
            sample_key, subkey = jax.random.split(sample_key)
            sampled_group_token = jax.random.categorical(
                subkey, jnp.log(jnp.clip(group_probs, min=1e-9)), axis=-1
            ).astype(jnp.int32)

        true_group_token = true_sequence[:, group[0]].astype(jnp.int32)
        group_design_mask = jnp.zeros((n_batch,), dtype=jnp.float32)
        for p in group:
            group_design_mask = jnp.maximum(group_design_mask, design_mask[:, p])
        sampled_group_token = jnp.where(group_design_mask > 0.5, sampled_group_token, true_group_token)
        for p in group:
            sampled_sequence = sampled_sequence.at[:, p].set(sampled_group_token)
            all_probs = all_probs.at[:, p, :].set(group_probs)

    return {"S": sampled_sequence, "probs": all_probs, "decoding_order": decoding_order}


def conditional_probs_decode(
    model,
    atom_coords,
    sequence_tokens,
    residue_mask,
    decode_mask,
    residue_idx,
    chain_ids,
    decode_noise,
    backbone_only=False,
):
    """Compute per-position conditional log-probabilities."""
    neighbor_idx, encoded_node_hidden, encoded_edge_hidden = model._encode_graph_from_structure(
        atom_coords=atom_coords,
        residue_mask=residue_mask,
        residue_idx=residue_idx,
        chain_ids=chain_ids,
    )
    sequence_hidden = model.W_s(sequence_tokens)
    seq_edge_hidden = cat_neighbors_nodes(sequence_hidden, encoded_edge_hidden, neighbor_idx)
    encoder_node_edge_context = model._build_encoder_context(
        encoded_node_hidden=encoded_node_hidden,
        encoded_edge_hidden=encoded_edge_hidden,
        neighbor_idx=neighbor_idx,
        sequence_hidden_template=sequence_hidden,
    )

    decode_mask = decode_mask * residue_mask
    idx_to_loop = [int(i) for i in jax.device_get(jnp.where(decode_mask[0] == 1)[0])]
    num_letters = model.W_out.out_features
    log_conditional_probs = jnp.zeros(
        (atom_coords.shape[0], decode_mask.shape[1], num_letters), dtype=jnp.float32
    )

    for idx in idx_to_loop:
        order_mask = jnp.zeros((decode_mask.shape[1],), dtype=jnp.float32).at[idx].set(1.0)
        if backbone_only:
            order_mask = jnp.ones((decode_mask.shape[1],), dtype=jnp.float32).at[idx].set(0.0)
        loop_decode_mask = jnp.broadcast_to(order_mask[None, :], decode_mask.shape)
        _, decoder_backward_mask, decoder_forward_mask = model._build_decoder_masks(
            neighbor_idx=neighbor_idx,
            residue_mask=residue_mask,
            decode_mask=loop_decode_mask,
            decode_noise=decode_noise,
        )
        encoder_context_forward_masked = decoder_forward_mask * encoder_node_edge_context
        decoder_node_hidden = model._run_decoder(
            node_hidden_init=encoded_node_hidden,
            seq_edge_hidden=seq_edge_hidden,
            neighbor_idx=neighbor_idx,
            decoder_backward_mask=decoder_backward_mask,
            encoder_context_forward_masked=encoder_context_forward_masked,
            residue_mask=residue_mask,
        )
        logits = model.W_out(decoder_node_hidden)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        log_conditional_probs = log_conditional_probs.at[:, idx, :].set(log_probs[:, idx, :])
    return log_conditional_probs


def unconditional_probs_decode(model, atom_coords, residue_mask, residue_idx, chain_ids):
    """Compute unconditional log-probabilities."""
    neighbor_idx, encoded_node_hidden, encoded_edge_hidden = model._encode_graph_from_structure(
        atom_coords=atom_coords,
        residue_mask=residue_mask,
        residue_idx=residue_idx,
        chain_ids=chain_ids,
    )
    sequence_hidden_template = jnp.zeros_like(encoded_node_hidden)
    encoder_node_edge_context = model._build_encoder_context(
        encoded_node_hidden=encoded_node_hidden,
        encoded_edge_hidden=encoded_edge_hidden,
        neighbor_idx=neighbor_idx,
        sequence_hidden_template=sequence_hidden_template,
    )
    decoder_forward_mask = residue_mask[..., None, None]
    encoder_context_forward_masked = decoder_forward_mask * encoder_node_edge_context
    seq_edge_hidden = cat_neighbors_nodes(sequence_hidden_template, encoded_edge_hidden, neighbor_idx)
    decoder_backward_mask = jnp.zeros_like(encoder_context_forward_masked[..., :1])
    decoder_node_hidden = model._run_decoder(
        node_hidden_init=encoded_node_hidden,
        seq_edge_hidden=seq_edge_hidden,
        neighbor_idx=neighbor_idx,
        decoder_backward_mask=decoder_backward_mask,
        encoder_context_forward_masked=encoder_context_forward_masked,
        residue_mask=residue_mask,
    )
    logits = model.W_out(decoder_node_hidden)
    return jax.nn.log_softmax(logits, axis=-1)
