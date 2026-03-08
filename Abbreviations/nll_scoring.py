"""
Negative Log-Likelihood (NLL) scoring utilities for sequence models.

Provides functions to compute sequence-level NLL from model logits,
used for label scoring in classification tasks and minimal pair evaluation.
"""
from typing import Optional

import torch
import torch.nn as nn


def compute_sequence_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    score_mask: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    length_normalize: bool = True
) -> torch.Tensor:
    """
    Compute per-sequence negative log-likelihood (NLL).
    
    For each sequence in the batch, computes the mean cross-entropy loss
    over valid (non-ignored) positions. This is suitable for:
    - Label scoring in classification (lower NLL = more likely label)
    - Minimal pair evaluation (compare NLL of grammatical vs ungrammatical)
    
    Args:
        logits: (B, T, V) model output logits
        labels: (B, T) target token IDs
        score_mask: Optional (B, T) binary mask (1=score, 0=ignore).
                    Applied in addition to ignore_index.
        ignore_index: Label value to ignore (typically -100 for padding)
        length_normalize: If True, compute mean loss per token (default).
                         If False, compute sum of losses (total NLL).
    
    Returns:
        (B,) per-sequence NLL
        - If length_normalize=True: mean token loss (NLL per token)
        - If length_normalize=False: sum of token losses (total NLL)
    
    Example:
        >>> logits = model(input_ids)  # (B, T, V)
        >>> labels = target_ids        # (B, T) with -100 for padding
        >>> nll = compute_sequence_nll(logits, labels)  # (B,)
        >>> best_label = nll.argmin()  # Select label with lowest NLL
    """
    B, T, V = logits.shape
    
    # Compute token-level cross-entropy loss (no reduction)
    loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
    token_loss = loss_fn(logits.view(-1, V), labels.view(-1))  # (B*T,)
    token_loss = token_loss.view(B, T)  # (B, T)
    
    # Mask for valid (non-ignored) positions
    valid_mask = (labels != ignore_index).float()  # (B, T)
    
    # Apply additional score mask if provided
    if score_mask is not None:
        valid_mask = valid_mask * score_mask.float()
    
    # Aggregate loss per sequence
    if length_normalize:
        # Mean loss per token (default, reduces length bias)
        sequence_nll = (token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)  # (B,)
    else:
        # Sum of losses (total NLL, biased toward shorter sequences)
        sequence_nll = (token_loss * valid_mask).sum(dim=1)  # (B,)
    
    return sequence_nll


def compute_token_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Compute per-token cross-entropy losses.
    
    Useful for:
    - Analyzing which tokens contribute most to NLL
    - Debugging model predictions
    - Implementing custom aggregation logic
    
    Args:
        logits: (B, T, V) model output logits
        labels: (B, T) target labels
        ignore_index: Label value to ignore (typically -100 for padding)
    
    Returns:
        (B, T) per-token CE loss with ignore_index positions set to 0
    
    Example:
        >>> token_losses = compute_token_losses(logits, labels)
        >>> # Find most surprising token per sequence
        >>> max_loss_pos = token_losses.argmax(dim=1)
    """
    B, T, V = logits.shape
    
    # Compute token-level cross-entropy loss (no reduction)
    loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
    token_loss = loss_fn(logits.view(-1, V), labels.view(-1))  # (B*T,)
    token_loss = token_loss.view(B, T)  # (B, T)
    
    # Set ignore_index positions to 0
    mask = (labels != ignore_index).float()
    token_loss = token_loss * mask
    
    return token_loss


def build_shifted_decoder_inputs(
    label_target_ids: list,
    bos_id: int,
    pad_id: int = 0,
    ignore_index: int = -100,
    device: torch.device = None
) -> tuple:
    """
    Build properly shifted decoder inputs and labels for a label sequence.
    
    Implements the correct autoregressive scoring convention from BLiMP evaluator:
    - label_target_ids: [tok1, tok2, ..., EOS]  (label tokens with EOS at end)
    - decoder_input_ids: [BOS, tok1, tok2, ...]  (shifted right by 1)
    - labels: [tok1, tok2, ..., EOS]  (targets for NLL computation)
    
    This prevents information leakage by ensuring decoder never sees the
    token it's predicting at the same timestep (proper teacher forcing).
    
    Args:
        label_target_ids: List of label token IDs INCLUDING EOS at end
        bos_id: BOS token ID
        pad_id: Padding token ID
        ignore_index: Value for padding in labels tensor
        device: Torch device (optional)
    
    Returns:
        (decoder_input_ids, labels) - both are lists of ints
        - decoder_input_ids: [BOS] + label_target_ids[:-1]
        - labels: label_target_ids
    
    Example:
        >>> label_target_ids = [100, 200, 3]  # [tok1, tok2, EOS]
        >>> decoder_input_ids, labels = build_shifted_decoder_inputs(label_target_ids, bos_id=2)
        >>> # decoder_input_ids = [2, 100, 200]     # [BOS, tok1, tok2]
        >>> # labels = [100, 200, 3]                # [tok1, tok2, EOS]
    """
    # Shifted decoder input: [BOS] + label_target_ids[:-1]
    decoder_input_ids = [bos_id] + label_target_ids[:-1]
    
    # Labels: label_target_ids (what we want to predict)
    labels = label_target_ids
    
    # Self-check: decoder_input_ids[1:] should equal labels[:-1]
    # (i.e., shifted by one position)
    assert len(decoder_input_ids) == len(labels), "Length mismatch after shift"
    for t in range(len(labels) - 1):
        if decoder_input_ids[t+1] != labels[t]:
            raise ValueError(
                f"Decoder inputs are not properly shifted; leakage risk. "
                f"At position {t}: decoder_input[{t+1}]={decoder_input_ids[t+1]} != labels[{t}]={labels[t]}"
            )
    
    return decoder_input_ids, labels


def batch_score_labels(
    model: nn.Module,
    encoder_input_ids: torch.Tensor,
    encoder_mask: torch.Tensor,
    label_ids_list: list,
    device: torch.device,
    bos_id: int,
    pad_id: int = 0,
    ignore_index: int = -100,
    length_normalize: bool = True
) -> torch.Tensor:
    """
    Score multiple label candidates for a batch of inputs using SHIFTED decoder inputs.
    
    For encoder-decoder classification: given encoder input, compute NLL
    for each possible label sequence. Uses proper autoregressive scoring
    with shifted decoder inputs to prevent information leakage.
    
    CRITICAL: Each label_ids in label_ids_list must be label_target_ids = [tok1, tok2, ..., EOS]
    (i.e., label tokens WITH EOS at end, but WITHOUT BOS).
    This function constructs:
    - decoder_input_ids = [BOS, tok1, tok2, ...]  (shifted right)
    - labels = [tok1, tok2, ..., EOS]  (targets for NLL)
    
    Args:
        model: Encoder-decoder model with forward(encoder_input_ids, decoder_input_ids,
               encoder_mask, decoder_mask, causal_mask) -> (logits, kl_div)
        encoder_input_ids: (B, T_enc) encoder input token IDs
        encoder_mask: (B, T_enc) encoder attention mask
        label_ids_list: List of L label token ID lists. Each should be label_target_ids
                        (label tokens + EOS, no BOS). E.g., [tok1, tok2, EOS]
        device: Device to run on
        bos_id: BOS token ID (required for shifting)
        pad_id: Padding token ID
        ignore_index: Label value to ignore in NLL computation
        length_normalize: If True, compute mean NLL per token
    
    Returns:
        (B, L) NLL scores for each (input, label) pair
    
    Example:
        >>> # Score 3 label candidates for each input
        >>> encoder_ids = tokenizer.encode_batch(inputs)  # (B, T_enc)
        >>> # Each label should be tokenized with EOS but no BOS
        >>> label_ids_list = [tokenizer.encode(label, add_bos=False, add_eos=True) for label in label_texts]
        >>> nlls = batch_score_labels(model, encoder_ids, encoder_mask, label_ids_list, device, bos_id=tokenizer.bos_id)
        >>> predictions = nlls.argmin(dim=1)  # (B,) best label per input
    """
    B = encoder_input_ids.size(0)
    L = len(label_ids_list)
    
    # Prepare label tensors
    max_label_len = max(len(ids) for ids in label_ids_list)
    
    # Storage for NLLs
    all_nlls = torch.zeros(B, L, device=device)
    
    with torch.no_grad():
        for label_idx, label_target_ids in enumerate(label_ids_list):
            # Build shifted decoder inputs and labels
            decoder_input_ids_list, labels_list = build_shifted_decoder_inputs(
                label_target_ids=label_target_ids,
                bos_id=bos_id,
                pad_id=pad_id,
                ignore_index=ignore_index,
                device=device
            )
            
            # Prepare tensors
            seq_len = len(decoder_input_ids_list)
            decoder_input_tensor = torch.full((B, seq_len), pad_id, dtype=torch.long, device=device)
            labels_tensor = torch.full((B, seq_len), ignore_index, dtype=torch.long, device=device)
            
            for b in range(B):
                # Decoder input: [BOS] + label_target_ids[:-1] (shifted)
                decoder_input_tensor[b, :seq_len] = torch.tensor(decoder_input_ids_list, dtype=torch.long, device=device)
                # Labels: label_target_ids (targets for loss)
                labels_tensor[b, :seq_len] = torch.tensor(labels_list, dtype=torch.long, device=device)
            
            # Create decoder mask (attend to non-padding)
            decoder_mask = (decoder_input_tensor != pad_id).long()
            
            # Dummy causal mask (model creates its own internally)
            causal_mask = torch.ones(1, 1, device=device)
            
            # Forward pass
            logits, _ = model(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_tensor,
                encoder_mask=encoder_mask,
                decoder_mask=decoder_mask,
                causal_mask=causal_mask
            )
            
            # Compute NLL for this label
            nll = compute_sequence_nll(
                logits=logits,
                labels=labels_tensor,
                ignore_index=ignore_index,
                length_normalize=length_normalize
            )
            
            all_nlls[:, label_idx] = nll
    
    return all_nlls


def calibrate_label_priors(
    model: nn.Module,
    label_ids_list: list,
    device: torch.device,
    bos_id: int,
    eos_id: int,
    pad_id: int = 0,
    ignore_index: int = -100,
    length_normalize: bool = True
) -> dict:
    """
    Compute label prior NLL by scoring each label with EMPTY encoder input.
    
    This measures the unconditional probability P(label) without any input context.
    Useful for calibration: calibrated_score = nll(label|x) - prior_nll[label]
    
    Uses the same scoring convention as BLiMP decoder mode with no prompt:
    - encoder_input_ids = [BOS, EOS] (minimal constant context)
    - decoder_input_ids = [BOS] + label_target_ids[:-1] (shifted)
    - labels = label_target_ids
    
    Args:
        model: Encoder-decoder model
        label_ids_list: List of label token ID lists (each with EOS, no BOS)
        device: Torch device
        bos_id: BOS token ID
        eos_id: EOS token ID
        pad_id: Padding token ID
        ignore_index: Label value to ignore in NLL computation
        length_normalize: If True, compute mean NLL per token
    
    Returns:
        Dictionary mapping label_idx -> prior_nll (float)
    
    Example:
        >>> label_ids_list = [[100, 200, 3], [300, 3]]  # Each ends with EOS=3
        >>> priors = calibrate_label_priors(model, label_ids_list, device, bos_id=2, eos_id=3)
        >>> # priors = {0: 2.5, 1: 1.8}
    """
    # Create empty encoder input: [BOS, EOS]
    encoder_input_ids = torch.tensor([[bos_id, eos_id]], dtype=torch.long, device=device)  # (1, 2)
    encoder_mask = torch.ones_like(encoder_input_ids)  # (1, 2)
    
    L = len(label_ids_list)
    prior_nlls = {}
    
    with torch.no_grad():
        for label_idx, label_target_ids in enumerate(label_ids_list):
            # Build shifted decoder inputs and labels
            decoder_input_ids_list, labels_list = build_shifted_decoder_inputs(
                label_target_ids=label_target_ids,
                bos_id=bos_id,
                pad_id=pad_id,
                ignore_index=ignore_index,
                device=device
            )
            
            # Prepare tensors
            seq_len = len(decoder_input_ids_list)
            decoder_input_tensor = torch.full((1, seq_len), pad_id, dtype=torch.long, device=device)
            labels_tensor = torch.full((1, seq_len), ignore_index, dtype=torch.long, device=device)
            
            # Fill in sequences
            decoder_input_tensor[0, :seq_len] = torch.tensor(decoder_input_ids_list, dtype=torch.long, device=device)
            labels_tensor[0, :seq_len] = torch.tensor(labels_list, dtype=torch.long, device=device)
            
            # Create decoder mask
            decoder_mask = (decoder_input_tensor != pad_id).long()
            
            # Dummy causal mask
            causal_mask = torch.ones(1, 1, device=device)
            
            # Forward pass with empty encoder
            logits, _ = model(
                encoder_input_ids=encoder_input_ids,
                decoder_input_ids=decoder_input_tensor,
                encoder_mask=encoder_mask,
                decoder_mask=decoder_mask,
                causal_mask=causal_mask
            )
            
            # Compute NLL
            nll = compute_sequence_nll(
                logits=logits,
                labels=labels_tensor,
                ignore_index=ignore_index,
                length_normalize=length_normalize
            )
            
            prior_nlls[label_idx] = nll.item()
    
    return prior_nlls


def summarize_token_losses(
    label_token_ids: list,
    token_losses: list,
    tokenizer,
    ignore_index: int = -100,
    pad_id: int = 0
) -> list:
    """
    Summarize token-level losses for a single label sequence.
    
    Returns a list of per-token info rows with loss, piece, and special token flags.
    Useful for displaying independent per-label loss tables without forced alignment.
    
    Args:
        label_token_ids: List of token IDs for the label
        token_losses: List of per-token losses (same length)
        tokenizer: SentencePiece tokenizer with eos_id, bos_id attributes
        ignore_index: Ignore index to skip
        pad_id: Padding token ID to skip
    
    Returns:
        List of dicts with: pos, tok_id, piece, loss, is_special
    
    Example:
        >>> rows = summarize_token_losses([100, 200, 3], [1.2, 2.3, 0.5], tokenizer)
        >>> # [{"pos": 0, "tok_id": 100, "piece": "▁b", "loss": 1.2, "is_special": False}, ...]
    """
    eos_id = tokenizer.eos_id
    bos_id = tokenizer.bos_id
    
    rows = []
    for t, (tok_id, loss) in enumerate(zip(label_token_ids, token_losses)):
        # Skip padding and ignore_index
        if tok_id == pad_id or tok_id == ignore_index:
            continue
        if loss == 0.0 and tok_id == pad_id:
            continue
        
        piece = tokenizer.id_to_piece(tok_id)
        is_special = tok_id in {bos_id, eos_id, pad_id}
        
        rows.append({
            "pos": t,
            "tok_id": tok_id,
            "piece": piece,
            "loss": loss,
            "is_special": is_special
        })
    
    return rows


def top_contributing_overlap_positions(
    token_losses_pred: list,
    token_losses_alt: list,
    label_ids_pred: list,
    label_ids_alt: list,
    tokenizer,
    ignore_eos: bool = True,
    ignore_bos: bool = True,
    pad_id: int = 0,
    ignore_index: int = -100,
) -> tuple:
    """
    Compute contributions only over OVERLAPPING positions where both labels have tokens.
    
    This is more honest than forcing alignment with padding. Only positions where
    BOTH labels have real (non-pad, non-ignore) tokens are included.
    
    Contribution = loss_alt[t] - loss_pred[t] (positive = supports predicted label)
    
    Args:
        token_losses_pred: List of per-token losses for predicted label
        token_losses_alt: List of per-token losses for alternative label
        label_ids_pred: List of token IDs for predicted label
        label_ids_alt: List of token IDs for alternative label
        tokenizer: SentencePiece tokenizer with eos_id, bos_id attributes
        ignore_eos: If True, exclude EOS positions from ranking
        ignore_bos: If True, exclude BOS positions from ranking
        pad_id: Padding token ID to exclude
        ignore_index: Ignore index to exclude
    
    Returns:
        Tuple of (contributions_list, has_special_only_flag)
        - contributions_list: List of dicts with pos, pred_piece, alt_piece, losses, contrib
        - has_special_only_flag: True if all positions were filtered (only special tokens)
    """
    eos_id = tokenizer.eos_id
    bos_id = tokenizer.bos_id
    
    # Only consider overlap positions
    min_len = min(len(token_losses_pred), len(token_losses_alt))
    contributions = []
    all_filtered = True
    
    for t in range(min_len):
        pred_loss = token_losses_pred[t]
        alt_loss = token_losses_alt[t]
        pred_id = label_ids_pred[t]
        alt_id = label_ids_alt[t]
        
        # Skip if either is padding/ignore
        if pred_id == pad_id or pred_id == ignore_index:
            continue
        if alt_id == pad_id or alt_id == ignore_index:
            continue
        if pred_loss == 0.0 or alt_loss == 0.0:
            continue
        
        pred_piece = tokenizer.id_to_piece(pred_id)
        alt_piece = tokenizer.id_to_piece(alt_id)
        contrib = alt_loss - pred_loss
        
        # Check if should be filtered
        is_special = False
        if ignore_eos and (pred_id == eos_id or alt_id == eos_id):
            is_special = True
        if ignore_bos and (pred_id == bos_id or alt_id == bos_id):
            is_special = True
        
        if not is_special:
            all_filtered = False
        
        entry = {
            "pos": t,
            "pred_piece": pred_piece,
            "alt_piece": alt_piece,
            "pred_loss": pred_loss,
            "alt_loss": alt_loss,
            "contrib": contrib,
            "is_special": is_special
        }
        
        if not is_special:
            contributions.append(entry)
    
    # Sort by contribution descending (positive = supports predicted)
    contributions.sort(key=lambda x: x["contrib"], reverse=True)
    
    return contributions, all_filtered


def margin_accounting(
    token_losses_pred: list,
    token_losses_alt: list,
    label_ids_pred: list,
    label_ids_alt: list,
    tokenizer,
    pad_id: int = 0,
    ignore_index: int = -100
) -> dict:
    """
    Decompose the margin between two labels into interpretable buckets.
    
    Shows where the margin comes from:
    - Overlap positions (both labels have tokens)
    - Extra pred-only positions (longer predicted label)
    - Extra alt-only positions (longer alternative label)
    - Special tokens (EOS, BOS)
    
    Args:
        token_losses_pred: List of per-token losses for predicted label
        token_losses_alt: List of per-token losses for alternative label
        label_ids_pred: List of token IDs for predicted label
        label_ids_alt: List of token IDs for alternative label
        tokenizer: SentencePiece tokenizer with eos_id, bos_id
        pad_id: Padding token ID
        ignore_index: Ignore index
    
    Returns:
        Dict with:
        - pred_sum, pred_count, pred_mean: Predicted label totals
        - alt_sum, alt_count, alt_mean: Alternative label totals
        - total_margin_mean: alt_mean - pred_mean (should match sample margin)
        - delta_sum_total: alt_sum - pred_sum (in SUM space)
        - Decomposition buckets:
            - delta_sum_overlap_non_special
            - delta_sum_overlap_special
            - delta_sum_extra_pred_only
            - delta_sum_extra_alt_only
    """
    eos_id = tokenizer.eos_id
    bos_id = tokenizer.bos_id
    
    # Compute totals
    pred_sum = 0.0
    pred_count = 0
    for t, (tid, loss) in enumerate(zip(label_ids_pred, token_losses_pred)):
        if tid != pad_id and tid != ignore_index and loss > 0.0:
            pred_sum += loss
            pred_count += 1
    
    alt_sum = 0.0
    alt_count = 0
    for t, (tid, loss) in enumerate(zip(label_ids_alt, token_losses_alt)):
        if tid != pad_id and tid != ignore_index and loss > 0.0:
            alt_sum += loss
            alt_count += 1
    
    pred_mean = pred_sum / pred_count if pred_count > 0 else 0.0
    alt_mean = alt_sum / alt_count if alt_count > 0 else 0.0
    total_margin_mean = alt_mean - pred_mean
    delta_sum_total = alt_sum - pred_sum
    
    # Decompose into buckets
    delta_overlap_non_special = 0.0
    delta_overlap_special = 0.0
    delta_extra_pred = 0.0
    delta_extra_alt = 0.0
    
    min_len = min(len(label_ids_pred), len(label_ids_alt))
    
    # Overlap positions
    for t in range(min_len):
        pred_id = label_ids_pred[t]
        alt_id = label_ids_alt[t]
        pred_loss = token_losses_pred[t]
        alt_loss = token_losses_alt[t]
        
        # Skip invalid
        if pred_id == pad_id or pred_id == ignore_index:
            continue
        if alt_id == pad_id or alt_id == ignore_index:
            continue
        if pred_loss == 0.0 or alt_loss == 0.0:
            continue
        
        delta = alt_loss - pred_loss
        is_special = (pred_id in {bos_id, eos_id}) or (alt_id in {bos_id, eos_id})
        
        if is_special:
            delta_overlap_special += delta
        else:
            delta_overlap_non_special += delta
    
    # Extra pred-only positions (pred is longer)
    for t in range(min_len, len(label_ids_pred)):
        pred_id = label_ids_pred[t]
        pred_loss = token_losses_pred[t]
        if pred_id != pad_id and pred_id != ignore_index and pred_loss > 0.0:
            # This contributes negatively to delta (increases pred score)
            delta_extra_pred -= pred_loss
    
    # Extra alt-only positions (alt is longer)
    for t in range(min_len, len(label_ids_alt)):
        alt_id = label_ids_alt[t]
        alt_loss = token_losses_alt[t]
        if alt_id != pad_id and alt_id != ignore_index and alt_loss > 0.0:
            # This contributes positively to delta (increases alt score)
            delta_extra_alt += alt_loss
    
    return {
        "pred_sum": pred_sum,
        "pred_count": pred_count,
        "pred_mean": pred_mean,
        "alt_sum": alt_sum,
        "alt_count": alt_count,
        "alt_mean": alt_mean,
        "total_margin_mean": total_margin_mean,
        "delta_sum_total": delta_sum_total,
        "delta_sum_overlap_non_special": delta_overlap_non_special,
        "delta_sum_overlap_special": delta_overlap_special,
        "delta_sum_extra_pred_only": delta_extra_pred,
        "delta_sum_extra_alt_only": delta_extra_alt,
    }


def top_contributing_positions(
    token_losses_pred: list,
    token_losses_alt: list,
    label_ids_pred: list,
    label_ids_alt: list,
    tokenizer,
    ignore_eos: bool = True,
    ignore_bos: bool = True,
    pad_id: int = 0,
    ignore_index: int = -100,
) -> tuple:
    """
    DEPRECATED: Use top_contributing_overlap_positions() instead.
    
    This function pads to max_len which can hide margin sources.
    Kept for backward compatibility.
    
    Contribution = loss_alt[t] - loss_pred[t] (positive = supports predicted label)
    
    Args:
        token_losses_pred: List of per-token losses for predicted label
        token_losses_alt: List of per-token losses for alternative label
        label_ids_pred: List of token IDs for predicted label
        label_ids_alt: List of token IDs for alternative label
        tokenizer: SentencePiece tokenizer with eos_id, bos_id attributes
        ignore_eos: If True, exclude EOS positions from ranking
        ignore_bos: If True, exclude BOS positions from ranking
        pad_id: Padding token ID to exclude
        ignore_index: Ignore index to exclude
    
    Returns:
        Tuple of (contributions_list, has_special_only_flag)
        - contributions_list: List of dicts with pos, pred_piece, alt_piece, losses, contrib
        - has_special_only_flag: True if all positions were filtered (only special tokens)
    """
    eos_id = tokenizer.eos_id
    bos_id = tokenizer.bos_id
    
    max_len = max(len(token_losses_pred), len(token_losses_alt))
    contributions = []
    all_filtered = True
    
    for t in range(max_len):
        pred_loss = token_losses_pred[t] if t < len(token_losses_pred) else 0.0
        alt_loss = token_losses_alt[t] if t < len(token_losses_alt) else 0.0
        pred_id = label_ids_pred[t] if t < len(label_ids_pred) else pad_id
        alt_id = label_ids_alt[t] if t < len(label_ids_alt) else pad_id
        
        # Skip positions where both are padding/ignore
        if pred_loss == 0.0 and alt_loss == 0.0:
            continue
        if pred_id == ignore_index or alt_id == ignore_index:
            continue
        if pred_id == pad_id and alt_id == pad_id:
            continue
        
        pred_piece = tokenizer.id_to_piece(pred_id) if pred_id != pad_id else "<pad>"
        alt_piece = tokenizer.id_to_piece(alt_id) if alt_id != pad_id else "<pad>"
        contrib = alt_loss - pred_loss
        
        # Check if should be filtered
        is_special = False
        if ignore_eos and (pred_id == eos_id or alt_id == eos_id):
            is_special = True
        if ignore_bos and (pred_id == bos_id or alt_id == bos_id):
            is_special = True
        
        contributions.append({
            "pos": t,
            "pred_piece": pred_piece,
            "alt_piece": alt_piece,
            "pred_loss": pred_loss,
            "alt_loss": alt_loss,
            "contrib": contrib,
            "pred_id": pred_id,
            "alt_id": alt_id,
            "is_special": is_special,
        })
        
        if not is_special:
            all_filtered = False
    
    # If all positions were special tokens, fall back to including them
    has_special_only = all_filtered and contributions
    if has_special_only:
        # Keep all contributions but mark that we had to include special tokens
        filtered_contributions = contributions
    else:
        # Filter out special tokens
        filtered_contributions = [c for c in contributions if not c["is_special"]]
    
    # Sort by contribution (descending: most supportive first)
    filtered_contributions.sort(key=lambda x: x["contrib"], reverse=True)
    
    return filtered_contributions, has_special_only
