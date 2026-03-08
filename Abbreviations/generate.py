"""
Text generation utilities for the encoder-decoder model.

Supports greedy, top-k, top-p (nucleus), and temperature sampling.
Can generate from initials-space prompts.
"""
from typing import Optional, List
from pathlib import Path

import torch
import torch.nn.functional as F

from model import EncoderDecoderModel
from tokenizer_spm import InitialsTokenizer
from config import ModelConfig, GenerationConfig


def _build_ngram_context(
    prompt_ids: List[int],
    generated_ids: List[int],
    bos_id: int,
    device: str
) -> torch.Tensor:
    """
    Build N-gram context tensor for prompt-aware bigram lookup.
    
    Constructs context from prompt + generated tokens (excluding decoder BOS),
    then returns the last token ID for bigram P(next | prev).
    
    Args:
        prompt_ids: List of prompt token IDs (used for encoder, no BOS)
        generated_ids: List of generated token IDs so far (excludes decoder BOS)
        bos_id: BOS token ID (used as fallback if context is empty)
        device: Device for output tensor
    
    Returns:
        Tensor of shape (1, 1) with last token ID (prev_id for bigram lookup)
    
    Examples:
        >>> # Empty generation
        >>> _build_ngram_context([10, 20, 30], [], bos_id=1, device="cpu")
        tensor([[30]])  # last prompt token
        
        >>> # With generated tokens
        >>> _build_ngram_context([10, 20, 30], [40, 50], bos_id=1, device="cpu")
        tensor([[50]])  # last generated token
        
        >>> # Empty prompt and generation (edge case)
        >>> _build_ngram_context([], [], bos_id=1, device="cpu")
        tensor([[1]])  # fallback to BOS
    """
    # Build full context: prompt + generated (no decoder BOS)
    context_ids = prompt_ids + generated_ids
    
    # Edge case: empty context, use BOS as prev_id
    if len(context_ids) == 0:
        prev_id = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    else:
        prev_id = torch.tensor([[context_ids[-1]]], dtype=torch.long, device=device)
    
    return prev_id


def _resolve_ngram_alpha(
    model: EncoderDecoderModel,
    model_config: Optional[ModelConfig],
    override: Optional[float]
) -> float:
    """
    Resolve N-gram alpha from multiple sources with fallback priority.
    
    Priority order:
    1. CLI override (if provided)
    2. model.prob_encoder.config.prior_alpha (if prob_encoder exists)
    3. model.config.prior_alpha (if available)
    4. model_config.prior_alpha (if passed in)
    5. Default to 0.0
    
    Args:
        model: EncoderDecoderModel instance
        model_config: Optional ModelConfig passed from checkpoint
        override: Optional CLI override value
    
    Returns:
        Resolved alpha value
    """
    # Priority 1: CLI override
    if override is not None:
        return override
    
    # Priority 2: model.prob_encoder.config.prior_alpha
    if hasattr(model, 'prob_encoder') and model.prob_encoder is not None:
        if hasattr(model.prob_encoder, 'config') and hasattr(model.prob_encoder.config, 'prior_alpha'):
            return model.prob_encoder.config.prior_alpha
    
    # Priority 3: model.config.prior_alpha
    if hasattr(model, 'config') and hasattr(model.config, 'prior_alpha'):
        return model.config.prior_alpha
    
    # Priority 4: model_config.prior_alpha (passed from checkpoint)
    if model_config is not None and hasattr(model_config, 'prior_alpha'):
        return model_config.prior_alpha
    
    # Priority 5: Default
    return 0.0


def _detect_repetition(
    token_ids: List[int],
    rep_ngram: int = 3,
    rep_window: int = 30,
    rep_max_hits: int = 3,
    rep_token_run: int = 8,
) -> bool:
    """
    Detect degenerate repetition in generated sequence.
    
    Triggers if:
    1. Any n-gram appears >= rep_max_hits times in the last rep_window tokens, OR
    2. The same token repeats > rep_token_run times in a row
    
    Args:
        token_ids: List of generated token IDs
        rep_ngram: N-gram size to check (default: 3)
        rep_window: Window size to check for repetition (default: 30)
        rep_max_hits: Max allowed repetitions of same n-gram (default: 3)
        rep_token_run: Max allowed consecutive repetitions of same token (default: 8)
    
    Returns:
        True if repetition detected, False otherwise
    """
    if len(token_ids) < rep_ngram:
        return False
    
    # Check 1: N-gram repetition in recent window
    window = token_ids[-rep_window:] if len(token_ids) > rep_window else token_ids
    if len(window) >= rep_ngram:
        ngrams = [tuple(window[i:i+rep_ngram]) for i in range(len(window) - rep_ngram + 1)]
        from collections import Counter
        ngram_counts = Counter(ngrams)
        if ngram_counts.most_common(1)[0][1] >= rep_max_hits:
            return True
    
    # Check 2: Same token repeated consecutively
    if len(token_ids) >= rep_token_run:
        last_tokens = token_ids[-rep_token_run:]
        if len(set(last_tokens)) == 1:  # All same token
            return True
    
    return False


def generate_text(
    model: EncoderDecoderModel,
    tokenizer: InitialsTokenizer,
    prompt: str = "",
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: str = "cuda",
    min_len: int = 0,
    debug: bool = False,
    generation_config: Optional[GenerationConfig] = None,
    ngram_alpha_override: Optional[float] = None,
    disable_ngram: bool = False,
    eos_suppress_len: Optional[int] = None,
    rep_ngram: int = 3,
    rep_window: int = 30,
    rep_max_hits: int = 3,
    rep_token_run: int = 8,
    min_prompt_tokens: int = 15,
    prior_temp: float = 1.5,
    model_config: Optional[ModelConfig] = None,
) -> str:
    """
    Generate text from a prompt using encoder-decoder causal objective.
    
    This matches the training objective:
    - Encoder processes prompt once: encoder_memory = encode([BOS] + prompt_ids)
    - Decoder autoregressively generates continuation:
      - Start with decoder_input_ids = [BOS]
      - Each step: feed decoder_input_ids + encoder_memory -> logits
      - Sample next token from logits[-1], append to decoder_input_ids
      - Stop on EOS or max_new_tokens reached
    
    Args:
        model: EncoderDecoderModel instance
        tokenizer: InitialsTokenizer instance
        prompt: Input prompt in initials space (e.g., "zh g sh h")
        max_new_tokens: Maximum NEW tokens to generate (excluding BOS)
        temperature: Sampling temperature (higher = more random)
        top_k: Top-k sampling (only consider top k tokens)
        top_p: Nucleus sampling (only consider tokens with cumulative prob > p)
        device: Device to run on
        min_len: Minimum generation length (blocks EOS until reached)
        debug: Enable debug mode
        generation_config: Optional GenerationConfig object
        ngram_alpha_override: Override n-gram prior alpha
        disable_ngram: Disable n-gram prior
        eos_suppress_len: Length at which to stop suppressing EOS (default: min_len)
        rep_ngram: N-gram size for repetition detection (default: 3)
        rep_window: Window size for repetition detection (default: 30)
        rep_max_hits: Max allowed n-gram repetitions (default: 3)
        rep_token_run: Max consecutive token repetitions (default: 8)
        min_prompt_tokens: Minimum prompt length to warn about (default: 15)
        prior_temp: Temperature for N-gram prior softmax (default: 1.5, flattens bigram)
        model_config: Optional ModelConfig from checkpoint (for alpha resolution)
    
    Returns:
        Generated text in initials space (continuation only, prompt not included)
    """
    model.eval()
    model = model.to(device)
    
    # Set EOS suppression length (default to min_len)
    if eos_suppress_len is None:
        eos_suppress_len = min_len
    
    # Use default GenerationConfig if not provided
    if generation_config is None:
        generation_config = GenerationConfig(debug=debug)
    
    # Encode prompt
    if prompt:
        prompt_ids = tokenizer.encode(prompt, add_bos=False, add_eos=False)
    else:
        # Empty prompt
        prompt_ids = []
    
    # Special token IDs
    eos_id = tokenizer.eos_id
    
    # Warn if prompt is too short
    if len(prompt_ids) < min_prompt_tokens:
        print(f"Warning: prompt is very short ({len(prompt_ids)} tokens); model was trained on longer prompts.")
        if generation_config.debug:
            print(f"[DEBUG] Consider using {min_prompt_tokens}+ tokens for better results.")
    
    # ===== DEBUG: Print generation settings once =====
    if generation_config.debug:
        print("\n[DEBUG] ===== Generation Settings =====")
        print(f"[DEBUG] Prompt token IDs length: {len(prompt_ids)}")
        print(f"[DEBUG] Prompt token IDs: {prompt_ids[:50]}..." if len(prompt_ids) > 50 else f"[DEBUG] Prompt token IDs: {prompt_ids}")
        print(f"[DEBUG] Generation params: temp={temperature}, top_k={top_k}, top_p={top_p}, max_new={max_new_tokens}, min_len={min_len}")
        print(f"[DEBUG] EOS token ID: {eos_id} (EOS will NOT be appended to output)")
    
    # Encoder input: [BOS] + prompt_ids (no EOS for encoder in generation mode)
    encoder_input_ids = torch.tensor([[tokenizer.bos_id] + prompt_ids], dtype=torch.long, device=device)
    encoder_mask = torch.ones_like(encoder_input_ids)
    
    # Encode ONCE (not re-encoded every step)
    with torch.no_grad():
        encoder_hidden = model.encoder(encoder_input_ids, encoder_mask)
        
        # Apply probabilistic encoder if enabled
        if model.prob_encoder is not None:
            encoder_memory, _, _ = model.prob_encoder(
                encoder_hidden,
                encoder_input_ids,
                encoder_mask
            )
        else:
            encoder_memory = encoder_hidden
    
    # Detect where n-gram prior lives and resolve alpha
    ngram_source = "not available"
    active_ngram_prior = None
    ngram_blending_active = False
    
    # Check where n-gram prior exists: model.prob_encoder.ngram_prior or model.ngram_prior
    if hasattr(model, 'prob_encoder') and model.prob_encoder is not None:
        if hasattr(model.prob_encoder, 'ngram_prior'):
            if model.prob_encoder.ngram_prior is not None and model.prob_encoder.ngram_prior.is_fitted:
                active_ngram_prior = model.prob_encoder.ngram_prior
                ngram_source = "model.prob_encoder.ngram_prior"
    
    # Fallback: check model.ngram_prior directly
    if active_ngram_prior is None and hasattr(model, 'ngram_prior'):
        if model.ngram_prior is not None and model.ngram_prior.is_fitted:
            active_ngram_prior = model.ngram_prior
            ngram_source = "model.ngram_prior"
    
    # Handle --no_ngram flag
    if disable_ngram:
        active_ngram_prior = None
        ngram_blending_active = False
        ngram_source = "disabled by --no_ngram"
    elif active_ngram_prior is not None:
        # Resolve alpha using robust helper with fallback priority
        ngram_alpha = _resolve_ngram_alpha(model, model_config, ngram_alpha_override)
        
        if ngram_alpha > 0:
            ngram_blending_active = True
        else:
            ngram_blending_active = False
            ngram_source = "not available (alpha is 0.0)"
    
    # Debug: Print encoder output info and n-gram status
    if generation_config.debug:
        print(f"[DEBUG] Encoder output shape: {encoder_memory.shape}")
        print(f"[DEBUG] Encoder output computed ONCE (id={id(encoder_memory)})")
        
        # N-gram prior status (truthful)
        if ngram_blending_active and active_ngram_prior is not None:
            print(f"[DEBUG] N-gram blending: ACTIVE")
            print(f"[DEBUG] N-gram alpha: {ngram_alpha}")
            print(f"[DEBUG] N-gram source: {ngram_source}")
        else:
            print(f"[DEBUG] N-gram blending: INACTIVE ({ngram_source})")
        print("[DEBUG] ================================\n")
    
    # Store encoder memory ID for debug assertion
    encoder_memory_id = id(encoder_memory)
    
    # Initialize decoder with BOS
    decoder_input_ids = torch.tensor([[tokenizer.bos_id]], dtype=torch.long, device=device)
    new_tokens = []  # Store generated tokens (excluding BOS, excluding EOS)
    eos_sampled_at_step = None
    eos_event_probs = None  # Store probs when EOS is sampled
    repetition_stopped = False  # Track if stopped due to repetition
    
    # Generate autoregressively (step starts at 1 for clarity)
    for step_idx in range(max_new_tokens):
        step = step_idx + 1  # Human-readable step (1-indexed)
        
        # Check for repetition (before generating next token)
        if _detect_repetition(new_tokens, rep_ngram, rep_window, rep_max_hits, rep_token_run):
            repetition_stopped = True
            if generation_config.debug:
                print(f"[DEBUG] Repetition detected at step {step}; stopping early (no EOS appended)")
            break
        
        # Debug: Verify encoder memory is not recomputed (self-check)
        if generation_config.debug and step == 1:
            assert id(encoder_memory) == encoder_memory_id, "ERROR: encoder_memory was recomputed!"
        
        T_dec = decoder_input_ids.size(1)
        
        # Create causal mask (upper triangular - mask future)
        tgt_mask = torch.triu(torch.ones(T_dec, T_dec, device=device, dtype=torch.bool), diagonal=1)
        
        # Decoder masks
        decoder_mask = torch.ones(1, T_dec, dtype=torch.long, device=device)
        tgt_key_padding_mask = (decoder_mask == 0)
        memory_key_padding_mask = (encoder_mask == 0)
        
        # Forward pass decoder
        with torch.no_grad():
            logits = model.decoder(
                input_ids=decoder_input_ids,
                encoder_memory=encoder_memory,  # Reuse encoded prompt
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )  # (1, T_dec, V)
        
        # Get logits for last position (next token prediction)
        decoder_logits_step = logits[:, -1, :] / temperature  # (1, V)
        
        # ===== N-GRAM PRIOR BLENDING IN LOG-PROB SPACE =====
        # Order: blend -> EOS suppression -> top-k -> top-p -> softmax -> sample
        
        # Compute decoder-only log-probs (for blending and debug)
        logp_dec = F.log_softmax(decoder_logits_step, dim=-1)  # (1, V)
        
        # Compute raw EOS probability BEFORE any modifications (for debugging)
        raw_eos_prob = torch.exp(logp_dec[0, eos_id]).item()
        
        # Blend with n-gram prior if active
        if ngram_blending_active and active_ngram_prior is not None:
            # Build prompt-aware context: prompt + generated_ids (no decoder BOS)
            prev_id = _build_ngram_context(
                prompt_ids=prompt_ids,
                generated_ids=new_tokens,
                bos_id=tokenizer.bos_id,
                device=device
            )  # (1, 1)
            
            # Get N-gram prior distribution P(next | prev) with prior_temp
            prior_probs = active_ngram_prior.get_prior_distribution(
                prev_ids=prev_id,
                use_bigram=True,
                temperature=prior_temp
            )  # (1, 1, V)
            ngram_probs = prior_probs[0, 0, :]  # (V,)
            
            # Convert to log-probs (clamp to avoid log(0))
            logp_ng = torch.log(ngram_probs.clamp_min(1e-12))  # (V,)
            
            # Interpolate in log-prob space
            logp = (1 - ngram_alpha) * logp_dec[0] + ngram_alpha * logp_ng  # (V,)
            next_token_logits = logp.unsqueeze(0)  # (1, V)
            
            # Compute blended EOS prob for debugging
            blended_eos_prob = torch.exp(logp[eos_id]).item()
            
            # Debug: Show top-5 from decoder-only and blended for first 2 steps
            if generation_config.debug and step <= 2:
                print(f"\n[DEBUG] ===== Step {step} N-gram Blending Debug (Log-Prob Space) =====")
                
                # Decoder-only top-5
                dec_probs = torch.exp(logp_dec[0])  # Convert log-probs to probs
                dec_top5_probs, dec_top5_ids = torch.topk(dec_probs, k=5)
                print(f"[DEBUG] Decoder-only top-5:")
                for i, (prob, idx) in enumerate(zip(dec_top5_probs.tolist(), dec_top5_ids.tolist())):
                    try:
                        piece = tokenizer.decode([idx], skip_special_tokens=False)
                    except:
                        piece = "<err>"
                    print(f"[DEBUG]   {i+1}. id={idx:5d}  prob={prob:.4f}  '{piece}'")
                
                # Blended top-5
                blended_probs = torch.exp(logp)  # Convert log-probs to probs
                blend_top5_probs, blend_top5_ids = torch.topk(blended_probs, k=5)
                print(f"[DEBUG] Blended (alpha={ngram_alpha:.2f}, prior_temp={prior_temp:.2f}) top-5:")
                for i, (prob, idx) in enumerate(zip(blend_top5_probs.tolist(), blend_top5_ids.tolist())):
                    try:
                        piece = tokenizer.decode([idx], skip_special_tokens=False)
                    except:
                        piece = "<err>"
                    print(f"[DEBUG]   {i+1}. id={idx:5d}  prob={prob:.4f}  '{piece}'")
                print(f"[DEBUG] raw_eos_prob={raw_eos_prob:.4f}, blended_eos_prob={blended_eos_prob:.4f}")
                print("[DEBUG] =====================================\n")
        else:
            # No blending: use decoder logits directly
            next_token_logits = decoder_logits_step
        
        # STRICT EOS SUPPRESSION: Set logits to -inf if below eos_suppress_len
        generated_count = len(new_tokens)
        eos_suppressed = False
        if generated_count < eos_suppress_len:
            next_token_logits[:, eos_id] = float("-inf")
            eos_suppressed = True
            if generation_config.debug and step <= 3:
                print(f"[DEBUG] Step {step}: EOS suppressed (generated_count={generated_count} < eos_suppress_len={eos_suppress_len})")
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float("inf")
        
        # Apply top-p (nucleus) filtering
        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = -float("inf")
        
        # Sample next token
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        next_token_id = next_token.item()
        
        # Debug: Log sampled token and EOS probability
        if generation_config.debug and (step <= 10 or next_token_id == eos_id):
            eos_prob = probs[0, eos_id].item()
            suppression_msg = " (after EOS suppression)" if eos_suppressed else ""
            print(f"[DEBUG] Step {step}: sampled_id={next_token_id}, raw_eos_prob={raw_eos_prob:.4f}, eos_prob={eos_prob:.4f}{suppression_msg}")
        
        # Check for EOS
        if next_token_id == eos_id:
            if generated_count >= eos_suppress_len:
                # EOS sampled at/after eos_suppress_len: STOP (do NOT append EOS)
                eos_sampled_at_step = step
                eos_event_probs = probs  # Save for top-k summary
                if generation_config.debug:
                    print(f"[DEBUG] EOS sampled at step {step}; stopping (EOS NOT appended). Total appended tokens: {generated_count}")
                break
            else:
                # EOS sampled before eos_suppress_len: should not happen due to suppression
                if generation_config.debug:
                    print(f"[DEBUG] WARNING: EOS sampled at step {step} despite suppression (generated_count={generated_count} < eos_suppress_len={eos_suppress_len})")
                # Mask out EOS and resample
                next_token_logits[:, eos_id] = float("-inf")
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                next_token_id = next_token.item()
                if generation_config.debug:
                    print(f"[DEBUG] Resampled token at step {step}: id={next_token_id}")
        
        # Append to generated sequence (new_tokens does NOT include EOS)
        new_tokens.append(next_token_id)
        
        # Append to decoder input for next step
        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
    
    # Debug: Print EOS top-k summary if EOS stopped generation
    if generation_config.debug and eos_sampled_at_step is not None and eos_event_probs is not None:
        print(f"\n[DEBUG] ===== EOS Event Top-10 Summary (step {eos_sampled_at_step}) =====")
        top_probs, top_indices = torch.topk(eos_event_probs[0], k=min(10, eos_event_probs.size(1)))
        for i, (prob, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist())):
            # Decode single token if possible
            try:
                piece = tokenizer.decode([idx], skip_special_tokens=False)
            except:
                piece = "<decode_error>"
            print(f"[DEBUG]   {i+1}. token_id={idx:5d}  prob={prob:.4f}  piece='{piece}'")
        print("[DEBUG] ================================================\n")
    
    # Debug: Print generation summary
    if generation_config.debug:
        print(f"[DEBUG] ===== Generation Summary =====")
        print(f"[DEBUG] Total appended tokens: {len(new_tokens)} (max_new={max_new_tokens})")
        print(f"[DEBUG] Generated token IDs (first 50): {new_tokens[:50]}")
        if repetition_stopped:
            print(f"[DEBUG] Stopped early due to repetition detection")
        elif eos_sampled_at_step is not None:
            print(f"[DEBUG] EOS sampled at step {eos_sampled_at_step} (stopped; EOS NOT appended)")
        else:
            print(f"[DEBUG] EOS was NOT sampled (hit max_new_tokens limit)")
        print("[DEBUG] ==================================\n")
    
    # Restore original n-gram settings (not needed anymore since we don't modify model)
    # Left for compatibility but does nothing
    pass
    
    # Decode generated tokens (excluding BOS, excluding EOS)
    generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return generated_text


def load_model_for_generation(
    checkpoint_path: str,
    device: str = "cuda",
    ngram_prior_path: Optional[str] = None,
) -> tuple[EncoderDecoderModel, InitialsTokenizer, ModelConfig]:
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # ---- Config: support both old and new checkpoint formats ----
    if "model_config" in checkpoint:
        cfg_dict = checkpoint["model_config"]
    elif "config" in checkpoint:
        cfg_dict = checkpoint["config"]
    else:
        raise KeyError("Checkpoint missing 'model_config' and 'config'.")

    # If cfg_dict is already a ModelConfig object, keep it; else reconstruct
    config = cfg_dict if isinstance(cfg_dict, ModelConfig) else ModelConfig(**cfg_dict)

    # ---- Tokenizer path: search with fallbacks ----
    checkpoint_dir = Path(checkpoint_path).parent

    candidates = [
        checkpoint_dir / "spm_model.model",           # preferred: next to checkpoint
        checkpoint_dir.parent / "spm_model.model",    # common: outputs/spm_model.model
    ]

    # If config stores a path, try it too
    spm_from_cfg = getattr(config, "spm_model_path", None)
    if spm_from_cfg:
        candidates.append(Path(spm_from_cfg))

    spm_model_path = next((p for p in candidates if p.exists() and p.is_file()), None)
    if spm_model_path is None:
        raise FileNotFoundError(
            "SentencePiece model not found. Tried:\n" + "\n".join(str(p) for p in candidates)
        )

    tokenizer = InitialsTokenizer(str(spm_model_path))

    # ---- Load n-gram prior if available ----
    from ngrams import load_ngram_prior
    
    ngram_prior = None
    if ngram_prior_path:
        ngram_prior = load_ngram_prior(ngram_prior_path, config.vocab_size, device)
    
    # ---- Create/load model ----
    model = EncoderDecoderModel(config, ngram_prior=ngram_prior)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"[OK] Model loaded successfully")
    print(f"  Config: d_model={config.d_model}, layers={config.n_encoder_layers}/{config.n_decoder_layers}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Tokenizer: {spm_model_path}")
    if ngram_prior:
        print(f"  N-gram prior: enabled")
    else:
        print(f"  N-gram prior: disabled")

    return model, tokenizer, config



def interactive_generation(
    model: EncoderDecoderModel,
    tokenizer: InitialsTokenizer,
    device: str = "cuda",
    debug: bool = False,
    generation_config: Optional[GenerationConfig] = None,
    model_config: Optional[ModelConfig] = None,
) -> None:
    """
    Interactive generation loop.
    
    User can enter prompts in initials space and see generated continuations.
    Handles user exit gracefully (empty line, /quit, Ctrl+C, Ctrl+D).
    """
    print("\n" + "="*60)
    print("Interactive Generation (Encoder-Decoder Causal Mode)")
    print("="*60)
    print("Enter prompts in initials space (e.g., 'zh g sh h')")
    print("Commands:")
    print("  /temp <value>       - Set temperature (default: 0.8)")
    print("  /topk <value>       - Set top-k (default: None)")
    print("  /topp <value>       - Set top-p (default: 0.9)")
    print("  /max_new <value>    - Set max new tokens (default: 50)")
    print("  /minlen <value>     - Set min length (default: 0)")
    print("  /alpha <value>|none - Set n-gram alpha override (default: none/checkpoint)")
    print("  /priortemp <value>  - Set prior temperature (default: 1.5)")
    print("  /no_ngram on|off    - Disable/enable n-gram blending (default: off)")
    print("  /quit or empty line - Exit")
    print("  Ctrl+C or Ctrl+D    - Exit")
    print("="*60 + "\n")
    
    # Use provided generation_config or create default
    if generation_config is None:
        generation_config = GenerationConfig(debug=debug)
    
    # Default parameters (updated to reduce token soup)
    temperature = 0.8
    top_k = None
    top_p = 0.9
    max_new_tokens = 50
    min_len = 0
    
    # N-gram blending parameters (PATCH 2)
    disable_ngram = False
    ngram_alpha_override = None  # None means use checkpoint/model default
    prior_temp = 1.5
    
    while True:
        try:
            prompt = input("Prompt: ").strip()
            
            # Handle empty line - exit gracefully
            if not prompt:
                print("Goodbye!")
                return
            
            # Handle commands
            if prompt.startswith("/"):
                parts = prompt.split()
                cmd = parts[0].lower()
                
                if cmd == "/quit":
                    print("Goodbye!")
                    return
                elif cmd == "/temp" and len(parts) > 1:
                    temperature = float(parts[1])
                    print(f"  Temperature set to {temperature}")
                    continue
                elif cmd == "/topk" and len(parts) > 1:
                    top_k = int(parts[1]) if parts[1].lower() != "none" else None
                    print(f"  Top-k set to {top_k}")
                    continue
                elif cmd == "/topp" and len(parts) > 1:
                    top_p = float(parts[1]) if parts[1].lower() != "none" else None
                    print(f"  Top-p set to {top_p}")
                    continue
                elif cmd == "/max_new" and len(parts) > 1:
                    max_new_tokens = int(parts[1])
                    print(f"  Max new tokens set to {max_new_tokens}")
                    continue
                elif cmd == "/minlen" and len(parts) > 1:
                    min_len = int(parts[1])
                    print(f"  Min length set to {min_len}")
                    continue
                elif cmd == "/alpha" and len(parts) > 1:
                    if parts[1].lower() == "none":
                        ngram_alpha_override = None
                        print(f"  N-gram alpha override: none (use checkpoint default)")
                    else:
                        try:
                            ngram_alpha_override = float(parts[1])
                            print(f"  N-gram alpha override set to {ngram_alpha_override}")
                        except ValueError:
                            print(f"  Invalid alpha value: {parts[1]}")
                    continue
                elif cmd == "/priortemp" and len(parts) > 1:
                    try:
                        prior_temp = float(parts[1])
                        print(f"  Prior temperature set to {prior_temp}")
                    except ValueError:
                        print(f"  Invalid prior_temp value: {parts[1]}")
                    continue
                elif cmd == "/no_ngram" and len(parts) > 1:
                    if parts[1].lower() == "on":
                        disable_ngram = True
                        print(f"  N-gram blending: DISABLED")
                    elif parts[1].lower() == "off":
                        disable_ngram = False
                        print(f"  N-gram blending: ENABLED")
                    else:
                        print(f"  Usage: /no_ngram on|off")
                    continue
                else:
                    print("  Unknown command")
                    continue
            
            # Generate (show current settings)
            ngram_status = "disabled" if disable_ngram else f"alpha={ngram_alpha_override if ngram_alpha_override is not None else 'auto'}, prior_temp={prior_temp}"
            print(f"\nGenerating (temp={temperature}, top_k={top_k}, top_p={top_p}, max_new={max_new_tokens}, min_len={min_len}, ngram={ngram_status})...")
            generated = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                device=device,
                min_len=min_len,
                generation_config=generation_config,
                ngram_alpha_override=ngram_alpha_override,
                disable_ngram=disable_ngram,
                eos_suppress_len=None,  # Use min_len as default
                rep_ngram=3,  # Use defaults for interactive mode
                rep_window=30,
                rep_max_hits=3,
                rep_token_run=8,
                min_prompt_tokens=15,
                prior_temp=prior_temp,
                model_config=model_config,
            )
        except EOFError:
            # Handle Ctrl+D (EOF) gracefully
            print("\nGoodbye!")
            return
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate text with trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", type=str, default="", help="Input prompt in initials space")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum NEW tokens to generate")
    parser.add_argument("--min_len", type=int, default=0, help="Minimum generation length (blocks EOS)")
    parser.add_argument("--eos_suppress_len", type=int, default=None, help="Suppress EOS until this many tokens generated (default: min_len)")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (default: 0.8)")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling (default: 0.9)")
    parser.add_argument("--ngram_prior", type=str, default="outputs/ngram_prior.npz", help="Path to n-gram prior")
    parser.add_argument("--ngram_alpha", type=float, default=None, help="Override n-gram prior alpha for generation (default: use checkpoint value)")
    parser.add_argument("--prior_temp", type=float, default=1.5, help="Temperature for N-gram prior softmax (default: 1.5, flattens bigram)")
    parser.add_argument("--no_ngram", action="store_true", help="Disable n-gram prior for generation (even if checkpoint has it)")
    parser.add_argument("--rep_ngram", type=int, default=3, help="N-gram size for repetition detection (default: 3)")
    parser.add_argument("--rep_window", type=int, default=30, help="Window size for repetition check (default: 30)")
    parser.add_argument("--rep_max_hits", type=int, default=3, help="Max n-gram repetitions before stopping (default: 3)")
    parser.add_argument("--rep_token_run", type=int, default=8, help="Max consecutive token repetitions (default: 8)")
    parser.add_argument("--min_prompt_tokens", type=int, default=15, help="Minimum prompt tokens (warn if below, default: 15)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (verbose generation diagnostics)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = "cpu"
    
    # PATCH 3: Conditionally load n-gram prior based on --no_ngram flag
    ngram_prior_path_to_load = None if args.no_ngram else args.ngram_prior
    
    # Load model with n-gram prior
    model, tokenizer, config = load_model_for_generation(
        args.checkpoint, 
        args.device,
        ngram_prior_path=ngram_prior_path_to_load,
    )
    
    # Create generation config
    generation_config = GenerationConfig(debug=args.debug)
    
    if args.interactive:
        # Interactive mode
        interactive_generation(
            model, tokenizer, args.device, 
            generation_config=generation_config,
            model_config=config
        )
    else:
        # Single generation
        print(f"\nPrompt: {args.prompt}")
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device,
            min_len=args.min_len,
            generation_config=generation_config,
            ngram_alpha_override=args.ngram_alpha,
            disable_ngram=args.no_ngram,
            eos_suppress_len=args.eos_suppress_len,
            rep_ngram=args.rep_ngram,
            rep_window=args.rep_window,
            rep_max_hits=args.rep_max_hits,
            rep_token_run=args.rep_token_run,
            min_prompt_tokens=args.min_prompt_tokens,
            prior_temp=args.prior_temp,
            model_config=config,
        )
        print(f"Generated: {generated}\n")


# ===== SMOKE TESTS (optional) =====
# Run with: python generate.py --help (will not execute tests)
# To test helper: uncomment and run directly
"""
def _test_build_ngram_context():
    '''Smoke test for _build_ngram_context helper.'''
    print("Testing _build_ngram_context...")
    
    # Test 1: Empty generation (use last prompt token)
    prev_id = _build_ngram_context([10, 20, 30], [], bos_id=1, device="cpu")
    assert prev_id.shape == (1, 1), f"Expected shape (1,1), got {prev_id.shape}"
    assert prev_id.item() == 30, f"Expected 30, got {prev_id.item()}"
    print("  ✓ Test 1: Empty generation (last prompt token)")
    
    # Test 2: With generated tokens (use last generated token)
    prev_id = _build_ngram_context([10, 20, 30], [40, 50], bos_id=1, device="cpu")
    assert prev_id.shape == (1, 1), f"Expected shape (1,1), got {prev_id.shape}"
    assert prev_id.item() == 50, f"Expected 50, got {prev_id.item()}"
    print("  ✓ Test 2: With generated tokens (last generated token)")
    
    # Test 3: Empty prompt and generation (fallback to BOS)
    prev_id = _build_ngram_context([], [], bos_id=1, device="cpu")
    assert prev_id.shape == (1, 1), f"Expected shape (1,1), got {prev_id.shape}"
    assert prev_id.item() == 1, f"Expected 1, got {prev_id.item()}"
    print("  ✓ Test 3: Empty context (fallback to BOS)")
    
    # Test 4: Only generated tokens (no prompt)
    prev_id = _build_ngram_context([], [100, 200], bos_id=1, device="cpu")
    assert prev_id.shape == (1, 1), f"Expected shape (1,1), got {prev_id.shape}"
    assert prev_id.item() == 200, f"Expected 200, got {prev_id.item()}"
    print("  ✓ Test 4: Only generated tokens (no prompt)")
    
    print("All tests passed! ✓")

# Uncomment to run tests:
# _test_build_ngram_context()
"""
