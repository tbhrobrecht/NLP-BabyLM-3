"""
PyTorch dataset and data loading utilities.

Supports multiple training modes:
- causal: Full sequence generation (encoder sees noised version, decoder predicts next token)
- taskA_prefix_suffix: Prefix→suffix (Task A evaluation mode)
- mixed: Random mix of above
"""
import random
from typing import List, Tuple, Dict, Literal, Optional, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from tokenizer_spm import InitialsTokenizer


class InitialsDataset(Dataset):
    """
    Dataset for initials corpus with multiple training modes.
    """
    
    def __init__(
        self,
        corpus_path: str,
        tokenizer: InitialsTokenizer,
        max_seq_len: int = 512,
        mode: Literal["causal", "taskA_prefix_suffix", "mixed"] = "causal",
        taskA_prob: float = 0.3,
        noise_type: Literal["none", "mask", "drop"] = "mask",
        noise_ratio: float = 0.15,
        split: Literal["train", "val"] = "train",
        train_ratio: float = 0.9,
        random_seed: int = 42,
        min_prompt_len: int = 16,
        min_target_len: int = 16,
        prompt_sampling: Literal["random", "fixed"] = "random",
        fixed_prompt_ratio: float = 0.5,
        fixed_prompt_len: Optional[int] = None,
        validation_prompt_sampling: Optional[Literal["random", "fixed"]] = None,
        validation_fixed_prompt_len: Optional[int] = None,
        validation_fixed_prompt_ratio: Optional[float] = None,
        config: Optional[Any] = None,
        min_new_tokens_train: Optional[int] = None,
        min_new_tokens_eval: Optional[int] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            corpus_path: Path to initials text corpus
            tokenizer: InitialsTokenizer instance
            max_seq_len: Maximum sequence length (longer sequences are truncated)
            mode: Training mode
            taskA_prob: Probability of taskA in mixed mode
            noise_type: Type of noising for causal mode
            noise_ratio: Ratio of tokens to noise
            split: 'train' or 'val'
            train_ratio: Ratio of data for training
            random_seed: Random seed for deterministic splits
            min_prompt_len: Minimum prompt tokens for causal mode
            min_target_len: Minimum target tokens for causal mode
            prompt_sampling: How to sample prompt/target split ("random" or "fixed")
            fixed_prompt_ratio: Ratio for fixed prompt split
            fixed_prompt_len: Exact prompt length (overrides ratio if set)
            validation_prompt_sampling: Override for validation split strategy
            validation_fixed_prompt_len: Override for validation fixed prompt length
            validation_fixed_prompt_ratio: Override for validation fixed prompt ratio
            config: Optional TrainingConfig object for accessing EOS collapse prevention settings
            min_new_tokens_train: Optional override for minimum new tokens during training
            min_new_tokens_eval: Optional override for minimum new tokens during evaluation
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mode = mode
        self.taskA_prob = taskA_prob
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.split = split
        self.min_prompt_len = min_prompt_len
        self.min_target_len = min_target_len
        
        # Store config and min_new_tokens overrides (for EOS collapse prevention)
        self.config = config
        self.min_new_tokens_train = min_new_tokens_train
        self.min_new_tokens_eval = min_new_tokens_eval
        
        # Use validation-specific settings if provided and this is val split
        if split == "val":
            self.prompt_sampling = validation_prompt_sampling if validation_prompt_sampling is not None else "fixed"
            self.fixed_prompt_len = validation_fixed_prompt_len if validation_fixed_prompt_len is not None else fixed_prompt_len
            self.fixed_prompt_ratio = validation_fixed_prompt_ratio if validation_fixed_prompt_ratio is not None else fixed_prompt_ratio
        else:
            self.prompt_sampling = prompt_sampling
            self.fixed_prompt_len = fixed_prompt_len
            self.fixed_prompt_ratio = fixed_prompt_ratio
        
        # Load corpus
        self.lines = self._load_corpus(corpus_path)
        
        # Train/val split
        self.lines = self._split_data(self.lines, train_ratio, random_seed, split)
        
        # Initialize statistics for debugging
        self.stats = {
            "resampled_short_target": 0,
            "dropped_too_short": 0,
        }
        
        # FIX: Precompute sample modes and split points for validation to ensure deterministic behavior
        # This prevents validation metrics from fluctuating due to different random augmentations each epoch
        self.precomputed_sample_modes = None
        self.precomputed_split_points = None
        if split == "val":
            self._precompute_val_samples(random_seed)
        
        print(f"Dataset ({split}): {len(self.lines)} sequences, mode={mode}, prompt_sampling={self.prompt_sampling}")

    def get_stats(self) -> Dict[str, int]:
        """Return dataset statistics."""
        return self.stats.copy()
    
    def _load_corpus(self, path: str) -> List[str]:
        """Load corpus lines."""
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    
    def _split_data(
        self,
        lines: List[str],
        train_ratio: float,
        seed: int,
        split: str
    ) -> List[str]:
        """Deterministic train/val split."""
        rng = random.Random(seed)
        indices = list(range(len(lines)))
        rng.shuffle(indices)
        
        n_train = int(len(indices) * train_ratio)
        
        if split == "train":
            indices = indices[:n_train]
        else:
            indices = indices[n_train:]
        
        return [lines[i] for i in indices]
    
    def _precompute_val_samples(self, seed: int) -> None:
        """
        Precompute sample modes and split points for validation split.
        
        This ensures validation is deterministic across epochs - each validation sample
        will always use the same mode (causal/taskA) and the same split point.
        Without this, validation loss fluctuates randomly due to different augmentations.
        """
        rng = random.Random(seed + 1000)  # Different seed from data split
        
        self.precomputed_sample_modes = []
        self.precomputed_split_points = []
        
        for i, line in enumerate(self.lines):
            # Tokenize to determine split point
            token_ids = self.tokenizer.encode(line, add_bos=False, add_eos=False)
            if len(token_ids) > self.max_seq_len - 2:
                token_ids = token_ids[:self.max_seq_len - 2]
            
            # Determine mode
            if self.mode == "mixed":
                sample_mode = "taskA" if rng.random() < self.taskA_prob else "causal"
            elif self.mode == "taskA_prefix_suffix":
                sample_mode = "taskA"
            else:
                sample_mode = "causal"
            
            # Determine split point based on mode
            if sample_mode == "taskA":
                # TaskA: bounded suffix policy (20-40% of tokens)
                if len(token_ids) >= 2:
                    N = len(token_ids)
                    min_suffix = max(1, int(0.2 * N))
                    max_suffix = max(min_suffix, int(0.4 * N))
                    suffix_len = rng.randint(min_suffix, max_suffix)
                    split_point = N - suffix_len
                else:
                    split_point = 0
            else:
                # Causal: prompt→continuation split
                split_point, is_valid = self._compute_causal_split_point(len(token_ids), rng)

                if not is_valid:
                    # SAFE FALLBACK: ensure target is non-empty
                    # best deterministic choice: split_point = 0 (all tokens go to target)
                    split_point = 0

                # also clamp to prevent split_point == len(token_ids)
                if len(token_ids) == 0:
                    split_point = 0
                else:
                    split_point = max(0, min(split_point, len(token_ids) - 1))
            
            self.precomputed_sample_modes.append(sample_mode)
            self.precomputed_split_points.append(split_point)
    
    def _compute_causal_split_point(self, seq_len: int, rng: Optional[random.Random] = None) -> tuple[int, bool]:
        """
        Compute split point k for causal mode: prompt = S[:k], target = S[k:]
        
        NEW STRATEGY (EOS collapse prevention):
        - Sample target length FIRST (favoring longer targets)
        - Then compute k = seq_len - target_len
        - STRICT ENFORCEMENT: Ensure target_len >= min_target_len
        
        Args:
            seq_len: Length of token sequence (excluding BOS/EOS initially)
            rng: Random number generator (if None, uses global random)
        
        Returns:
            (split_point, is_valid): Split point k and validity flag.
                                     If is_valid=False, sample should be resampled.
        """
        # Handle edge cases - mark as invalid for resampling
        if seq_len < 2:
            return 1, False
        
        # Compute constraints with robust config handling
        min_prompt = self.min_prompt_len
        
        # Build min_target robustly even if config is None
        base_min_target = self.min_target_len
        
        # Try to get min_new_tokens from config if available
        cfg_min_new = None
        if self.config is not None:
            cfg_min_new = getattr(self.config, 'min_new_tokens_train', None)
        
        # Apply overrides and config values
        if self.split == "train" and self.min_new_tokens_train is not None:
            base_min_target = max(base_min_target, self.min_new_tokens_train)
        elif self.split == "val" and self.min_new_tokens_eval is not None:
            base_min_target = max(base_min_target, self.min_new_tokens_eval)
        elif cfg_min_new is not None:
            base_min_target = max(base_min_target, cfg_min_new)
        
        min_target = base_min_target
        
        # Get other config values with safe fallbacks
        max_target = None
        if self.config is not None:
            max_target = getattr(self.config, 'max_target_len', None)
        
        # Compute valid target length range
        min_target_len = max(1, min_target)
        max_target_len = seq_len - min_prompt
        if max_target is not None:
            max_target_len = min(max_target_len, max_target)
        
        # Check if constraints are satisfiable
        if max_target_len < min_target_len:
            # Sequence too short - mark as INVALID for resampling
            # Return a safe split point but flag as invalid
            k = max(min_prompt, seq_len // 2)
            return max(1, min(k, seq_len - 1)), False
        
        # Sample target length
        if self.prompt_sampling == "fixed":
            # Fixed split strategy
            if self.fixed_prompt_len is not None:
                k = self.fixed_prompt_len
            else:
                k = int(seq_len * self.fixed_prompt_ratio)
            # Clamp to valid range
            k = max(min_prompt, min(k, seq_len - min_target_len))
        else:
            # Sample target length with strategy
            strategy = "favor_long"  # Default
            gamma = 2.0  # Default
            
            if self.config is not None:
                strategy = getattr(self.config, 'target_len_strategy', 'favor_long')
                gamma = getattr(self.config, 'favor_long_gamma', 2.0)
            
            if strategy == "favor_long" and max_target_len > min_target_len:
                # Sample with p(t) ∝ (t - min_target_len + 1)^gamma
                # Generate unnormalized weights
                weights = []
                for t in range(min_target_len, max_target_len + 1):
                    weight = (t - min_target_len + 1) ** gamma
                    weights.append(weight)
                
                # Normalize
                total = sum(weights)
                probs = [w / total for w in weights]
                
                # Sample
                if rng is None:
                    target_len = random.choices(
                        range(min_target_len, max_target_len + 1),
                        weights=probs,
                        k=1
                    )[0]
                else:
                    # Emulate random.choices with rng
                    cumsum = []
                    acc = 0.0
                    for p in probs:
                        acc += p
                        cumsum.append(acc)
                    r = rng.random()
                    target_len = min_target_len
                    for i, c in enumerate(cumsum):
                        if r <= c:
                            target_len = min_target_len + i
                            break
            else:
                # Uniform sampling
                if rng is None:
                    target_len = random.randint(min_target_len, max_target_len)
                else:
                    target_len = rng.randint(min_target_len, max_target_len)
            
            # Compute split point
            k = seq_len - target_len
        
        # Clamp to valid range
        k_clamped = max(1, min(k, seq_len - 1))
        
        # Verify the split satisfies minimum target length
        actual_target_len = seq_len - k_clamped
        is_valid = actual_target_len >= min_target_len
        
        return k_clamped, is_valid

    
    def __len__(self) -> int:
        return len(self.lines)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item.
        
        Returns:
            Dictionary with:
            - mode: 'causal' or 'taskA'
            - encoder_input_ids: Token IDs for encoder (B, T_enc)
            - decoder_input_ids: Token IDs for decoder (B, T_dec), shifted right with BOS
            - labels: Target IDs for decoder (B, T_dec)
            - encoder_mask: Attention mask for encoder (B, T_enc)
            - decoder_mask: Attention mask for decoder (B, T_dec)
        """
        text = self.lines[idx]
        
        # Tokenize (without BOS/EOS for now)
        token_ids = self.tokenizer.encode(text, add_bos=False, add_eos=False)
        
        # Truncate if needed
        if len(token_ids) > self.max_seq_len - 2:  # Reserve space for BOS/EOS
            token_ids = token_ids[:self.max_seq_len - 2]
        
        # Decide mode for this sample
        # FIX: Use precomputed values for validation to ensure deterministic behavior
        if self.precomputed_sample_modes is not None:
            # Validation: use precomputed deterministic values
            sample_mode = self.precomputed_sample_modes[idx]
            split_point = self.precomputed_split_points[idx]

            if sample_mode == "causal":
                if len(token_ids) == 0:
                    split_point = 0
                else:
                    split_point = max(0, min(split_point, len(token_ids) - 1))

        else:
            # Training: use random sampling (stochastic augmentation)
            if self.mode == "mixed":
                sample_mode = "taskA" if random.random() < self.taskA_prob else "causal"
            elif self.mode == "taskA_prefix_suffix":
                sample_mode = "taskA"
            else:
                sample_mode = "causal"
            
            # Determine split point based on mode
            if sample_mode == "taskA":
                if len(token_ids) < 2:
                    split_point = 0
                else:
                    N = len(token_ids)
                    min_suffix = max(1, int(0.2 * N))
                    max_suffix = max(min_suffix, int(0.4 * N))
                    suffix_len = random.randint(min_suffix, max_suffix)
                    split_point = N - suffix_len
            else:
                # Causal mode: compute prompt→continuation split
                # STRICT ENFORCEMENT: Resample if target too short
                max_resample_attempts = 10
                
                for attempt in range(max_resample_attempts):
                    # Handle short sequences robustly
                    if len(token_ids) < 2:
                        # Try resampling a different sequence
                        if attempt < max_resample_attempts - 1:
                            new_idx = random.randint(0, len(self.lines) - 1)
                            text = self.lines[new_idx]
                            token_ids = self.tokenizer.encode(text, add_bos=False, add_eos=False)
                            if len(token_ids) > self.max_seq_len - 2:
                                token_ids = token_ids[:self.max_seq_len - 2]
                            self.stats["resampled_short_target"] += 1
                            continue
                        else:
                            # Last attempt: use minimal safe split and break
                            split_point = 1
                            self.stats["dropped_too_short"] += 1
                            break
                    
                    split_point, is_valid = self._compute_causal_split_point(len(token_ids))
                    
                    # Verify target length meets minimum requirement
                    target_len = len(token_ids) - split_point
                    if is_valid and target_len >= self.min_target_len:
                        # Valid sample - use it
                        break
                    
                    # Invalid: resample a different sequence
                    if attempt < max_resample_attempts - 1:
                        new_idx = random.randint(0, len(self.lines) - 1)
                        text = self.lines[new_idx]
                        token_ids = self.tokenizer.encode(text, add_bos=False, add_eos=False)
                        if len(token_ids) > self.max_seq_len - 2:
                            token_ids = token_ids[:self.max_seq_len - 2]
                        self.stats["resampled_short_target"] += 1
                    else:
                        # Last attempt failed: keep current split and mark as dropped
                        self.stats["dropped_too_short"] += 1
                        break
                
                # CRITICAL: Clamp split_point to ensure valid prompt and target
                # Must satisfy: 1 <= split_point <= len(token_ids) - 1
                split_point = max(1, min(split_point, len(token_ids) - 1))
        
        if sample_mode == "taskA":
            # Task A: split into prefix and suffix
            # Encoder: [BOS] prefix [EOS]
            # Decoder: [BOS] suffix [EOS]
            # Labels: suffix [EOS]
            
            if split_point == 0:
                # Handle edge case: too short, use minimal split
                if len(token_ids) >= 2:
                    split_point = 1
                else:
                    split_point = 0
            
            prefix_ids = token_ids[:split_point]
            suffix_ids = token_ids[split_point:]
            
            # Encoder input: [BOS] prefix [EOS]
            encoder_input_ids = [self.tokenizer.bos_id] + prefix_ids + [self.tokenizer.eos_id]
            
            # Decoder input: [BOS] suffix
            decoder_input_ids = [self.tokenizer.bos_id] + suffix_ids
            
            # Labels: suffix [EOS]
            labels = suffix_ids + [self.tokenizer.eos_id]
        
        else:
            # Causal mode: ENCODER-DECODER GENERATIVE OBJECTIVE
            # Given full sequence S = token_ids,
            # split at point k into:
            #   - prompt (source): S[:k] -> encoder input
            #   - continuation (target): S[k:] -> decoder generates this
            #
            # CRITICAL INVARIANT: Encoder sees ONLY prompt, not continuation tokens.
            # This prevents future token leakage and makes training consistent with inference.
            #
            # Construction:
            #   prompt_ids = token_ids[:split_point]
            #   target_ids = token_ids[split_point:]
            #   encoder_input_ids = [BOS] + prompt_ids
            #   decoder_input_ids = [BOS] + target_ids[:-1]  (teacher forcing)
            #   labels = target_ids  (what decoder should predict)
            
            # Handle degenerate cases for very short sequences
            if len(token_ids) < 1:
                # Skip degenerate empty sample deterministically
                # For validation, fallback to next index (cyclic)
                if self.split == "val":
                    return self[(idx + 1) % len(self)]
                else:
                    # For training, resample randomly
                    new_idx = random.randint(0, len(self.lines) - 1)
                    return self[new_idx]
            else:
                # Split sequence (split_point already clamped to valid range)
                prompt_ids = token_ids[:split_point]
                target_ids = token_ids[split_point:]
                
                # Add EOS to target for proper termination
                if len(target_ids) == 0 or target_ids[-1] != self.tokenizer.eos_id:
                    target_ids = target_ids + [self.tokenizer.eos_id]
                
                # EOS collapse prevention: optionally remove EOS from labels
                eos_in_labels = True  # Default
                if self.config is not None:
                    eos_in_labels = getattr(self.config, 'eos_in_labels', True)
                
                if not eos_in_labels and len(target_ids) > 0 and target_ids[-1] == self.tokenizer.eos_id:
                    target_ids = target_ids[:-1]
                    if len(target_ids) == 0:
                        # Edge case: target was only EOS, keep at least one token
                        target_ids = [self.tokenizer.eos_id]
            
            # Encoder input: [BOS] + prompt (source/context only)
            encoder_input_ids = [self.tokenizer.bos_id] + prompt_ids
            
            # Decoder input: [BOS] + target[:-1] (teacher forcing)
            decoder_input_ids = [self.tokenizer.bos_id] + target_ids[:-1]
            
            # Labels: target (may or may not include EOS depending on eos_in_labels)
            labels = target_ids
            
            # Track stats for debug reporting (store in instance variable)
            if not hasattr(self, '_causal_stats'):
                self._causal_stats = {
                    'prompt_lens': [],
                    'target_lens': [],
                    'targets_with_eos': 0,
                    'samples_collected': 0,
                }
            
            if self._causal_stats['samples_collected'] < 1000:
                self._causal_stats['prompt_lens'].append(len(prompt_ids))
                self._causal_stats['target_lens'].append(len(target_ids))
                if len(target_ids) > 0 and target_ids[-1] == self.tokenizer.eos_id:
                    self._causal_stats['targets_with_eos'] += 1
                self._causal_stats['samples_collected'] += 1
                
                # Print stats once when we hit 1000 samples
                debug_enabled = False
                if self.config is not None:
                    debug_enabled = getattr(self.config, 'debug', False)
                
                if self._causal_stats['samples_collected'] == 1000 and debug_enabled:
                    avg_prompt = sum(self._causal_stats['prompt_lens']) / 1000
                    avg_target = sum(self._causal_stats['target_lens']) / 1000
                    eos_pct = self._causal_stats['targets_with_eos'] / 10.0
                    print(f"\n[DEBUG] Causal dataset stats (first 1000 samples):")
                    print(f"  Avg prompt length: {avg_prompt:.1f} tokens")
                    print(f"  Avg target length: {avg_target:.1f} tokens")
                    print(f"  Targets ending with EOS: {eos_pct:.1f}%")
                    print()
        
        # Convert to tensors
        encoder_input_ids_t = torch.tensor(encoder_input_ids, dtype=torch.long)
        decoder_input_ids_t = torch.tensor(decoder_input_ids, dtype=torch.long)
        labels_t = torch.tensor(labels, dtype=torch.long)
        
        # CRITICAL ASSERTION: Verify decoder_input_ids and labels are properly shifted
        # For both causal and taskA modes:
        # decoder_input_ids = [BOS, tok1, tok2, ...]
        # labels = [tok1, tok2, ..., EOS]
        # Therefore: decoder_input_ids[1:] should equal labels[:-1]
        assert len(decoder_input_ids_t) == len(labels_t), \
            f"Length mismatch in {sample_mode}: decoder_input_ids={len(decoder_input_ids_t)}, labels={len(labels_t)}"
        if len(decoder_input_ids_t) > 1:
            # Verify shifting: decoder_input[1:] should match labels[:-1]
            assert torch.equal(decoder_input_ids_t[1:], labels_t[:-1]), \
                f"Shift verification failed in {sample_mode} mode: "\
                f"decoder_input_ids[1:]={decoder_input_ids_t[1:].tolist()[:5]}..., "\
                f"labels[:-1]={labels_t[:-1].tolist()[:5]}..."
            
            # Additional check: decoder_input_ids and labels should NOT be identical
            assert not torch.equal(decoder_input_ids_t, labels_t), \
                f"ERROR in {sample_mode}: decoder_input_ids == labels (NO SHIFTING!)"
        
        # CRITICAL LEAKAGE PREVENTION ASSERTION FOR CAUSAL MODE:
        # In causal mode, encoder must ONLY see prompt tokens, NOT target/continuation tokens
        if sample_mode == "causal" and len(token_ids) >= 1:
            # STRICT CONTENT-BASED LEAKAGE CHECKS (no overlap warnings, only invariants)
            # These invariants (1-4) fully guarantee no leakage. If all pass, data is correct.
            
            # 1) encoder must be exactly [BOS] + prompt_ids
            assert encoder_input_ids_t[0].item() == self.tokenizer.bos_id, \
                "Causal mode: encoder must start with BOS"
            assert torch.equal(encoder_input_ids_t[1:], torch.tensor(prompt_ids, dtype=torch.long)), \
                "Causal mode invariant failed: encoder != [BOS] + prompt"

            # 2) labels must be exactly target_ids (including EOS)
            assert torch.equal(labels_t, torch.tensor(target_ids, dtype=torch.long)), \
                "Causal mode invariant failed: labels != target_ids"

            # 3) decoder shift
            assert decoder_input_ids_t[0].item() == self.tokenizer.bos_id, \
                "Causal mode: decoder_input must start with BOS"
            if len(target_ids) > 1:
                assert torch.equal(decoder_input_ids_t[1:], labels_t[:-1]), \
                    "Causal mode invariant failed: decoder_input != [BOS] + labels[:-1]"

            # 4) reconstruction check: prompt_ids + target_ids_without_eos == token_ids
            target_wo_eos = target_ids[:-1] if len(target_ids) > 0 and target_ids[-1] == self.tokenizer.eos_id else target_ids
            reconstructed = prompt_ids + target_wo_eos
            assert reconstructed == token_ids, \
                f"Causal mode invariant failed: prompt+target does not reconstruct original sequence. "\
                f"prompt={len(prompt_ids)}, target_wo_eos={len(target_wo_eos)}, original={len(token_ids)}"

            # NOTE: encoder_input_ids CAN legitimately equal decoder_input_ids for some samples
            # (e.g., when prompt and target_prefix happen to have the same tokens, or very short sequences).
            # This is NOT leakage - invariants (1-4) already guarantee correctness.
            # Per-sample equality checks removed to avoid false positives.
            # If needed, use batch-level checks in trainer.py debug mode instead.
        
        # FINAL ASSERTION: Verify supervised token count
        # This catches any samples that slipped through with degenerate targets
        supervised_count = len(labels_t)
        
        # Basic check: must have at least 2 supervised tokens (target + EOS)
        assert supervised_count >= 2, \
            f"Degenerate sample in {sample_mode}: labels has only {supervised_count} tokens. "\
            f"decoder_input_ids={decoder_input_ids_t.tolist()[:10]}, "\
            f"labels={labels_t.tolist()[:10]}"
        
        # For causal training samples, enforce min_target_len
        if sample_mode == "causal" and self.split == "train":
            # Count supervised tokens (for causal mode, all labels are supervised)
            # labels = target_ids (which includes EOS if eos_in_labels=True)
            # So supervised_count should be >= min_target_len
            assert supervised_count >= self.min_target_len, \
                f"Causal training sample too short: {supervised_count} < {self.min_target_len}. "\
                f"This should not happen after resampling. idx={idx}, "\
                f"labels={labels_t.tolist()[:20]}"
        
        return {
            "mode": sample_mode,
            "task_type": sample_mode,  # "causal" or "taskA"
            "encoder_input_ids": encoder_input_ids_t,
            "decoder_input_ids": decoder_input_ids_t,
            "labels": labels_t,
        }
    
    def _noise_sequence(self, token_ids: List[int], ratio: float, noise_type: str) -> List[int]:
        """
        Apply noising to a sequence.
        
        Args:
            token_ids: Original token IDs
            ratio: Fraction of tokens to noise
            noise_type: 'mask' (replace with UNK) or 'drop' (remove)
        
        Returns:
            Noised token IDs
        """
        if ratio <= 0:
            return token_ids.copy()
        
        noised = []
        for token_id in token_ids:
            if random.random() < ratio:
                if noise_type == "mask":
                    noised.append(self.tokenizer.unk_id)
                # elif noise_type == "drop": skip (don't append)
            else:
                noised.append(token_id)
        
        return noised
    
    def debug_sample(self, idx: int) -> Dict[str, Any]:
        """
        Debug utility to inspect a sample's mode, split point, and token sequences.
        
        Useful for verifying deterministic validation behavior and diagnosing
        training/validation discrepancies.
        
        Args:
            idx: Sample index
        
        Returns:
            Dictionary with debug information
        """
        sample = self[idx]
        
        return {
            "idx": idx,
            "split": self.split,
            "mode": sample["mode"],
            "split_point": self.precomputed_split_points[idx] if self.precomputed_split_points else None,
            "encoder_input_ids_head": sample["encoder_input_ids"][:20].tolist(),
            "decoder_input_ids_head": sample["decoder_input_ids"][:20].tolist(),
            "labels_head": sample["labels"][:20].tolist(),
            "encoder_length": len(sample["encoder_input_ids"]),
            "decoder_length": len(sample["decoder_input_ids"]),
            "labels_length": len(sample["labels"]),
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    """
    Collate batch with dynamic padding (pads to max length in batch, not global max).
    
    This reduces padding waste and speeds up training by avoiding unnecessary computation
    on padding tokens. Batch dimensions will vary per batch based on actual sequence lengths.
    
    Args:
        batch: List of samples from dataset
        pad_id: Padding token ID (for input_ids)
    
    Returns:
        Batched tensors with dynamic padding and masks
    """
    # Extract sequences (all are 1D tensors of varying lengths)
    encoder_input_ids = [item["encoder_input_ids"] for item in batch]
    decoder_input_ids = [item["decoder_input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    # Extract task types and map to integers: {"causal": 0, "taskA": 1}
    task_types = [item["task_type"] for item in batch]
    task_id_mapping = {"causal": 0, "taskA": 1}
    task_ids = torch.tensor([task_id_mapping[t] for t in task_types], dtype=torch.long)
    
    # Dynamic padding: pad_sequence pads to the max length IN THIS BATCH only
    # This is much more efficient than padding to global max_seq_len
    encoder_input_ids_padded = pad_sequence(encoder_input_ids, batch_first=True, padding_value=pad_id)
    decoder_input_ids_padded = pad_sequence(decoder_input_ids, batch_first=True, padding_value=pad_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # Ignore index for loss
    
    # Create attention masks (1 for real tokens, 0 for padding)
    encoder_mask = (encoder_input_ids_padded != pad_id).long()
    decoder_mask = (decoder_input_ids_padded != pad_id).long()
    
    # Create causal mask for decoder (lower triangular)
    # NOTE: This is a lower triangular matrix with 1s (allows attending to current and past)
    # model.py will invert this to upper triangular boolean (True = ignore future)
    # which is the format expected by PyTorch's Transformer layers
    T_dec = decoder_input_ids_padded.size(1)
    causal_mask = torch.tril(torch.ones(T_dec, T_dec)).unsqueeze(0)  # (1, T, T)
    
    return {
        "encoder_input_ids": encoder_input_ids_padded,
        "decoder_input_ids": decoder_input_ids_padded,
        "labels": labels_padded,
        "encoder_mask": encoder_mask,
        "decoder_mask": decoder_mask,
        "causal_mask": causal_mask,
        "task_id": task_ids,  # (B,) tensor with 0=causal, 1=taskA
    }


def create_dataloaders(
    corpus_path: str,
    tokenizer: InitialsTokenizer,
    batch_size: int,
    max_seq_len: int,
    mode: str,
    taskA_prob: float,
    noise_type: str,
    noise_ratio: float,
    train_ratio: float,
    random_seed: int,
    num_workers: int = 0,
    val_mode: Optional[str] = None,  # Separate validation mode
    min_prompt_len: int = 16,
    min_target_len: int = 16,
    prompt_sampling: str = "random",
    fixed_prompt_ratio: float = 0.5,
    fixed_prompt_len: Optional[int] = None,
    validation_prompt_sampling: Optional[str] = None,
    validation_fixed_prompt_len: Optional[int] = None,
    validation_fixed_prompt_ratio: Optional[float] = None,
    config: Optional[Any] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders with dynamic padding.
    
    Args:
        val_mode: Optional separate mode for validation. If None, uses same mode as training.
        validation_prompt_sampling: Override prompt sampling strategy for validation
        validation_fixed_prompt_len: Override fixed prompt length for validation
        validation_fixed_prompt_ratio: Override fixed prompt ratio for validation
    
    Returns:
        (train_loader, val_loader)
    """
    # Use separate val_mode if provided, otherwise default to same as train mode
    if val_mode is None:
        val_mode = mode
    
    # Train dataset
    train_dataset = InitialsDataset(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        mode=mode,
        taskA_prob=taskA_prob,
        noise_type=noise_type,
        noise_ratio=noise_ratio,
        split="train",
        train_ratio=train_ratio,
        random_seed=random_seed,
        min_prompt_len=min_prompt_len,
        min_target_len=min_target_len,
        prompt_sampling=prompt_sampling,
        fixed_prompt_ratio=fixed_prompt_ratio,
        fixed_prompt_len=fixed_prompt_len,
        config=config,
    )
    
    # Val dataset (use val_mode and validation-specific strategies)
    val_dataset = InitialsDataset(
        corpus_path=corpus_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        mode=val_mode,  # Use separate validation mode
        taskA_prob=taskA_prob,
        noise_type="none",  # No noising for validation
        noise_ratio=0.0,
        split="val",
        train_ratio=train_ratio,
        random_seed=random_seed,
        min_prompt_len=min_prompt_len,
        min_target_len=min_target_len,
        prompt_sampling=prompt_sampling,
        fixed_prompt_ratio=fixed_prompt_ratio,
        fixed_prompt_len=fixed_prompt_len,
        validation_prompt_sampling=validation_prompt_sampling,
        validation_fixed_prompt_len=validation_fixed_prompt_len,
        validation_fixed_prompt_ratio=validation_fixed_prompt_ratio,
        config=config,
    )
    
    # Create dataloaders with dynamic padding collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_id),
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_id),
        num_workers=num_workers,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    print("=== Dataset Smoke Test ===\n")
    
    # Create test corpus
    test_corpus = Path("test_dataset_corpus.txt")
    with open(test_corpus, "w", encoding="utf-8") as f:
        for i in range(20):
            f.write(f"zh g sh h ch h w a\n")
    
    # Create tokenizer (mock)
    from tokenizer_spm import InitialsTokenizer, train_sentencepiece, TokenizerConfig
    
    tok_config = TokenizerConfig(vocab_size=12)
    train_sentencepiece(str(test_corpus), "test_dataset_spm", tok_config)
    tokenizer = InitialsTokenizer("test_dataset_spm.model")
    
    # Create dataset
    print("Creating dataset...")
    dataset = InitialsDataset(
        corpus_path=str(test_corpus),
        tokenizer=tokenizer,
        max_seq_len=64,
        mode="mixed",
        split="train",
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test getting items
    print("\nTest samples:")
    for i in range(3):
        sample = dataset[i]
        print(f"  Sample {i}:")
        print(f"    Mode: {sample['mode']}")
        print(f"    Encoder input shape: {sample['encoder_input_ids'].shape}")
        print(f"    Decoder input shape: {sample['decoder_input_ids'].shape}")
        print(f"    Labels shape: {sample['labels'].shape}")
    
    # Test dataloader
    print("\nTest dataloader:")
    loader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_id),
    )
    
    batch = next(iter(loader))
    print(f"  Batch encoder_input_ids: {batch['encoder_input_ids'].shape}")
    print(f"  Batch decoder_input_ids: {batch['decoder_input_ids'].shape}")
    print(f"  Batch labels: {batch['labels'].shape}")
    print(f"  Batch encoder_mask: {batch['encoder_mask'].shape}")
    print(f"  Batch causal_mask: {batch['causal_mask'].shape}")
    
    # Cleanup
    import os
    for ext in [".model", ".vocab"]:
        try:
            os.remove(f"test_dataset_spm{ext}")
        except:
            pass
    try:
        os.remove(test_corpus)
    except:
        pass
    
    print("\n[OK] Dataset module smoke test passed")
