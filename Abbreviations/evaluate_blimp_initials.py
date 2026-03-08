"""
BLiMP-style Minimal Pair Evaluation for Pinyin-Initials Models

This script evaluates BLiMP-style minimal pairs using custom Mandarin pinyin-initials 
encoder-decoder or decoder-only models. It supports:
- JSONL datasets with "sentence_good" and "sentence_bad" fields
- SentencePiece tokenization (unigram/BPE)
- Custom PyTorch checkpoints (non-HuggingFace)
- Both decoder-only LM and encoder-decoder conditional LM modes

JSONL Format:
Each line should be a JSON object with at least:
  {
    "sentence_good": "wm sh sm shh",   # grammatical sentence (space-separated tokens)
    "sentence_bad": "wm sh sm shh",    # ungrammatical sentence
    "prompt": "wm sh"                  # (optional, encdec mode only) shared prompt
  }

Scoring:
- Decoder mode: Compare P(sentence) via next-token prediction NLL
  - Lower NLL = higher probability = preferred
  - Accuracy = fraction where good_nll < bad_nll

- Encoder-decoder mode: Compare P(target | prompt) 
  - If "prompt" field exists: use it as encoder input, sentence_good/bad as decoder target
  - Else: split sentence into prefix (encoder) and suffix (decoder)
    - Prefix = first min(8, len//2) tokens
    - Ensures good/bad use identical encoder input for fair comparison
  - Accuracy = fraction where good_nll < bad_nll

Usage:
  # Decoder-only mode
  python evaluate_blimp_initials.py \
    --mode decoder \
    --checkpoint outputs/checkpoints/best_model.pt \
    --spm_model outputs/spm_model.model \
    --data_path blimp_jsonl \
    --batch_size 64

  # Encoder-decoder mode
  python evaluate_blimp_initials.py \
    --mode encdec \
    --checkpoint outputs/checkpoints/best_model.pt \
    --spm_model outputs/spm_model.model \
    --data_path blimp_jsonl \
    --batch_size 32

  # Auto-discover all .jsonl files in directory (NEW)
  python evaluate_blimp_initials.py \
    --mode decoder \
    --checkpoint outputs/checkpoints/best_model.pt \
    --spm_model outputs/spm_model.model \
    --data_path outputs/converted/blimp \
    --subset_mode auto \
    --batch_size 64

  # Use standard BLiMP subset list (backwards compatible)
  python evaluate_blimp_initials.py \
    --mode decoder \
    --checkpoint outputs/checkpoints/best_model.pt \
    --spm_model outputs/spm_model.model \
    --data_path outputs/converted/blimp \
    --subset_mode standard \
    --batch_size 64
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import sentencepiece as spm

# Import model and config from local repo
from model import EncoderDecoderModel
from config import ModelConfig


class SentencePieceTokenizer:
    """Wrapper around SentencePiece for encoding/decoding."""
    
    def __init__(self, model_path: str):
        """Load SentencePiece model."""
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(str(model_path))
        
        # Get special token IDs
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.vocab_size = self.sp.GetPieceSize()
        
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs (no special tokens added)."""
        return self.sp.EncodeAsIds(text)
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.sp.DecodeIds(ids)
    
    def id_to_piece(self, token_id: int) -> str:
        """Convert token ID to piece string."""
        return self.sp.IdToPiece(token_id)


def pick_first_present_str(example: dict, keys: List[str]) -> Optional[str]:
    """
    Return the first non-empty string value found among the given keys.
    
    Args:
        example: Dictionary to search in
        keys: List of key names to check in order
    
    Returns:
        First non-empty string found, or None if all are missing/empty
    """
    for key in keys:
        if key in example:
            value = example[key]
            if value and isinstance(value, str) and value.strip():
                return value
    return None


class BlimpDataset(Dataset):
    """Dataset for BLiMP minimal pairs."""
    
    def __init__(
        self, 
        jsonl_path: str,
        tokenizer: SentencePieceTokenizer,
        mode: str = "decoder",
        max_len: int = 1024,
        score_scope: str = "full",
        window_left: int = 3,
        window_right: int = 3,
        prefix_len: int = 10,
    ):
        """
        Load BLiMP-style JSONL dataset.
        
        Args:
            jsonl_path: Path to .jsonl file
            tokenizer: SentencePiece tokenizer
            mode: "decoder" or "encdec"
            max_len: Maximum sequence length (for truncation warning)
            score_scope: "full", "diff_window", or "prefix"
            window_left: Tokens to include before diff region
            window_right: Tokens to include after diff region
            prefix_len: Number of tokens to score in prefix mode
        """
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_len = max_len
        self.score_scope = score_scope
        self.window_left = window_left
        self.window_right = window_right
        self.prefix_len = prefix_len
        self.examples = []
        self.truncation_count = 0
        
        # Mask statistics for diff_window diagnostics
        self.mask_stats = {
            "diff_window_calls": 0,
            "diff_window_fallback_full": 0,
            "diff_window_used_window": 0,
            "scored_positions_good": 0,
            "scored_positions_bad": 0,
        }
        self._warned_sp_alignment = False
        self._printed_hanzi_key_debug = False
        
        # Load JSONL
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                example = json.loads(line)
                
                # Extract required fields (support both naming conventions)
                # Prefer sentence_good_initials/sentence_bad_initials if available
                if 'sentence_good_initials' in example and 'sentence_bad_initials' in example:
                    # Normalize field names to sentence_good/sentence_bad
                    example['sentence_good'] = example['sentence_good_initials']
                    example['sentence_bad'] = example['sentence_bad_initials']
                
                if 'sentence_good' not in example or 'sentence_bad' not in example:
                    warnings.warn(f"Skipping example without sentence_good/sentence_bad or sentence_good_initials/sentence_bad_initials: {example}")
                    continue
                
                # Extract Hanzi fields if present (support multiple naming conventions)
                # Priority order: sentence_*_hanzi (primary), *_hanzi, sentence_*_zh, *_zh, then raw good/bad if CJK
                good_hanzi_keys = [
                    'sentence_good_hanzi', 'good_hanzi', 'sentence_good_zh', 'good_zh'
                ]
                bad_hanzi_keys = [
                    'sentence_bad_hanzi', 'bad_hanzi', 'sentence_bad_zh', 'bad_zh'
                ]
                
                good_hanzi = pick_first_present_str(example, good_hanzi_keys)
                bad_hanzi = pick_first_present_str(example, bad_hanzi_keys)
                
                # Debug print for first example to show which keys were found
                if not self._printed_hanzi_key_debug:
                    found_good_key = None
                    for key in good_hanzi_keys:
                        if key in example and example[key]:
                            found_good_key = key
                            break
                    found_bad_key = None
                    for key in bad_hanzi_keys:
                        if key in example and example[key]:
                            found_bad_key = key
                            break
                    
                    if found_good_key or found_bad_key:
                        print(f"  [hanzi_keys] First example: good_hanzi from '{found_good_key}', bad_hanzi from '{found_bad_key}'")
                    else:
                        print(f"  [hanzi_keys] First example: No Hanzi fields found (checked: {good_hanzi_keys})")
                    self._printed_hanzi_key_debug = True
                
                # Store Hanzi fields in example dict for later access
                example['good_hanzi'] = good_hanzi
                example['bad_hanzi'] = bad_hanzi
                
                self.examples.append(example)
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Get tokenized example.
        
        Returns dict with keys depending on mode:
        - Decoder mode: good_enc_ids, good_enc_mask, good_dec_inputs, good_labels, good_dec_mask,
                       bad_enc_ids, bad_enc_mask, bad_dec_inputs, bad_labels, bad_dec_mask
                       (encoder inputs are IDENTICAL for good/bad to prevent leakage)
        - Encdec mode: good_enc_ids, good_dec_inputs, good_dec_labels,
                       bad_enc_ids, bad_dec_inputs, bad_dec_labels
        """
        example = self.examples[idx]
        good_text = example['sentence_good']
        bad_text = example['sentence_bad']
        
        if self.mode == "decoder":
            # Decoder mode: proper encoder-decoder setup to prevent leakage
            # Encoder gets shared prompt (or minimal BOS), decoder does autoregressive scoring
            
            # Get shared prompt if available
            prompt = example.get('prompt', '')
            
            # Build inputs for good and bad (encoder_input_ids will be identical)
            good_enc_ids, good_enc_mask, good_dec_inputs, good_labels = build_decoder_mode_inputs(
                good_text, self.tokenizer, prompt
            )
            bad_enc_ids, bad_enc_mask, bad_dec_inputs, bad_labels = build_decoder_mode_inputs(
                bad_text, self.tokenizer, prompt
            )
            
            # CRITICAL: Ensure encoder inputs are IDENTICAL for good/bad
            # (they should be by construction, but verify)
            assert good_enc_ids == bad_enc_ids, "Encoder inputs must be identical for fair comparison"
            
            # Check truncation
            if len(good_dec_inputs) > self.max_len or len(bad_dec_inputs) > self.max_len:
                self.truncation_count += 1
            
            # Decoder masks: 1 for real tokens, 0 for padding (will be added by collate_fn)
            good_dec_mask = [1] * len(good_dec_inputs)
            bad_dec_mask = [1] * len(bad_dec_inputs)
            
            # Compute score masks based on scope (aligned to LABEL positions)
            good_score_mask = self._compute_score_mask(
                good_text, bad_text, len(good_labels), is_good=True
            )
            bad_score_mask = self._compute_score_mask(
                good_text, bad_text, len(bad_labels), is_good=False
            )
            
            return {
                'good_enc_ids': good_enc_ids,
                'good_enc_mask': good_enc_mask,
                'good_dec_inputs': good_dec_inputs,
                'good_labels': good_labels,
                'good_dec_mask': good_dec_mask,
                'bad_enc_ids': bad_enc_ids,
                'bad_enc_mask': bad_enc_mask,
                'bad_dec_inputs': bad_dec_inputs,
                'bad_labels': bad_labels,
                'bad_dec_mask': bad_dec_mask,
                'good_score_mask': good_score_mask,
                'bad_score_mask': bad_score_mask,
                'good_hanzi': example.get('good_hanzi', None),
                'bad_hanzi': example.get('bad_hanzi', None),
            }
        
        elif self.mode == "encdec":
            # Encoder-decoder: P(target | prompt)
            
            # Check if prompt is provided
            if 'prompt' in example and example['prompt']:
                # Use explicit prompt as encoder input
                prompt = example['prompt']
                prompt_tokens = self.tokenizer.encode(prompt)
                
                # Encoder input: [BOS] + prompt + [EOS] (standardized format)
                shared_enc_ids = [self.tokenizer.bos_id] + prompt_tokens + [self.tokenizer.eos_id]
                
                # Decoder target: full sentence
                good_tokens = self.tokenizer.encode(good_text)
                bad_tokens = self.tokenizer.encode(bad_text)
                
                # Decoder sequences: [BOS] + tokens + [EOS]
                good_dec_ids = [self.tokenizer.bos_id] + good_tokens + [self.tokenizer.eos_id]
                bad_dec_ids = [self.tokenizer.bos_id] + bad_tokens + [self.tokenizer.eos_id]
                
                # Suffix texts for score mask computation (full target in explicit prompt case)
                suffix_good_text = good_text
                suffix_bad_text = bad_text
                
            else:
                # No prompt: use longest common prefix (LCP) for BLiMP fairness
                # BLiMP minimal pairs require IDENTICAL encoder context to ensure
                # differences are isolated to the decoder-predicted region only.
                
                # Tokenize both sentences (no BOS/EOS)
                good_tokens = self.tokenizer.encode(good_text)
                bad_tokens = self.tokenizer.encode(bad_text)
                
                # Find longest common prefix length in SentencePiece token space
                lcp_len = 0
                for i in range(min(len(good_tokens), len(bad_tokens))):
                    if good_tokens[i] == bad_tokens[i]:
                        lcp_len = i + 1
                    else:
                        break
                
                # Also compute LCP in space-token domain for suffix text extraction
                good_words = good_text.split()
                bad_words = bad_text.split()
                lcp_len_space = 0
                for i in range(min(len(good_words), len(bad_words))):
                    if good_words[i] == bad_words[i]:
                        lcp_len_space = i + 1
                    else:
                        break
                
                # Shared encoder input: [BOS] + common_prefix + [EOS]
                shared_enc_ids = [self.tokenizer.bos_id] + good_tokens[:lcp_len] + [self.tokenizer.eos_id]
                
                # Decoder targets: suffixes after common prefix
                good_suffix = good_tokens[lcp_len:]
                bad_suffix = bad_tokens[lcp_len:]
                
                # Decoder sequences: [BOS] + suffix + [EOS]
                good_dec_ids = [self.tokenizer.bos_id] + good_suffix + [self.tokenizer.eos_id]
                bad_dec_ids = [self.tokenizer.bos_id] + bad_suffix + [self.tokenizer.eos_id]
                
                # Suffix texts for score mask computation (suffix in space-token domain)
                suffix_good_text = " ".join(good_words[lcp_len_space:])
                suffix_bad_text = " ".join(bad_words[lcp_len_space:])
            
            # Check truncation
            if len(shared_enc_ids) > self.max_len or len(good_dec_ids) > self.max_len or len(bad_dec_ids) > self.max_len:
                self.truncation_count += 1
            
            # Compute score masks aligned to decoder LABEL positions
            # Masks operate on the suffix text that the decoder is predicting
            good_dec_labels = good_dec_ids[1:]
            bad_dec_labels = bad_dec_ids[1:]
            
            good_score_mask = self._compute_score_mask(
                suffix_good_text, suffix_bad_text, len(good_dec_labels), is_good=True
            )
            bad_score_mask = self._compute_score_mask(
                suffix_good_text, suffix_bad_text, len(bad_dec_labels), is_good=False
            )
            
            # Decoder inputs (for computing attention masks in collate_fn)
            good_dec_inputs = good_dec_ids[:-1]
            bad_dec_inputs = bad_dec_ids[:-1]
            
            # Decoder masks: 1 for real tokens, 0 for padding (will be added by collate_fn)
            good_dec_mask = [1] * len(good_dec_inputs)
            bad_dec_mask = [1] * len(bad_dec_inputs)
            
            # Return IDENTICAL encoder inputs for both good and bad
            return {
                'good_enc_ids': shared_enc_ids,
                'good_dec_inputs': good_dec_inputs,
                'good_dec_labels': good_dec_labels,
                'good_dec_mask': good_dec_mask,
                'bad_enc_ids': shared_enc_ids,
                'bad_dec_inputs': bad_dec_inputs,
                'bad_dec_labels': bad_dec_labels,
                'bad_dec_mask': bad_dec_mask,
                'good_score_mask': good_score_mask,
                'bad_score_mask': bad_score_mask,
                'good_hanzi': example.get('good_hanzi', None),
                'bad_hanzi': example.get('bad_hanzi', None),
            }
    
    def _build_sp_token_boundaries(self, text: str) -> Optional[List[Tuple[int, int]]]:
        """
        Build exact mapping from space-separated tokens to SentencePiece token spans.
        
        Uses whitespace-aware encoding to match SentencePiece's ▁ (space) tokenization.
        
        Args:
            text: Space-separated input text
        
        Returns:
            List of (start_idx, end_idx) for each space token, or None if alignment fails
            Each tuple represents the span [start_idx, end_idx) in the full SentencePiece encoding
        """
        space_tokens = text.split()
        if not space_tokens:
            return None
        
        # Encode full text to verify consistency
        full_ids = self.tokenizer.encode(text)
        
        # Build boundaries by encoding each space token with whitespace awareness
        boundaries = []
        cumulative_count = 0
        
        for i, space_token in enumerate(space_tokens):
            # CRITICAL: SentencePiece tokenization depends on preceding whitespace
            # First token: no preceding space
            # Subsequent tokens: include preceding space
            if i == 0:
                piece_text = space_token
            else:
                piece_text = " " + space_token
            
            sp_ids = self.tokenizer.encode(piece_text)
            start_idx = cumulative_count
            end_idx = cumulative_count + len(sp_ids)
            boundaries.append((start_idx, end_idx))
            cumulative_count = end_idx
        
        # Verify consistency: whitespace-aware encoding should match full encoding
        if cumulative_count != len(full_ids):
            if not self._warned_sp_alignment:
                warnings.warn(
                    f"SentencePiece token alignment mismatch: "
                    f"whitespace-aware encoding sums to {cumulative_count}, "
                    f"but full encoding has {len(full_ids)} tokens. "
                    f"Falling back to full sequence scoring for diff_window. "
                    f"This warning will only be shown once."
                )
                self._warned_sp_alignment = True
            return None
        
        return boundaries
    
    def _compute_score_mask(self, good_text: str, bad_text: str, seq_len: int, is_good: bool) -> List[int]:
        """
        Compute score mask for a sequence based on score_scope.
        
        Args:
            good_text: Good sentence text (space-separated)
            bad_text: Bad sentence text (space-separated)
            seq_len: Length of tokenized sequence (labels length)
            is_good: Whether this is the good sentence
        
        Returns:
            List of 1s and 0s indicating which positions to score
        """
        if self.score_scope == "full":
            # Score all positions
            return [1] * seq_len
        
        elif self.score_scope == "prefix":
            # Score only first prefix_len tokens
            mask = [1 if i < self.prefix_len else 0 for i in range(seq_len)]
            return mask
        
        elif self.score_scope == "diff_window":
            # Track diff_window usage for diagnostics
            self.mask_stats["diff_window_calls"] += 1
            
            # Use EXACT SentencePiece token alignment for diff_window scoring
            # 
            # Why exact alignment is critical for BLiMP minimal pairs:
            # - Minimal pairs differ by 1-2 space-tokens, often subword changes
            # - Proportional heuristics (tokens_per_word scaling) misalign boundaries
            #   by several SentencePiece tokens, causing:
            #     * Scoring irrelevant tokens (noise)
            #     * Missing the actual difference (signal loss)
            #     * BLiMP accuracies collapsing toward chance (~50%)
            # - Exact alignment ensures we score only the true differing region
            # 
            # Why heuristics are unsafe for SentencePiece:
            # - Unigram/BPE tokenization is highly non-uniform
            # - A single space-token may produce 1-5+ subword pieces
            # - Proportional scaling cannot capture this variability
            
            # Find differing region in space-separated text
            first_diff, last_diff = find_diff_region(good_text, bad_text)
            
            # Select text and build exact boundaries
            text = good_text if is_good else bad_text
            boundaries = self._build_sp_token_boundaries(text)
            
            # Validate boundaries and diff indices
            if boundaries is None or first_diff >= len(boundaries) or last_diff >= len(boundaries):
                # Safe fallback: score full sequence
                self.mask_stats["diff_window_fallback_full"] += 1
                # Track scored positions for fallback
                if is_good:
                    self.mask_stats["scored_positions_good"] += seq_len
                else:
                    self.mask_stats["scored_positions_bad"] += seq_len
                return [1] * seq_len
            
            # Alignment succeeded
            self.mask_stats["diff_window_used_window"] += 1
            
            # Get SentencePiece token span for the differing region
            start_token = boundaries[first_diff][0]  # Start of first differing space-token
            end_token = boundaries[last_diff][1]     # End of last differing space-token (exclusive)
            
            # Apply window in SentencePiece token space
            full_ids = self.tokenizer.encode(text)
            num_sp_tokens = len(full_ids)
            
            window_start = max(0, start_token - self.window_left)
            window_end = min(num_sp_tokens, end_token + self.window_right)
            
            # Build score_mask aligned to LABEL positions
            # Labels = cand_ids = ids_raw + [EOS]
            # full_ids = ids_raw (does NOT include EOS)
            # seq_len = len(labels) = len(full_ids) + 1
            #
            # Mask strategy:
            # - Mask position i if i < len(full_ids) AND window_start <= i < window_end
            # - EOS position (i == len(full_ids)) is masked out (set to 0) because it's
            #   not part of the content-level diff; it's a structural token
            mask = []
            for i in range(seq_len):
                if i < num_sp_tokens and window_start <= i < window_end:
                    mask.append(1)
                else:
                    mask.append(0)
            
            # Track scored positions
            num_scored = sum(mask)
            if is_good:
                self.mask_stats["scored_positions_good"] += num_scored
            else:
                self.mask_stats["scored_positions_bad"] += num_scored
            
            return mask
        
        else:
            # Unknown scope, score everything
            return [1] * seq_len


def padding_collate_fn_decoder(batch: List[Dict], pad_id: int, max_len: int, left_pad: bool = False) -> Dict[str, torch.Tensor]:
    """
    Collate function for decoder mode (uses encoder-decoder structure).
    Pads encoder and decoder inputs separately.
    
    Args:
        batch: List of dicts with keys: good_enc_ids, good_enc_mask, good_dec_inputs, good_labels, good_dec_mask,
                                        bad_enc_ids, bad_enc_mask, bad_dec_inputs, bad_labels, bad_dec_mask
        pad_id: Padding token ID
        max_len: Maximum sequence length (truncate if needed)
        left_pad: Whether to pad on left (default: right padding)
    
    Returns:
        Dict with padded tensors and attention masks
    """
    # Find max lengths for encoder and decoder
    max_good_enc_len = min(max_len, max(len(ex['good_enc_ids']) for ex in batch))
    max_good_dec_len = min(max_len, max(len(ex['good_dec_inputs']) for ex in batch))
    max_bad_enc_len = min(max_len, max(len(ex['bad_enc_ids']) for ex in batch))
    max_bad_dec_len = min(max_len, max(len(ex['bad_dec_inputs']) for ex in batch))
    
    B = len(batch)
    
    # Initialize tensors
    good_enc_ids = torch.full((B, max_good_enc_len), pad_id, dtype=torch.long)
    good_enc_mask = torch.zeros((B, max_good_enc_len), dtype=torch.long)
    good_dec_inputs = torch.full((B, max_good_dec_len), pad_id, dtype=torch.long)
    good_labels = torch.full((B, max_good_dec_len), -100, dtype=torch.long)  # -100 = ignore index
    good_dec_mask = torch.zeros((B, max_good_dec_len), dtype=torch.long)
    
    bad_enc_ids = torch.full((B, max_bad_enc_len), pad_id, dtype=torch.long)
    bad_enc_mask = torch.zeros((B, max_bad_enc_len), dtype=torch.long)
    bad_dec_inputs = torch.full((B, max_bad_dec_len), pad_id, dtype=torch.long)
    bad_labels = torch.full((B, max_bad_dec_len), -100, dtype=torch.long)
    bad_dec_mask = torch.zeros((B, max_bad_dec_len), dtype=torch.long)
    
    good_score_masks = torch.zeros((B, max_good_dec_len), dtype=torch.long)
    bad_score_masks = torch.zeros((B, max_bad_dec_len), dtype=torch.long)
    
    # Fill with data
    for i, ex in enumerate(batch):
        good_enc_len = min(max_len, len(ex['good_enc_ids']))
        good_dec_len = min(max_len, len(ex['good_dec_inputs']))
        bad_enc_len = min(max_len, len(ex['bad_enc_ids']))
        bad_dec_len = min(max_len, len(ex['bad_dec_inputs']))
        score_mask_good_len = min(max_len, len(ex.get('good_score_mask', [])))
        score_mask_bad_len = min(max_len, len(ex.get('bad_score_mask', [])))
        
        if left_pad:
            good_enc_ids[i, -good_enc_len:] = torch.LongTensor(ex['good_enc_ids'][:good_enc_len])
            good_enc_mask[i, -good_enc_len:] = torch.LongTensor(ex['good_enc_mask'][:good_enc_len])
            good_dec_inputs[i, -good_dec_len:] = torch.LongTensor(ex['good_dec_inputs'][:good_dec_len])
            good_labels[i, -good_dec_len:] = torch.LongTensor(ex['good_labels'][:good_dec_len])
            good_dec_mask[i, -good_dec_len:] = torch.LongTensor(ex['good_dec_mask'][:good_dec_len])
            
            bad_enc_ids[i, -bad_enc_len:] = torch.LongTensor(ex['bad_enc_ids'][:bad_enc_len])
            bad_enc_mask[i, -bad_enc_len:] = torch.LongTensor(ex['bad_enc_mask'][:bad_enc_len])
            bad_dec_inputs[i, -bad_dec_len:] = torch.LongTensor(ex['bad_dec_inputs'][:bad_dec_len])
            bad_labels[i, -bad_dec_len:] = torch.LongTensor(ex['bad_labels'][:bad_dec_len])
            bad_dec_mask[i, -bad_dec_len:] = torch.LongTensor(ex['bad_dec_mask'][:bad_dec_len])
            
            if 'good_score_mask' in ex:
                good_score_masks[i, -score_mask_good_len:] = torch.LongTensor(ex['good_score_mask'][:score_mask_good_len])
            if 'bad_score_mask' in ex:
                bad_score_masks[i, -score_mask_bad_len:] = torch.LongTensor(ex['bad_score_mask'][:score_mask_bad_len])
        else:
            good_enc_ids[i, :good_enc_len] = torch.LongTensor(ex['good_enc_ids'][:good_enc_len])
            good_enc_mask[i, :good_enc_len] = torch.LongTensor(ex['good_enc_mask'][:good_enc_len])
            good_dec_inputs[i, :good_dec_len] = torch.LongTensor(ex['good_dec_inputs'][:good_dec_len])
            good_labels[i, :good_dec_len] = torch.LongTensor(ex['good_labels'][:good_dec_len])
            good_dec_mask[i, :good_dec_len] = torch.LongTensor(ex['good_dec_mask'][:good_dec_len])
            
            bad_enc_ids[i, :bad_enc_len] = torch.LongTensor(ex['bad_enc_ids'][:bad_enc_len])
            bad_enc_mask[i, :bad_enc_len] = torch.LongTensor(ex['bad_enc_mask'][:bad_enc_len])
            bad_dec_inputs[i, :bad_dec_len] = torch.LongTensor(ex['bad_dec_inputs'][:bad_dec_len])
            bad_labels[i, :bad_dec_len] = torch.LongTensor(ex['bad_labels'][:bad_dec_len])
            bad_dec_mask[i, :bad_dec_len] = torch.LongTensor(ex['bad_dec_mask'][:bad_dec_len])
            
            if 'good_score_mask' in ex:
                good_score_masks[i, :score_mask_good_len] = torch.LongTensor(ex['good_score_mask'][:score_mask_good_len])
            if 'bad_score_mask' in ex:
                bad_score_masks[i, :score_mask_bad_len] = torch.LongTensor(ex['bad_score_mask'][:score_mask_bad_len])
    
    return {
        'good_enc_ids': good_enc_ids,
        'good_enc_mask': good_enc_mask,
        'good_dec_inputs': good_dec_inputs,
        'good_labels': good_labels,
        'good_dec_mask': good_dec_mask,
        'bad_enc_ids': bad_enc_ids,
        'bad_enc_mask': bad_enc_mask,
        'bad_dec_inputs': bad_dec_inputs,
        'bad_labels': bad_labels,
        'bad_dec_mask': bad_dec_mask,
        'good_score_mask': good_score_masks,
        'bad_score_mask': bad_score_masks,
    }


def padding_collate_fn_encdec(batch: List[Dict], pad_id: int, max_len: int, left_pad: bool = False) -> Dict[str, torch.Tensor]:
    """
    Collate function for encoder-decoder mode.
    Pads encoder/decoder inputs and labels separately.
    
    Args:
        batch: List of dicts with keys: good_enc_ids, good_dec_inputs, good_dec_labels,
                                         bad_enc_ids, bad_dec_inputs, bad_dec_labels
        pad_id: Padding token ID
        max_len: Maximum sequence length
        left_pad: Whether to pad on left
    
    Returns:
        Dict with padded tensors and attention masks
    """
    # Find max lengths
    max_good_enc_len = min(max_len, max(len(ex['good_enc_ids']) for ex in batch))
    max_good_dec_len = min(max_len, max(len(ex['good_dec_inputs']) for ex in batch))
    max_bad_enc_len = min(max_len, max(len(ex['bad_enc_ids']) for ex in batch))
    max_bad_dec_len = min(max_len, max(len(ex['bad_dec_inputs']) for ex in batch))
    
    B = len(batch)
    
    # Initialize tensors
    good_enc_ids = torch.full((B, max_good_enc_len), pad_id, dtype=torch.long)
    good_dec_inputs = torch.full((B, max_good_dec_len), pad_id, dtype=torch.long)
    good_dec_labels = torch.full((B, max_good_dec_len), -100, dtype=torch.long)
    
    bad_enc_ids = torch.full((B, max_bad_enc_len), pad_id, dtype=torch.long)
    bad_dec_inputs = torch.full((B, max_bad_dec_len), pad_id, dtype=torch.long)
    bad_dec_labels = torch.full((B, max_bad_dec_len), -100, dtype=torch.long)
    
    good_score_masks = torch.zeros((B, max_good_dec_len), dtype=torch.long)
    bad_score_masks = torch.zeros((B, max_bad_dec_len), dtype=torch.long)
    
    # Fill with data
    for i, ex in enumerate(batch):
        good_enc_len = min(max_len, len(ex['good_enc_ids']))
        good_dec_len = min(max_len, len(ex['good_dec_inputs']))
        bad_enc_len = min(max_len, len(ex['bad_enc_ids']))
        bad_dec_len = min(max_len, len(ex['bad_dec_inputs']))
        score_mask_good_len = min(max_len, len(ex.get('good_score_mask', [])))
        score_mask_bad_len = min(max_len, len(ex.get('bad_score_mask', [])))
        
        if left_pad:
            good_enc_ids[i, -good_enc_len:] = torch.LongTensor(ex['good_enc_ids'][:good_enc_len])
            good_dec_inputs[i, -good_dec_len:] = torch.LongTensor(ex['good_dec_inputs'][:good_dec_len])
            good_dec_labels[i, -good_dec_len:] = torch.LongTensor(ex['good_dec_labels'][:good_dec_len])
            
            bad_enc_ids[i, -bad_enc_len:] = torch.LongTensor(ex['bad_enc_ids'][:bad_enc_len])
            bad_dec_inputs[i, -bad_dec_len:] = torch.LongTensor(ex['bad_dec_inputs'][:bad_dec_len])
            bad_dec_labels[i, -bad_dec_len:] = torch.LongTensor(ex['bad_dec_labels'][:bad_dec_len])
            
            if 'good_score_mask' in ex:
                good_score_masks[i, -score_mask_good_len:] = torch.LongTensor(ex['good_score_mask'][:score_mask_good_len])
            if 'bad_score_mask' in ex:
                bad_score_masks[i, -score_mask_bad_len:] = torch.LongTensor(ex['bad_score_mask'][:score_mask_bad_len])
        else:
            good_enc_ids[i, :good_enc_len] = torch.LongTensor(ex['good_enc_ids'][:good_enc_len])
            good_dec_inputs[i, :good_dec_len] = torch.LongTensor(ex['good_dec_inputs'][:good_dec_len])
            good_dec_labels[i, :good_dec_len] = torch.LongTensor(ex['good_dec_labels'][:good_dec_len])
            
            bad_enc_ids[i, :bad_enc_len] = torch.LongTensor(ex['bad_enc_ids'][:bad_enc_len])
            bad_dec_inputs[i, :bad_dec_len] = torch.LongTensor(ex['bad_dec_inputs'][:bad_dec_len])
            bad_dec_labels[i, :bad_dec_len] = torch.LongTensor(ex['bad_dec_labels'][:bad_dec_len])
            
            if 'good_score_mask' in ex:
                good_score_masks[i, :score_mask_good_len] = torch.LongTensor(ex['good_score_mask'][:score_mask_good_len])
            if 'bad_score_mask' in ex:
                bad_score_masks[i, :score_mask_bad_len] = torch.LongTensor(ex['bad_score_mask'][:score_mask_bad_len])
    
    # Attention masks
    good_enc_mask = (good_enc_ids != pad_id).long()
    good_dec_mask = (good_dec_inputs != pad_id).long()
    bad_enc_mask = (bad_enc_ids != pad_id).long()
    bad_dec_mask = (bad_dec_inputs != pad_id).long()
    
    return {
        'good_enc_ids': good_enc_ids,
        'good_enc_mask': good_enc_mask,
        'good_dec_inputs': good_dec_inputs,
        'good_dec_labels': good_dec_labels,
        'good_dec_mask': good_dec_mask,
        'bad_enc_ids': bad_enc_ids,
        'bad_enc_mask': bad_enc_mask,
        'bad_dec_inputs': bad_dec_inputs,
        'bad_dec_labels': bad_dec_labels,
        'bad_dec_mask': bad_dec_mask,
        'good_score_mask': good_score_masks,
        'bad_score_mask': bad_score_masks,
    }


def find_diff_region(good_text: str, bad_text: str) -> Tuple[int, int]:
    """
    Find first and last differing token positions between two sentences.
    
    Args:
        good_text: Good sentence (space-separated tokens)
        bad_text: Bad sentence (space-separated tokens)
    
    Returns:
        (first_diff, last_diff) - indices of first and last differing tokens
        Returns (0, max_len) if no differences or to handle length mismatches
    """
    good_tokens = good_text.split()
    bad_tokens = bad_text.split()
    
    max_len = max(len(good_tokens), len(bad_tokens))
    min_len = min(len(good_tokens), len(bad_tokens))
    
    # Find first diff
    first_diff = 0
    for i in range(min_len):
        if good_tokens[i] != bad_tokens[i]:
            first_diff = i
            break
    else:
        # No differences in common prefix, diff is in length mismatch
        first_diff = min_len
    
    # Find last diff (search backwards)
    last_diff = max_len - 1
    if len(good_tokens) == len(bad_tokens):
        for i in range(min_len - 1, -1, -1):
            if good_tokens[i] != bad_tokens[i]:
                last_diff = i
                break
    
    return first_diff, last_diff


def build_decoder_mode_inputs(
    text: str,
    tokenizer: SentencePieceTokenizer,
    prompt: Optional[str] = None,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Build encoder and decoder inputs for decoder mode evaluation.
    
    This prevents information leakage by ensuring:
    - Encoder receives only shared prompt (or minimal [BOS][EOS])
    - Decoder receives proper shifted inputs for autoregressive scoring
    
    Args:
        text: Candidate sentence to score
        tokenizer: SentencePiece tokenizer
        prompt: Optional shared prompt context (identical for good/bad pairs)
    
    Returns:
        (encoder_input_ids, encoder_mask, decoder_input_ids, labels)
        - encoder_input_ids: Shared prompt or [BOS][EOS] (identical for good/bad)
        - encoder_mask: Attention mask for encoder (all 1s)
        - decoder_input_ids: [BOS] + cand_ids[:-1] (shifted)
        - labels: cand_ids (targets for NLL computation)
    """
    # Tokenize candidate WITHOUT auto-adding BOS/EOS
    ids_raw = tokenizer.encode(text)
    
    # Build cand_ids: add EOS at end (consistent with training convention)
    cand_ids = ids_raw + [tokenizer.eos_id]
    
    # Encoder input: shared prompt or minimal [BOS][EOS]
    # Standardized format: [BOS] + content + [EOS]
    if prompt and prompt.strip():
        # Use provided prompt as encoder context
        prompt_ids = tokenizer.encode(prompt)
        encoder_input_ids = [tokenizer.bos_id] + prompt_ids + [tokenizer.eos_id]
    else:
        # No prompt: use minimal constant context [BOS][EOS]
        encoder_input_ids = [tokenizer.bos_id, tokenizer.eos_id]
    
    # Encoder mask: all 1s for valid tokens
    encoder_mask = [1] * len(encoder_input_ids)
    
    # Decoder input: [BOS] + cand_ids[:-1] (shifted right)
    decoder_input_ids = [tokenizer.bos_id] + cand_ids[:-1]
    
    # Labels: cand_ids (what we want to predict)
    labels = cand_ids
    
    return encoder_input_ids, encoder_mask, decoder_input_ids, labels


def compute_sequence_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    score_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute per-sequence negative log-likelihood.
    
    Args:
        logits: (B, T, V) model output logits
        labels: (B, T) target labels
        ignore_index: Label value to ignore (typically -100 for padding)
        score_mask: (B, T) optional mask specifying which positions to score
                    (1 = score, 0 = ignore). If None, score all valid positions.
    
    Returns:
        (B,) per-sequence mean NLL over scored positions
    """
    B, T, V = logits.shape
    
    # Compute token-level cross-entropy loss (no reduction)
    loss_fn = nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)
    token_loss = loss_fn(logits.view(-1, V), labels.view(-1))  # (B*T,)
    token_loss = token_loss.view(B, T)  # (B, T)
    
    # Mask for valid (non-ignored) positions
    valid_mask = (labels != ignore_index).float()  # (B, T)
    
    # Apply score mask if provided
    if score_mask is not None:
        valid_mask = valid_mask * score_mask.float()
    
    # Compute mean loss per sequence (over valid positions only)
    sequence_nll = (token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)  # (B,)
    
    return sequence_nll


def compute_token_losses(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100
) -> torch.Tensor:
    """
    Compute per-token cross-entropy losses.
    
    Args:
        logits: (B, T, V) model output logits
        labels: (B, T) target labels
        ignore_index: Label value to ignore (typically -100 for padding)
    
    Returns:
        (B, T) per-token CE loss with ignore_index positions set to 0
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


def compute_key_tokens(
    good_token_losses: torch.Tensor,
    bad_token_losses: torch.Tensor,
    good_labels: torch.Tensor,
    bad_labels: torch.Tensor,
    score_mask_good: Optional[torch.Tensor],
    score_mask_bad: Optional[torch.Tensor],
    topk: int,
    tokenizer: SentencePieceTokenizer,
    ignore_index: int = -100
) -> List[List[Dict]]:
    """
    Compute top-K tokens with largest contribution to decision gap.
    
    Args:
        good_token_losses: (B, T) per-token losses for good sequences
        bad_token_losses: (B, T) per-token losses for bad sequences
        good_labels: (B, T) labels for good sequences
        bad_labels: (B, T) labels for bad sequences
        score_mask_good: (B, T) or None, mask for good sequences
        score_mask_bad: (B, T) or None, mask for bad sequences
        topk: Number of top tokens to return
        tokenizer: SentencePiece tokenizer for decoding
        ignore_index: Label value to ignore
    
    Returns:
        List of length B, where each element is a list of dicts with keys:
        - rank: int (1-indexed)
        - pos: int (position in sequence)
        - good_token_id: int
        - bad_token_id: int
        - good_piece: str (decoded piece from good sequence)
        - bad_piece: str (decoded piece from bad sequence)
        - good_loss: float
        - bad_loss: float
        - contribution: float (bad_loss - good_loss)
    """
    B = good_token_losses.shape[0]
    results = []
    
    for b in range(B):
        # Determine maximum comparable length per sample
        T = min(good_labels.size(1), bad_labels.size(1))
        
        # Build boolean valid vectors by POSITION
        contributions = []
        for i in range(T):
            # Check if position i is valid in both sequences (not ignore_index)
            good_ok = (good_labels[b, i] != ignore_index)
            bad_ok = (bad_labels[b, i] != ignore_index)
            
            # If score masks provided, require mask[b,i] == 1
            if score_mask_good is not None:
                good_ok = good_ok and (score_mask_good[b, i] > 0)
            if score_mask_bad is not None:
                bad_ok = bad_ok and (score_mask_bad[b, i] > 0)
            
            # Only consider positions where BOTH are valid
            if good_ok and bad_ok:
                good_loss = good_token_losses[b, i].item()
                bad_loss = bad_token_losses[b, i].item()
                contribution = bad_loss - good_loss
                
                # Get token info from BOTH sequences (may differ!)
                good_token_id = int(good_labels[b, i].item())
                bad_token_id = int(bad_labels[b, i].item())
                
                contributions.append({
                    'pos': i,
                    'good_token_id': good_token_id,
                    'bad_token_id': bad_token_id,
                    'good_piece': tokenizer.id_to_piece(good_token_id),
                    'bad_piece': tokenizer.id_to_piece(bad_token_id),
                    'good_loss': good_loss,
                    'bad_loss': bad_loss,
                    'contribution': contribution,
                })
        
        # Sort by absolute contribution (descending)
        # This captures both strong good-favoring (negative contribution) 
        # and bad-favoring (positive contribution) tokens
        contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        
        # Take top-K
        top_contribs = contributions[:topk]
        
        # Add rank
        for rank, item in enumerate(top_contribs, start=1):
            item['rank'] = rank
        
        results.append(top_contribs)
    
    return results


def make_filename_safe(name: str) -> str:
    """
    Convert phenomenon/subset name to filesystem-safe filename.
    
    Args:
        name: Original name
    
    Returns:
        Filesystem-safe name (replace '/' and whitespace with '_')
    """
    return name.replace('/', '_').replace(' ', '_')


def compute_margin_summary(margin_stats: Dict) -> Dict:
    """
    Compute summary statistics for margin/confidence analysis.
    
    Args:
        margin_stats: Dict with 'deltas', 'margins', and 'correct' lists
    
    Returns:
        Dict with summary statistics
    """
    import numpy as np
    
    deltas = np.array(margin_stats['deltas'])
    margins = np.array(margin_stats['margins'])
    correct = np.array(margin_stats['correct'], dtype=bool)
    
    n = len(deltas)
    if n == 0:
        return {
            'mean_delta': float('nan'),
            'mean_margin': float('nan'),
            'median_margin': float('nan'),
            'mean_delta_correct': float('nan'),
            'mean_delta_incorrect': float('nan'),
            'pct_low_margin': float('nan'),
            'acc_margin_ge_0_1': float('nan'),
            'acc_margin_ge_0_5': float('nan'),
            'acc_margin_ge_1_0': float('nan'),
        }
    
    # Basic stats
    mean_delta = float(np.mean(deltas))
    mean_margin = float(np.mean(margins))
    median_margin = float(np.median(margins))
    
    # Delta by correctness
    if correct.sum() > 0:
        mean_delta_correct = float(np.mean(deltas[correct]))
    else:
        mean_delta_correct = float('nan')
    
    if (~correct).sum() > 0:
        mean_delta_incorrect = float(np.mean(deltas[~correct]))
    else:
        mean_delta_incorrect = float('nan')
    
    # Low margin percentage
    pct_low_margin = float(100.0 * (margins < 0.1).sum() / n)
    
    # Accuracy at margin thresholds
    def acc_at_threshold(threshold):
        mask = margins >= threshold
        if mask.sum() == 0:
            return float('nan')
        return float(correct[mask].sum() / mask.sum())
    
    acc_margin_ge_0_1 = acc_at_threshold(0.1)
    acc_margin_ge_0_5 = acc_at_threshold(0.5)
    acc_margin_ge_1_0 = acc_at_threshold(1.0)
    
    return {
        'mean_delta': mean_delta,
        'mean_margin': mean_margin,
        'median_margin': median_margin,
        'mean_delta_correct': mean_delta_correct,
        'mean_delta_incorrect': mean_delta_incorrect,
        'pct_low_margin': pct_low_margin,
        'acc_margin_ge_0_1': acc_margin_ge_0_1,
        'acc_margin_ge_0_5': acc_margin_ge_0_5,
        'acc_margin_ge_1_0': acc_margin_ge_1_0,
    }


def evaluate_decoder(
    model: EncoderDecoderModel,
    dataloader: DataLoader,
    device: torch.device,
    tokenizer: Optional[SentencePieceTokenizer] = None,
    subset_name: Optional[str] = None,
    txt_file: Optional[object] = None,
    jsonl_file: Optional[object] = None,
    topk_tokens: int = 10,
    save_mispred_only: bool = False,
    score_scope: str = "full",
    dataset_examples: Optional[List[Dict]] = None,
) -> Tuple[float, int, int, int, int, Dict]:
    """
    Evaluate BLiMP in decoder mode (fixed to prevent information leakage).
    
    Decoder mode now uses proper encoder-decoder structure:
    - Encoder receives shared prompt (or minimal BOS) - IDENTICAL for good/bad
    - Decoder performs autoregressive scoring with shifted inputs
    - No future information leaks through cross-attention
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with batches
        device: Device to run on
        tokenizer: Optional tokenizer for logging
        subset_name: Optional subset/phenomenon name for logging
        txt_file: Optional file handle for text logs
        jsonl_file: Optional file handle for JSONL logs
        topk_tokens: Number of top contributing tokens to log
        save_mispred_only: Only log mispredictions
        score_scope: Scoring scope (for logging)
        dataset_examples: Optional list of original examples (for logging metadata)
    
    Returns:
        (accuracy, correct_count, total_count, correct_flipped, total_flipped, margin_stats)
        flipped tracks inverted polarity (good_nll > bad_nll) for diagnostics
        margin_stats: dict with deltas, margins, and correct arrays for confidence analysis
    """
    model.eval()
    
    correct = 0
    correct_flipped = 0
    total = 0
    
    # Track sample index within subset for logging
    sample_idx = 0
    
    # Track margin statistics for confidence analysis
    all_deltas = []
    all_margins = []
    all_correct = []
    
    # Check if logging is enabled
    logging_enabled = (txt_file is not None and jsonl_file is not None and 
                      tokenizer is not None and dataset_examples is not None)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            good_enc_ids = batch['good_enc_ids'].to(device)
            good_enc_mask = batch['good_enc_mask'].to(device)
            good_dec_inputs = batch['good_dec_inputs'].to(device)
            good_labels = batch['good_labels'].to(device)
            good_dec_mask = batch['good_dec_mask'].to(device)
            
            bad_enc_ids = batch['bad_enc_ids'].to(device)
            bad_enc_mask = batch['bad_enc_mask'].to(device)
            bad_dec_inputs = batch['bad_dec_inputs'].to(device)
            bad_labels = batch['bad_labels'].to(device)
            bad_dec_mask = batch['bad_dec_mask'].to(device)
            
            good_score_mask = batch.get('good_score_mask', None)
            bad_score_mask = batch.get('bad_score_mask', None)
            
            if good_score_mask is not None:
                good_score_mask = good_score_mask.to(device)
            if bad_score_mask is not None:
                bad_score_mask = bad_score_mask.to(device)
            
            # CRITICAL: Verify encoder inputs are identical (prevent leakage)
            assert torch.equal(good_enc_ids, bad_enc_ids), "Encoder inputs must be identical for good/bad pairs"
            assert torch.equal(good_enc_mask, bad_enc_mask), "Encoder masks must be identical for good/bad pairs"
            
            # Forward pass - good
            # Encoder: shared prompt/BOS (no candidate info)
            # Decoder: autoregressive scoring of candidate
            good_logits, _ = model(
                encoder_input_ids=good_enc_ids,
                decoder_input_ids=good_dec_inputs,
                encoder_mask=good_enc_mask,
                decoder_mask=good_dec_mask,
                causal_mask=None,  # Will be created inside model
            )
            
            # Forward pass - bad
            bad_logits, _ = model(
                encoder_input_ids=bad_enc_ids,
                decoder_input_ids=bad_dec_inputs,
                encoder_mask=bad_enc_mask,
                decoder_mask=bad_dec_mask,
                causal_mask=None,
            )
            
            # Compute NLL over labels (aligned to label positions)
            # Score mask is already aligned to label length
            good_nll = compute_sequence_nll(good_logits, good_labels, score_mask=good_score_mask)  # (B,)
            bad_nll = compute_sequence_nll(bad_logits, bad_labels, score_mask=bad_score_mask)  # (B,)
            
            # Compute token-level losses if logging enabled
            if logging_enabled:
                good_token_losses = compute_token_losses(good_logits, good_labels)  # (B, T)
                bad_token_losses = compute_token_losses(bad_logits, bad_labels)  # (B, T)
                
                # Compute key tokens
                key_tokens_batch = compute_key_tokens(
                    good_token_losses, bad_token_losses,
                    good_labels, bad_labels,
                    good_score_mask, bad_score_mask,
                    topk_tokens, tokenizer
                )
            
            # Count correct predictions (good has lower NLL)
            predictions = (good_nll < bad_nll)
            correct += predictions.sum().item()
            # Count flipped predictions (for polarity inversion diagnostics)
            correct_flipped += (good_nll > bad_nll).sum().item()
            
            # Log per-sample if enabled
            if logging_enabled:
                B = good_nll.shape[0]
                for b in range(B):
                    # Get original example
                    example = dataset_examples[sample_idx]
                    
                    # Extract metadata
                    good_text = example.get('sentence_good', example.get('sentence_good_initials', ''))
                    bad_text = example.get('sentence_bad', example.get('sentence_bad_initials', ''))
                    good_hanzi = example.get('good_hanzi', None)
                    bad_hanzi = example.get('bad_hanzi', None)
                    uid = example.get('uid', None)
                    
                    # Compute metrics
                    good_nll_val = good_nll[b].item()
                    bad_nll_val = bad_nll[b].item()
                    delta = bad_nll_val - good_nll_val
                    margin = abs(delta)
                    pred = "good<bad" if predictions[b] else "good>=bad"
                    correct_bool = predictions[b].item()
                    
                    # Track for margin statistics
                    all_deltas.append(delta)
                    all_margins.append(margin)
                    all_correct.append(correct_bool)
                    
                    # Count scored positions
                    if good_score_mask is not None:
                        num_scored_good = good_score_mask[b].sum().item()
                    else:
                        num_scored_good = (good_labels[b] != -100).sum().item()
                    
                    if bad_score_mask is not None:
                        num_scored_bad = bad_score_mask[b].sum().item()
                    else:
                        num_scored_bad = (bad_labels[b] != -100).sum().item()
                    
                    # Get key tokens
                    key_tokens = key_tokens_batch[b]
                    
                    # Skip if save_mispred_only and prediction is correct
                    if save_mispred_only and correct_bool:
                        sample_idx += 1
                        continue
                    
                    # Write to text file
                    txt_file.write("="*80 + "\n")
                    txt_file.write(f"Sample {sample_idx} | Phenomenon: {subset_name}\n")
                    if uid is not None:
                        txt_file.write(f"UID: {uid}\n")
                    txt_file.write("-"*80 + "\n")
                    txt_file.write(f"Good text:  {good_text}\n")
                    txt_file.write(f"Bad text:   {bad_text}\n")
                    if good_hanzi and isinstance(good_hanzi, str) and good_hanzi.strip():
                        txt_file.write(f"Good Hanzi: {good_hanzi}\n")
                    if bad_hanzi and isinstance(bad_hanzi, str) and bad_hanzi.strip():
                        txt_file.write(f"Bad Hanzi:  {bad_hanzi}\n")
                    txt_file.write("-"*80 + "\n")
                    txt_file.write(f"Good NLL:   {good_nll_val:.4f}\n")
                    txt_file.write(f"Bad NLL:    {bad_nll_val:.4f}\n")
                    txt_file.write(f"Delta:      {delta:.4f} (bad - good)\n")
                    txt_file.write(f"Prediction: {pred}\n")
                    txt_file.write(f"Correct:    {correct_bool}\n")
                    txt_file.write("-"*80 + "\n")
                    txt_file.write(f"Score scope: {score_scope}\n")
                    txt_file.write(f"Scored positions (good): {num_scored_good}\n")
                    txt_file.write(f"Scored positions (bad):  {num_scored_bad}\n")
                    txt_file.write("-"*80 + "\n")
                    txt_file.write("Top contributing tokens:\n")
                    txt_file.write(f"{'Rank':<6} {'Pos':<6} {'GoodTok':<10} {'GoodPiece':<15} {'BadTok':<10} {'BadPiece':<15} {'GoodLoss':<12} {'BadLoss':<12} {'Contrib':<12}\n")
                    for tok in key_tokens:
                        txt_file.write(f"{tok['rank']:<6} {tok['pos']:<6} {tok['good_token_id']:<10} {tok['good_piece']:<15} "
                                     f"{tok['bad_token_id']:<10} {tok['bad_piece']:<15} "
                                     f"{tok['good_loss']:<12.4f} {tok['bad_loss']:<12.4f} {tok['contribution']:<12.4f}\n")
                    txt_file.write("\n")
                    
                    # Write to JSONL file
                    jsonl_entry = {
                        'sample_idx': sample_idx,
                        'phenomenon': subset_name,
                        'uid': uid,
                        'good_text': good_text,
                        'bad_text': bad_text,
                        'good_hanzi': good_hanzi,
                        'bad_hanzi': bad_hanzi,
                        'good_nll': good_nll_val,
                        'bad_nll': bad_nll_val,
                        'delta': delta,
                        'margin': margin,
                        'prediction': pred,
                        'correct': correct_bool,
                        'score_scope': score_scope,
                        'num_scored_good': num_scored_good,
                        'num_scored_bad': num_scored_bad,
                        'top_tokens': key_tokens,
                    }
                    jsonl_file.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
                    
                    sample_idx += 1
            else:
                # No logging, but still track margin statistics
                B = good_nll.shape[0]
                for b in range(B):
                    delta = (bad_nll[b] - good_nll[b]).item()
                    margin = abs(delta)
                    correct_bool = predictions[b].item()
                    all_deltas.append(delta)
                    all_margins.append(margin)
                    all_correct.append(correct_bool)
                sample_idx += B
            
            total += len(good_nll)
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Compute margin statistics  
    margin_stats = {
        'deltas': all_deltas,
        'margins': all_margins,
        'correct': all_correct,
    }
    return accuracy, correct, total, correct_flipped, total, margin_stats


def evaluate_encdec(
    model: EncoderDecoderModel,
    dataloader: DataLoader,
    device: torch.device,
    tokenizer: Optional[SentencePieceTokenizer] = None,
    subset_name: Optional[str] = None,
    txt_file: Optional[object] = None,
    jsonl_file: Optional[object] = None,
    topk_tokens: int = 10,
    save_mispred_only: bool = False,
    score_scope: str = "full",
    dataset_examples: Optional[List[Dict]] = None,
) -> Tuple[float, int, int, int, int, Dict]:
    """
    Evaluate BLiMP in encoder-decoder mode.
    
    Uses encoder for prompt/prefix and decoder for target prediction.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader with batches
        device: Device to run on
        tokenizer: Optional tokenizer for logging
        subset_name: Optional subset/phenomenon name for logging
        txt_file: Optional file handle for text logs
        jsonl_file: Optional file handle for JSONL logs
        topk_tokens: Number of top contributing tokens to log
        save_mispred_only: Only log mispredictions
        score_scope: Scoring scope (for logging)
        dataset_examples: Optional list of original examples (for logging metadata)
    
    Returns:
        (accuracy, correct_count, total_count, correct_flipped, total_flipped, margin_stats)
        flipped tracks inverted polarity (good_nll > bad_nll) for diagnostics
        margin_stats: dict with deltas, margins, and correct arrays for confidence analysis
    """
    model.eval()
    
    correct = 0
    correct_flipped = 0
    total = 0
    
    # Track sample index within subset for logging
    sample_idx = 0
    
    # Track margin statistics for confidence analysis
    all_deltas = []
    all_margins = []
    all_correct = []
    
    # Check if logging is enabled
    logging_enabled = (txt_file is not None and jsonl_file is not None and 
                      tokenizer is not None and dataset_examples is not None)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            good_enc_ids = batch['good_enc_ids'].to(device)
            good_enc_mask = batch['good_enc_mask'].to(device)
            good_dec_inputs = batch['good_dec_inputs'].to(device)
            good_dec_labels = batch['good_dec_labels'].to(device)
            good_dec_mask = batch['good_dec_mask'].to(device)
            
            bad_enc_ids = batch['bad_enc_ids'].to(device)
            bad_enc_mask = batch['bad_enc_mask'].to(device)
            bad_dec_inputs = batch['bad_dec_inputs'].to(device)
            bad_dec_labels = batch['bad_dec_labels'].to(device)
            bad_dec_mask = batch['bad_dec_mask'].to(device)
            
            # Get score masks if present
            good_score_mask = batch.get('good_score_mask', None)
            bad_score_mask = batch.get('bad_score_mask', None)
            
            if good_score_mask is not None:
                good_score_mask = good_score_mask.to(device)
            if bad_score_mask is not None:
                bad_score_mask = bad_score_mask.to(device)
            
            # Forward pass - good
            good_logits, _ = model(
                encoder_input_ids=good_enc_ids,
                decoder_input_ids=good_dec_inputs,
                encoder_mask=good_enc_mask,
                decoder_mask=good_dec_mask,
                causal_mask=None,  # Will be created inside model
            )
            
            # Forward pass - bad
            bad_logits, _ = model(
                encoder_input_ids=bad_enc_ids,
                decoder_input_ids=bad_dec_inputs,
                encoder_mask=bad_enc_mask,
                decoder_mask=bad_dec_mask,
                causal_mask=None,
            )
            
            # Compute NLL on decoder targets with score masks
            good_nll = compute_sequence_nll(good_logits, good_dec_labels, score_mask=good_score_mask)  # (B,)
            bad_nll = compute_sequence_nll(bad_logits, bad_dec_labels, score_mask=bad_score_mask)  # (B,)
            
            # Compute token-level losses if logging enabled
            if logging_enabled:
                good_token_losses = compute_token_losses(good_logits, good_dec_labels)  # (B, T)
                bad_token_losses = compute_token_losses(bad_logits, bad_dec_labels)  # (B, T)
                
                # Compute key tokens
                key_tokens_batch = compute_key_tokens(
                    good_token_losses, bad_token_losses,
                    good_dec_labels, bad_dec_labels,
                    good_score_mask, bad_score_mask,
                    topk_tokens, tokenizer
                )
            
            # Count correct predictions
            predictions = (good_nll < bad_nll)
            correct += predictions.sum().item()
            # Count flipped predictions (for polarity inversion diagnostics)
            correct_flipped += (good_nll > bad_nll).sum().item()
            
            # Log per-sample if enabled
            if logging_enabled:
                B = good_nll.shape[0]
                for b in range(B):
                    # Get original example
                    example = dataset_examples[sample_idx]
                    
                    # Extract metadata
                    good_text = example.get('sentence_good', example.get('sentence_good_initials', ''))
                    bad_text = example.get('sentence_bad', example.get('sentence_bad_initials', ''))
                    good_hanzi = example.get('good_hanzi', example.get('sentence_good_hanzi', ''))
                    bad_hanzi = example.get('bad_hanzi', example.get('sentence_bad_hanzi', ''))
                    uid = example.get('uid', None)
                    
                    # Compute metrics
                    good_nll_val = good_nll[b].item()
                    bad_nll_val = bad_nll[b].item()
                    delta = bad_nll_val - good_nll_val
                    margin = abs(delta)
                    pred = "good<bad" if predictions[b] else "good>=bad"
                    correct_bool = predictions[b].item()
                    
                    # Track for margin statistics
                    all_deltas.append(delta)
                    all_margins.append(margin)
                    all_correct.append(correct_bool)
                    
                    # Count scored positions
                    if good_score_mask is not None:
                        num_scored_good = good_score_mask[b].sum().item()
                    else:
                        num_scored_good = (good_dec_labels[b] != -100).sum().item()
                    
                    if bad_score_mask is not None:
                        num_scored_bad = bad_score_mask[b].sum().item()
                    else:
                        num_scored_bad = (bad_dec_labels[b] != -100).sum().item()
                    
                    # Get key tokens
                    key_tokens = key_tokens_batch[b]
                    
                    # Skip if save_mispred_only and prediction is correct
                    if save_mispred_only and correct_bool:
                        sample_idx += 1
                        continue
                    
                    # Write to text file
                    txt_file.write("="*80 + "\n")
                    txt_file.write(f"Sample {sample_idx} | Phenomenon: {subset_name}\n")
                    if uid is not None:
                        txt_file.write(f"UID: {uid}\n")
                    txt_file.write("-"*80 + "\n")
                    txt_file.write(f"Good text:  {good_text}\n")
                    txt_file.write(f"Bad text:   {bad_text}\n")
                    if good_hanzi is not None:
                        txt_file.write(f"Good Hanzi: {good_hanzi}\n")
                    if bad_hanzi is not None:
                        txt_file.write(f"Bad Hanzi:  {bad_hanzi}\n")
                    txt_file.write("-"*80 + "\n")
                    txt_file.write(f"Good NLL:   {good_nll_val:.4f}\n")
                    txt_file.write(f"Bad NLL:    {bad_nll_val:.4f}\n")
                    txt_file.write(f"Delta:      {delta:.4f} (bad - good)\n")
                    txt_file.write(f"Prediction: {pred}\n")
                    txt_file.write(f"Correct:    {correct_bool}\n")
                    txt_file.write("-"*80 + "\n")
                    txt_file.write(f"Score scope: {score_scope}\n")
                    txt_file.write(f"Scored positions (good): {num_scored_good}\n")
                    txt_file.write(f"Scored positions (bad):  {num_scored_bad}\n")
                    txt_file.write("-"*80 + "\n")
                    txt_file.write("Top contributing tokens:\n")
                    txt_file.write(f"{'Rank':<6} {'Pos':<6} {'GoodTok':<10} {'GoodPiece':<15} {'BadTok':<10} {'BadPiece':<15} {'GoodLoss':<12} {'BadLoss':<12} {'Contrib':<12}\n")
                    for tok in key_tokens:
                        txt_file.write(f"{tok['rank']:<6} {tok['pos']:<6} {tok['good_token_id']:<10} {tok['good_piece']:<15} "
                                     f"{tok['bad_token_id']:<10} {tok['bad_piece']:<15} "
                                     f"{tok['good_loss']:<12.4f} {tok['bad_loss']:<12.4f} {tok['contribution']:<12.4f}\n")
                    txt_file.write("\n")
                    
                    # Write to JSONL file
                    jsonl_entry = {
                        'sample_idx': sample_idx,
                        'phenomenon': subset_name,
                        'uid': uid,
                        'good_text': good_text,
                        'bad_text': bad_text,
                        'good_hanzi': good_hanzi,
                        'bad_hanzi': bad_hanzi,
                        'good_nll': good_nll_val,
                        'bad_nll': bad_nll_val,
                        'delta': delta,
                        'margin': margin,
                        'prediction': pred,
                        'correct': correct_bool,
                        'score_scope': score_scope,
                        'num_scored_good': num_scored_good,
                        'num_scored_bad': num_scored_bad,
                        'top_tokens': key_tokens,
                    }
                    jsonl_file.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
                    
                    sample_idx += 1
            else:
                # No logging, but still track margin statistics
                B = good_nll.shape[0]
                for b in range(B):
                    delta = (bad_nll[b] - good_nll[b]).item()
                    margin = abs(delta)
                    correct_bool = predictions[b].item()
                    all_deltas.append(delta)
                    all_margins.append(margin)
                    all_correct.append(correct_bool)
                sample_idx += B
            
            total += len(good_nll)
    
    accuracy = correct / total if total > 0 else 0.0
    
    # Compute margin statistics
    margin_stats = {
        'deltas': all_deltas,
        'margins': all_margins,
        'correct': all_correct,
    }
    return accuracy, correct, total, correct_flipped, total, margin_stats


class BlimpDatasetFromExamples(BlimpDataset):
    """Dataset for BLiMP minimal pairs from in-memory examples list."""
    
    def __init__(
        self,
        examples: List[Dict],
        tokenizer: SentencePieceTokenizer,
        mode: str = "decoder",
        max_len: int = 1024,
        score_scope: str = "full",
        window_left: int = 3,
        window_right: int = 3,
        prefix_len: int = 10,
    ):
        """
        Create dataset from pre-loaded examples.
        
        Args:
            examples: List of example dicts
            tokenizer: SentencePiece tokenizer
            mode: "decoder" or "encdec"
            max_len: Maximum sequence length
            score_scope: "full", "diff_window", or "prefix"
            window_left: Tokens to include before diff region
            window_right: Tokens to include after diff region
            prefix_len: Number of tokens to score in prefix mode
        """
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_len = max_len
        self.score_scope = score_scope
        self.window_left = window_left
        self.window_right = window_right
        self.prefix_len = prefix_len
        self.truncation_count = 0
        
        # Mask statistics for diff_window diagnostics
        self.mask_stats = {
            "diff_window_calls": 0,
            "diff_window_fallback_full": 0,
            "diff_window_used_window": 0,
            "scored_positions_good": 0,
            "scored_positions_bad": 0,
        }
        self._warned_sp_alignment = False
        self._printed_hanzi_key_debug = False
        
        # Normalize field names for each example
        self.examples = []
        for example in examples:
            # Support both naming conventions
            if 'sentence_good_initials' in example and 'sentence_bad_initials' in example:
                example['sentence_good'] = example['sentence_good_initials']
                example['sentence_bad'] = example['sentence_bad_initials']
            
            if 'sentence_good' not in example or 'sentence_bad' not in example:
                warnings.warn(f"Skipping example without sentence_good/sentence_bad: {example}")
                continue
            
            # Normalize Hanzi fields using same helper
            if 'good_hanzi' not in example or example['good_hanzi'] is None:
                good_hanzi_keys = [
                    'sentence_good_hanzi', 'good_hanzi', 'sentence_good_zh', 'good_zh'
                ]
                example['good_hanzi'] = pick_first_present_str(example, good_hanzi_keys)
            
            if 'bad_hanzi' not in example or example['bad_hanzi'] is None:
                bad_hanzi_keys = [
                    'sentence_bad_hanzi', 'bad_hanzi', 'sentence_bad_zh', 'bad_zh'
                ]
                example['bad_hanzi'] = pick_first_present_str(example, bad_hanzi_keys)
            
            self.examples.append(example)


def discover_subsets_from_dir(data_path: Path) -> List[str]:
    """
    Auto-discover all .jsonl files in a directory (non-recursive).
    
    Args:
        data_path: Path to directory containing .jsonl files
    
    Returns:
        Sorted list of subset names (file stems)
    """
    if not data_path.is_dir():
        return []
    
    # Find all .jsonl files
    jsonl_files = sorted(data_path.glob("*.jsonl"))
    
    # Extract stems as subset names
    subset_names = [f.stem for f in jsonl_files]
    
    return subset_names


def load_single_jsonl_by_phenomenon(jsonl_path: str) -> Dict[str, List[Dict]]:
    """
    Load a single JSONL file and group examples by phenomenon.
    
    Args:
        jsonl_path: Path to JSONL file
    
    Returns:
        Dict mapping phenomenon name -> list of examples
    """
    phenomena_groups = {}
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                example = json.loads(line)
                
                # Get phenomenon name (default to 'default' if missing)
                phenomenon = example.get('phenomenon', 'default')
                
                # Add to group
                if phenomenon not in phenomena_groups:
                    phenomena_groups[phenomenon] = []
                phenomena_groups[phenomenon].append(example)
            
            except json.JSONDecodeError as e:
                warnings.warn(f"Skipping invalid JSON line: {line[:100]}... Error: {e}")
                continue
    
    return phenomena_groups


def load_checkpoint(checkpoint_path: str, config: ModelConfig, device: torch.device) -> EncoderDecoderModel:
    """
    Load model from custom PyTorch checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file
        config: Model configuration
        device: Device to load model on
    
    Returns:
        Loaded model in eval mode
    """
    # Create model
    model = EncoderDecoderModel(config, ngram_prior=None)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict (handle different checkpoint formats)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Assume checkpoint is the state dict itself
        state_dict = checkpoint
    
    # Load state dict
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print(f"[OK] Model loaded from {checkpoint_path}")
    print(f"     Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate BLiMP-style minimal pairs for pinyin-initials model')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pt file)')
    parser.add_argument('--spm_model', type=str, required=True,
                        help='Path to SentencePiece model (.model file)')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Directory containing subset.jsonl files')
    
    # Mode
    parser.add_argument('--mode', type=str, default='decoder', choices=['decoder', 'encdec'],
                        help='Evaluation mode: decoder (LM) or encdec (conditional LM)')
    
    # Optional arguments
    parser.add_argument('--subsets', type=str, default=None,
                        help='Comma-separated list of subsets to evaluate (default: depends on subset_mode)')
    parser.add_argument('--subset_mode', type=str, default='auto', choices=['auto', 'standard'],
                        help='Subset discovery mode: auto (discover all .jsonl files) or standard (use BLIMP_SUBSETS)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--max_len', type=int, default=1024,
                        help='Maximum sequence length (truncate longer sequences)')
    parser.add_argument('--left_pad', action='store_true',
                        help='Use left padding instead of right padding')
    
    # Scoring scope arguments
    parser.add_argument('--score_scope', type=str, default='full',
                        choices=['full', 'diff_window', 'prefix'],
                        help='Scoring scope: full sentence, diff window, or prefix only')
    parser.add_argument('--window_left', type=int, default=3,
                        help='Tokens to include before diff region (for diff_window mode)')
    parser.add_argument('--window_right', type=int, default=3,
                        help='Tokens to include after diff region (for diff_window mode)')
    parser.add_argument('--prefix_len', type=int, default=10,
                        help='Number of tokens to score (for prefix mode)')
    
    # Model config (optional overrides)
    parser.add_argument('--vocab_size', type=int, default=None,
                        help='Vocabulary size (default: from tokenizer)')
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--n_encoder_layers', type=int, default=6,
                        help='Number of encoder layers')
    parser.add_argument('--n_decoder_layers', type=int, default=6,
                        help='Number of decoder layers')
    
    # Logging arguments
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for detailed per-sample logs (if None, logging disabled)')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name for this run (default: auto timestamp)')
    parser.add_argument('--topk_tokens', type=int, default=10,
                        help='Number of top contributing tokens to show in logs')
    parser.add_argument('--save_mispred_only', action='store_true',
                        help='Only log mispredicted samples (where model is wrong)')
    
    args = parser.parse_args()
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {args.spm_model}...")
    tokenizer = SentencePieceTokenizer(args.spm_model)
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  PAD: {tokenizer.pad_id}, BOS: {tokenizer.bos_id}, EOS: {tokenizer.eos_id}")
    
    # Create model config
    vocab_size = args.vocab_size if args.vocab_size else tokenizer.vocab_size
    config = ModelConfig(
        vocab_size=vocab_size,
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        use_probabilistic_encoder=False,  # Disable for evaluation simplicity
    )
    config.max_seq_len = args.max_len
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_checkpoint(args.checkpoint, config, device)
    
    # Setup logging directory if output_dir is specified
    logging_enabled = args.output_dir is not None
    if logging_enabled:
        from datetime import datetime
        
        # Create run name if not provided
        if args.run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            run_name = args.run_name
        
        # Create output directory
        output_dir = Path(args.output_dir) / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nLogging enabled:")
        print(f"  Output directory: {output_dir}")
        print(f"  Top-K tokens: {args.topk_tokens}")
        print(f"  Save mispredictions only: {args.save_mispred_only}")
        
        # Create summary file
        summary_path = output_dir / "summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("BLiMP Evaluation Summary\n")
            f.write("="*80 + "\n")
            f.write(f"Run name: {run_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mode: {args.mode}\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"SPM model: {args.spm_model}\n")
            f.write(f"Data path: {args.data_path}\n")
            f.write(f"Device: {device}\n")
            if device.type == 'cuda':
                f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"Vocab size: {tokenizer.vocab_size}\n")
            f.write(f"Batch size: {args.batch_size}\n")
            f.write(f"Max length: {args.max_len}\n")
            f.write(f"Score scope: {args.score_scope}\n")
            if args.score_scope == 'diff_window':
                f.write(f"Window: [{args.window_left} before, {args.window_right} after]\n")
            elif args.score_scope == 'prefix':
                f.write(f"Prefix length: {args.prefix_len}\n")
            f.write(f"Top-K tokens: {args.topk_tokens}\n")
            f.write(f"Save mispredictions only: {args.save_mispred_only}\n")
            f.write("="*80 + "\n\n")
    else:
        output_dir = None
        print("\nLogging disabled (no --output_dir specified)")
    
    # Detect if data_path is a file or directory
    data_path = Path(args.data_path)
    is_single_file = data_path.is_file() and data_path.suffix == '.jsonl'
    
    if is_single_file:
        # Single JSONL file mode: load and group by phenomenon
        print(f"\nLoading single JSONL file: {args.data_path}")
        phenomena_groups = load_single_jsonl_by_phenomenon(args.data_path)
        
        # Determine phenomena to evaluate
        if args.subsets:
            # Filter to specified phenomena
            requested_phenomena = [s.strip() for s in args.subsets.split(',')]
            phenomena_to_eval = {k: v for k, v in phenomena_groups.items() if k in requested_phenomena}
            
            # Warn about missing phenomena
            missing = set(requested_phenomena) - set(phenomena_groups.keys())
            if missing:
                warnings.warn(f"Requested phenomena not found in file: {', '.join(missing)}")
        else:
            # Evaluate all phenomena found in file
            phenomena_to_eval = phenomena_groups
        
        print(f"Found {len(phenomena_groups)} phenomena in file")
        print(f"Evaluating {len(phenomena_to_eval)} phenomena in {args.mode} mode...")
        print(f"Phenomena: {', '.join(sorted(phenomena_to_eval.keys()))}")
    else:
        # Directory mode: subset selection based on subset_mode and --subsets
        if args.subsets:
            # Explicit subset list always takes precedence
            subsets = [s.strip() for s in args.subsets.split(',')]
            print(f"\nUsing explicitly specified subsets: {len(subsets)} subsets")
        else:
            # No explicit subsets: use subset_mode to determine behavior
            if args.subset_mode == 'auto':
                # Auto-discover all .jsonl files in directory
                subsets = discover_subsets_from_dir(data_path)
                print(f"\n[subset_mode=auto] Auto-discovered {len(subsets)} .jsonl files in directory")
                if len(subsets) > 10:
                    print(f"  Preview (first 10): {', '.join(subsets[:10])}")
                    print(f"  ... and {len(subsets) - 10} more")
                else:
                    print(f"  Files: {', '.join(subsets)}")
           
        print(f"\nEvaluating {len(subsets)} subsets in {args.mode} mode...")
        print(f"Data path: {args.data_path}")
    
    print(f"Batch size: {args.batch_size}")
    print(f"Max length: {args.max_len}")
    print(f"Score scope: {args.score_scope}")
    if args.score_scope == 'diff_window':
        print(f"  Window: [{args.window_left} tokens before, {args.window_right} tokens after diff]")
    elif args.score_scope == 'prefix':
        print(f"  Prefix length: {args.prefix_len} tokens")
    
    # Evaluate each subset/phenomenon
    results = {}
    skipped_subsets = []
    skipped_reasons = {}  # Track why subsets were skipped
    
    # Choose collate function based on mode
    if args.mode == 'decoder':
        collate_fn = lambda batch: padding_collate_fn_decoder(
            batch, tokenizer.pad_id, args.max_len, args.left_pad
        )
        eval_fn = evaluate_decoder
    else:
        collate_fn = lambda batch: padding_collate_fn_encdec(
            batch, tokenizer.pad_id, args.max_len, args.left_pad
        )
        eval_fn = evaluate_encdec
    
    if is_single_file:
        # Single file mode: iterate over phenomena
        for phenomenon in tqdm(sorted(phenomena_to_eval.keys()), desc="Evaluating phenomena"):
            examples = phenomena_to_eval[phenomenon]
            
            try:
                # Create dataset from examples
                dataset = BlimpDatasetFromExamples(
                    examples=examples,
                    tokenizer=tokenizer,
                    mode=args.mode,
                    max_len=args.max_len,
                    score_scope=args.score_scope,
                    window_left=args.window_left,
                    window_right=args.window_right,
                    prefix_len=args.prefix_len,
                )
                
                if len(dataset) == 0:
                    warnings.warn(f"Empty dataset for phenomenon: {phenomenon}")
                    skipped_subsets.append(phenomenon)
                    skipped_reasons[phenomenon] = "empty_dataset"
                    continue
                
                # Create dataloader
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                )
                
                # Open logging files if enabled
                if logging_enabled:
                    safe_name = make_filename_safe(phenomenon)
                    txt_path = output_dir / f"{safe_name}.txt"
                    jsonl_path = output_dir / f"{safe_name}.jsonl"
                    txt_file = open(txt_path, 'w', encoding='utf-8')
                    jsonl_file = open(jsonl_path, 'w', encoding='utf-8')
                else:
                    txt_file = None
                    jsonl_file = None
                
                try:
                    # Evaluate
                    accuracy, correct, total, correct_flipped, total_flipped, margin_stats = eval_fn(
                        model, dataloader, device,
                        tokenizer=tokenizer if logging_enabled else None,
                        subset_name=phenomenon,
                        txt_file=txt_file,
                        jsonl_file=jsonl_file,
                        topk_tokens=args.topk_tokens,
                        save_mispred_only=args.save_mispred_only,
                        score_scope=args.score_scope,
                        dataset_examples=examples if logging_enabled else None,
                    )
                    results[phenomenon] = {
                        'accuracy': accuracy,
                        'correct': correct,
                        'total': total,
                        'margin_stats': margin_stats,
                    }
                finally:
                    # Close logging files
                    if txt_file is not None:
                        txt_file.close()
                    if jsonl_file is not None:
                        jsonl_file.close()
                
                # Print diff_window diagnostics (if applicable)
                if args.score_scope == "diff_window":
                    stats = dataset.mask_stats
                    total = stats['diff_window_calls']
                    avg_good = stats['scored_positions_good'] / total if total > 0 else 0
                    avg_bad = stats['scored_positions_bad'] / total if total > 0 else 0
                    print(f"  [mask] {phenomenon}: calls={total} "
                          f"used_window={stats['diff_window_used_window']} "
                          f"fallback={stats['diff_window_fallback_full']} "
                          f"avg_scored_good={avg_good:.1f} avg_scored_bad={avg_bad:.1f}")
                
                # Print flipped accuracy diagnostic for negation_mei_vs_bu_experiential
                if phenomenon == "negation_mei_vs_bu_experiential":
                    flipped_acc = correct_flipped / total_flipped if total_flipped > 0 else 0.0
                    print(f"  [diag] flipped_acc = {flipped_acc:.2%} ({correct_flipped}/{total_flipped})")
                
                # Report truncation warnings
                if dataset.truncation_count > 0:
                    warnings.warn(f"Truncated {dataset.truncation_count} examples in {phenomenon}")
            
            except Exception as e:
                warnings.warn(f"Error evaluating {phenomenon}: {e}")
                skipped_subsets.append(phenomenon)
                skipped_reasons[phenomenon] = f"exception: {str(e)[:50]}"
                continue
    else:
        # Directory mode: iterate over subsets
        for subset in tqdm(subsets, desc="Evaluating subsets"):
            # Build path to JSONL file
            jsonl_path = os.path.join(args.data_path, f"{subset}.jsonl")
            
            # Check if file exists
            if not os.path.exists(jsonl_path):
                # Only warn about missing files if in standard mode or explicit subsets
                # In auto mode, we discovered these files, so this shouldn't happen
                if args.subset_mode == 'standard' or args.subsets:
                    warnings.warn(f"Subset file not found: {jsonl_path}")
                skipped_subsets.append(subset)
                skipped_reasons[subset] = "file_not_found"
                continue
            
            # Load dataset
            try:
                dataset = BlimpDataset(
                    jsonl_path=jsonl_path,
                    tokenizer=tokenizer,
                    mode=args.mode,
                    max_len=args.max_len,
                    score_scope=args.score_scope,
                    window_left=args.window_left,
                    window_right=args.window_right,
                    prefix_len=args.prefix_len,
                )
                
                if len(dataset) == 0:
                    warnings.warn(f"Empty dataset: {subset}")
                    skipped_subsets.append(subset)
                    skipped_reasons[subset] = "empty_dataset"
                    continue
                
                # Create dataloader
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                )
                
                # Open logging files if enabled
                if logging_enabled:
                    safe_name = make_filename_safe(subset)
                    txt_path = output_dir / f"{safe_name}.txt"
                    jsonl_path_log = output_dir / f"{safe_name}.jsonl"
                    txt_file = open(txt_path, 'w', encoding='utf-8')
                    jsonl_file = open(jsonl_path_log, 'w', encoding='utf-8')
                else:
                    txt_file = None
                    jsonl_file = None
                
                try:
                    # Evaluate
                    accuracy, correct, total, correct_flipped, total_flipped, margin_stats = eval_fn(
                        model, dataloader, device,
                        tokenizer=tokenizer if logging_enabled else None,
                        subset_name=subset,
                        txt_file=txt_file,
                        jsonl_file=jsonl_file,
                        topk_tokens=args.topk_tokens,
                        save_mispred_only=args.save_mispred_only,
                        score_scope=args.score_scope,
                        dataset_examples=dataset.examples if logging_enabled else None,
                    )
                    results[subset] = {
                        'accuracy': accuracy,
                        'correct': correct,
                        'total': total,
                        'margin_stats': margin_stats,
                    }
                finally:
                    # Close logging files
                    if txt_file is not None:
                        txt_file.close()
                    if jsonl_file is not None:
                        jsonl_file.close()
                
                # Print diff_window diagnostics (if applicable)
                if args.score_scope == "diff_window":
                    stats = dataset.mask_stats
                    total = stats['diff_window_calls']
                    avg_good = stats['scored_positions_good'] / total if total > 0 else 0
                    avg_bad = stats['scored_positions_bad'] / total if total > 0 else 0
                    print(f"  [mask] {subset}: calls={total} "
                          f"used_window={stats['diff_window_used_window']} "
                          f"fallback={stats['diff_window_fallback_full']} "
                          f"avg_scored_good={avg_good:.1f} avg_scored_bad={avg_bad:.1f}")
                
                # Report truncation warnings
                if dataset.truncation_count > 0:
                    warnings.warn(f"Truncated {dataset.truncation_count} examples in {subset}")
            
            except Exception as e:
                warnings.warn(f"Error evaluating {subset}: {e}")
                skipped_subsets.append(subset)
                skipped_reasons[subset] = f"exception: {str(e)[:50]}"
                continue
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    label = "Phenomenon" if is_single_file else "Subset"
    
    if results:
        print(f"\n{label:<50} {'Accuracy':<12} {'Correct/Total'}")
        print("-"*80)
        
        for name, result in sorted(results.items()):
            acc = result['accuracy']
            correct = result['correct']
            total = result['total']
            print(f"{name:<50} {acc:>7.2%}      {correct:>4}/{total:<4}")
        
        # Compute macro-average
        macro_avg = sum(r['accuracy'] for r in results.values()) / len(results)
        print("-"*80)
        print(f"{'MACRO-AVERAGE':<50} {macro_avg:>7.2%}")
        print("="*80)
        
        # Compute and print margin statistics
        print("\n" + "="*80)
        print("CONFIDENCE (MARGIN) SUMMARY")
        print("="*80)
        
        # Aggregate all margin stats across phenomena
        all_deltas = []
        all_margins = []
        all_correct = []
        for result in results.values():
            ms = result['margin_stats']
            all_deltas.extend(ms['deltas'])
            all_margins.extend(ms['margins'])
            all_correct.extend(ms['correct'])
        
        macro_margin_stats = {
            'deltas': all_deltas,
            'margins': all_margins,
            'correct': all_correct,
        }
        macro_summary = compute_margin_summary(macro_margin_stats)
        
        print(f"\nMACRO (all samples, n={len(all_deltas)}):")
        print(f"  Mean |Δ| (margin):        {macro_summary['mean_margin']:.4f}")
        print(f"  Median |Δ|:               {macro_summary['median_margin']:.4f}")
        print(f"  Mean Δ (all):             {macro_summary['mean_delta']:.4f}")
        if not np.isnan(macro_summary['mean_delta_correct']):
            print(f"  Mean Δ (correct):         {macro_summary['mean_delta_correct']:.4f}")
        if not np.isnan(macro_summary['mean_delta_incorrect']):
            print(f"  Mean Δ (incorrect):       {macro_summary['mean_delta_incorrect']:.4f}")
        print(f"  % samples with |Δ|<0.1:   {macro_summary['pct_low_margin']:.1f}%")
        if not np.isnan(macro_summary['acc_margin_ge_0_1']):
            print(f"  Acc(|Δ|>=0.1):            {macro_summary['acc_margin_ge_0_1']:.2%}")
        if not np.isnan(macro_summary['acc_margin_ge_0_5']):
            print(f"  Acc(|Δ|>=0.5):            {macro_summary['acc_margin_ge_0_5']:.2%}")
        if not np.isnan(macro_summary['acc_margin_ge_1_0']):
            print(f"  Acc(|Δ|>=1.0):            {macro_summary['acc_margin_ge_1_0']:.2%}")
        print("="*80)
        
        # Write final summary to summary.txt if logging enabled
        if logging_enabled:
            summary_path = output_dir / "summary.txt"
            with open(summary_path, 'a', encoding='utf-8') as f:
                f.write("\n" + "="*80 + "\n")
                f.write("RESULTS\n")
                f.write("="*80 + "\n\n")
                f.write(f"{label:<50} {'Accuracy':<12} {'Correct/Total'}\n")
                f.write("-"*80 + "\n")
                
                for name, result in sorted(results.items()):
                    acc = result['accuracy']
                    correct = result['correct']
                    total = result['total']
                    f.write(f"{name:<50} {acc:>7.2%}      {correct:>4}/{total:<4}\n")
                
                f.write("-"*80 + "\n")
                f.write(f"{'MACRO-AVERAGE':<50} {macro_avg:>7.2%}\n")
                f.write("="*80 + "\n")
                
                # Write margin statistics
                f.write("\n" + "="*80 + "\n")
                f.write("CONFIDENCE (MARGIN) SUMMARY\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"MACRO (all samples, n={len(all_deltas)}):\n")
                f.write(f"  Mean |Δ| (margin):        {macro_summary['mean_margin']:.4f}\n")
                f.write(f"  Median |Δ|:               {macro_summary['median_margin']:.4f}\n")
                f.write(f"  Mean Δ (all):             {macro_summary['mean_delta']:.4f}\n")
                if not np.isnan(macro_summary['mean_delta_correct']):
                    f.write(f"  Mean Δ (correct):         {macro_summary['mean_delta_correct']:.4f}\n")
                if not np.isnan(macro_summary['mean_delta_incorrect']):
                    f.write(f"  Mean Δ (incorrect):       {macro_summary['mean_delta_incorrect']:.4f}\n")
                f.write(f"  % samples with |Δ|<0.1:   {macro_summary['pct_low_margin']:.1f}%\n")
                if not np.isnan(macro_summary['acc_margin_ge_0_1']):
                    f.write(f"  Acc(|Δ|>=0.1):            {macro_summary['acc_margin_ge_0_1']:.2%}\n")
                if not np.isnan(macro_summary['acc_margin_ge_0_5']):
                    f.write(f"  Acc(|Δ|>=0.5):            {macro_summary['acc_margin_ge_0_5']:.2%}\n")
                if not np.isnan(macro_summary['acc_margin_ge_1_0']):
                    f.write(f"  Acc(|Δ|>=1.0):            {macro_summary['acc_margin_ge_1_0']:.2%}\n")
                
                # Write per-phenomenon margin stats
                f.write("\n" + "-"*80 + "\n")
                f.write("Per-Phenomenon Margin Statistics:\n")
                f.write("-"*80 + "\n\n")
                
                for name in sorted(results.keys()):
                    result = results[name]
                    phenom_summary = compute_margin_summary(result['margin_stats'])
                    n_samples = result['total']
                    
                    f.write(f"{name} (n={n_samples}):\n")
                    f.write(f"  Mean |Δ|: {phenom_summary['mean_margin']:.4f}, ")
                    f.write(f"Median |Δ|: {phenom_summary['median_margin']:.4f}, ")
                    f.write(f"Mean Δ: {phenom_summary['mean_delta']:.4f}\n")
                    if not np.isnan(phenom_summary['mean_delta_correct']):
                        f.write(f"  Mean Δ (correct): {phenom_summary['mean_delta_correct']:.4f}, ")
                    if not np.isnan(phenom_summary['mean_delta_incorrect']):
                        f.write(f"Mean Δ (incorrect): {phenom_summary['mean_delta_incorrect']:.4f}\n")
                    else:
                        f.write("\n")
                    f.write(f"  Low margin %: {phenom_summary['pct_low_margin']:.1f}%")
                    if not np.isnan(phenom_summary['acc_margin_ge_0_5']):
                        f.write(f", Acc(|Δ|>=0.5): {phenom_summary['acc_margin_ge_0_5']:.2%}")
                    f.write("\n\n")
                
                f.write("="*80 + "\n")
                
                if skipped_subsets:
                    f.write(f"\nSkipped {label.lower()}s ({len(skipped_subsets)}): {', '.join(skipped_subsets)}\n")
                    if skipped_reasons:
                        f.write("\nSkip reasons:\n")
                        for name in sorted(skipped_reasons.keys()):
                            f.write(f"  {name}: {skipped_reasons[name]}\n")
            
            print(f"\nSummary written to: {summary_path}")
    else:
        print(f"No results to report (all {label.lower()}s failed or were skipped).")
    
    if skipped_subsets:
        print(f"\nSkipped {label.lower()}s ({len(skipped_subsets)}): {', '.join(skipped_subsets)}")
        if skipped_reasons:
            print("\nSkip reasons:")
            for name in sorted(skipped_reasons.keys()):
                print(f"  {name}: {skipped_reasons[name]}")
    
    print(f"\n[OK] Evaluation complete")


if __name__ == '__main__':
    main()
