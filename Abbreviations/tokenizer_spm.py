"""
SentencePiece tokenizer training and utilities.

CRITICAL: Train ONLY on initials corpus (never on Hanzi).
Validate that vocab contains NO CJK characters.
"""
import re
from pathlib import Path
from typing import List, Optional

import sentencepiece as spm

from config import TokenizerConfig


def is_cjk_char(char: str) -> bool:
    """Check if character is in CJK unicode ranges."""
    if not char:
        return False
    code = ord(char)
    # CJK Unified Ideographs: 4E00-9FFF
    # CJK Extension A: 3400-4DBF
    # CJK Extension B-F: 20000-2EBEF
    # CJK Compatibility: F900-FAFF
    return (0x4E00 <= code <= 0x9FFF or
            0x3400 <= code <= 0x4DBF or
            0x20000 <= code <= 0x2EBEF or
            0xF900 <= code <= 0xFAFF)


def validate_vocab_no_cjk(model_path: str) -> None:
    """
    Validate that SentencePiece vocab contains NO CJK characters.
    
    Raises:
        AssertionError if CJK characters found in vocab
    """
    sp = spm.SentencePieceProcessor()
    sp.Load(model_path)
    
    vocab_size = sp.GetPieceSize()
    cjk_pieces = []
    
    for i in range(vocab_size):
        piece = sp.IdToPiece(i)
        # Check each character in the piece
        for char in piece:
            if is_cjk_char(char):
                cjk_pieces.append((i, piece, char))
    
    if cjk_pieces:
        error_msg = "❌ CRITICAL: CJK characters found in SentencePiece vocab!\n"
        error_msg += "This means the model was trained on Hanzi instead of initials.\n"
        error_msg += "CJK pieces found:\n"
        for idx, piece, char in cjk_pieces[:10]:  # Show first 10
            error_msg += f"  ID {idx}: '{piece}' contains '{char}' (U+{ord(char):04X})\n"
        raise AssertionError(error_msg)
    
    print(f"[OK] SentencePiece vocab validated: NO CJK characters found ({vocab_size} pieces)")


def train_sentencepiece(
    corpus_path: str,
    model_prefix: str,
    config: TokenizerConfig
) -> None:
    """
    Train SentencePiece model on initials corpus.
    
    Args:
        corpus_path: Path to initials text corpus (NOT Hanzi!)
        model_prefix: Output model prefix (will create .model and .vocab files)
        config: Tokenizer configuration
    """
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    
    # Validate corpus is initials-only
    print(f"Validating corpus contains only initials...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            for char in line:
                if is_cjk_char(char):
                    raise AssertionError(
                        f"CJK character '{char}' found in corpus at line {i}!\n"
                        f"Corpus must contain ONLY initials (a-z and spaces)."
                    )
            if i >= 1000:  # Check first 1000 lines
                break
    print(f"[OK] Corpus validation passed")
    
    # Prepare arguments
    user_defined_symbols_str = ",".join(config.user_defined_symbols) if config.user_defined_symbols else ""
    
    args = [
        f"--input={corpus_path}",
        f"--model_prefix={model_prefix}",
        f"--vocab_size={config.vocab_size}",
        f"--model_type={config.model_type}",
        f"--character_coverage={config.character_coverage}",
        f"--pad_id={config.pad_id}",
        f"--unk_id={config.unk_id}",
        f"--bos_id={config.bos_id}",
        f"--eos_id={config.eos_id}",
        "--normalization_rule_name=identity",  # Don't normalize (already processed)
        "--minloglevel=2",  # Suppress verbose output to avoid encoding issues
    ]
    
    if user_defined_symbols_str:
        args.append(f"--user_defined_symbols={user_defined_symbols_str}")
    
    print(f"Training SentencePiece model...")
    print(f"  Input: {corpus_path}")
    print(f"  Output: {model_prefix}.model")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Model type: {config.model_type}")
    
    spm.SentencePieceTrainer.Train(" ".join(args))
    
    # Validate output
    model_path = f"{model_prefix}.model"
    validate_vocab_no_cjk(model_path)
    
    print(f"[OK] SentencePiece model trained successfully")


class InitialsTokenizer:
    """
    Wrapper for SentencePiece tokenizer with initials-specific utilities.
    """
    
    def __init__(self, model_path: str):
        """
        Load SentencePiece model.
        
        Args:
            model_path: Path to .model file
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        
        # Cache special token IDs
        self.pad_id = self.sp.pad_id()
        self.unk_id = self.sp.unk_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.vocab_size = self.sp.GetPieceSize()
        
        print(f"Loaded SentencePiece model: {model_path}")
        print(f"  Vocab size: {self.vocab_size}")
        print(f"  Special IDs: PAD={self.pad_id}, UNK={self.unk_id}, BOS={self.bos_id}, EOS={self.eos_id}")
    
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text (should be initials string)
            add_bos: Prepend BOS token
            add_eos: Append EOS token
        
        Returns:
            List of token IDs
        """
        ids = self.sp.EncodeAsIds(text)
        
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        
        return ids
    
    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
            skip_special_tokens: Remove special tokens from output
        
        Returns:
            Decoded text
        """
        if skip_special_tokens:
            # Filter out special tokens
            special_ids = {self.pad_id, self.bos_id, self.eos_id}
            ids = [i for i in ids if i not in special_ids]
        
        return self.sp.DecodeIds(ids)
    
    def encode_batch(self, texts: List[str], add_bos: bool = False, add_eos: bool = False) -> List[List[int]]:
        """Encode batch of texts."""
        return [self.encode(text, add_bos, add_eos) for text in texts]
    
    def decode_batch(self, ids_batch: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        """Decode batch of token ID sequences."""
        return [self.decode(ids, skip_special_tokens) for ids in ids_batch]
    
    def get_vocab(self) -> List[str]:
        """Get all vocabulary pieces."""
        return [self.sp.IdToPiece(i) for i in range(self.vocab_size)]


if __name__ == "__main__":
    print("=== Tokenizer Smoke Test ===\n")
    
    # Create a tiny test corpus
    test_corpus = Path("test_initials_corpus.txt")
    with open(test_corpus, "w", encoding="utf-8") as f:
        f.write("zh g\n")  # 中国
        f.write("sh h\n")  # 上海
        f.write("ch h\n")  # 长河
        f.write("sh h zh g ch h\n")
        f.write("w a zh zh g j sh h\n")
    
    # Train small model
    test_config = TokenizerConfig(vocab_size=13, model_type="unigram")
    model_prefix = "test_spm"
    
    print("Training test SentencePiece model...")
    train_sentencepiece(str(test_corpus), model_prefix, test_config)
    
    # Load and test
    print("\nLoading tokenizer...")
    tokenizer = InitialsTokenizer(f"{model_prefix}.model")
    
    # Test encoding/decoding
    print("\nTest encode/decode:")
    test_text = "zh g sh h"
    encoded = tokenizer.encode(test_text, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(encoded)
    
    print(f"  Original: {test_text}")
    print(f"  Encoded: {encoded}")
    print(f"  Decoded: {decoded}")
    
    # Check vocab
    print("\nSample vocab pieces:")
    vocab = tokenizer.get_vocab()
    for i in range(min(20, len(vocab))):
        piece = vocab[i]
        # Replace non-ASCII chars (e.g., '\u2581') with '<ws>'
        safe_piece = ''.join(c if 32 <= ord(c) < 127 else '<ws>' if c == '\u2581' or c == '▁' else f'<U+{ord(c):04X}>' for c in piece)
        print(f"  {i}: '{safe_piece}'")
    
    # Cleanup
    import os
    for ext in [".model", ".vocab"]:
        try:
            os.remove(f"{model_prefix}{ext}")
        except:
            pass
    try:
        os.remove(test_corpus)
    except:
        pass
    
    print("\n[OK] Tokenizer module smoke test passed")
