"""
Configuration dataclasses for the pinyin-initials encoder-decoder model.
"""
from dataclasses import dataclass, field
from typing import Optional, Literal

local_vocab_size = 3900  # Small vocab size for smoke test; increase to 2000-8000 for production

@dataclass
class TokenizerConfig:
    """SentencePiece tokenizer configuration."""
    vocab_size: int = local_vocab_size  # Set to 64 for small test corpus (160 lines). Increase to 2000-8000 for production with larger corpus.
    model_type: str = "unigram"  # unigram, bpe, char, word
    character_coverage: float = 1.0
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3
    # Additional special tokens if needed
    user_defined_symbols: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Encoder-decoder model architecture configuration."""
    # Vocabulary
    vocab_size: int = local_vocab_size  # Must match TokenizerConfig.vocab_size. Increase to 2000-8000 for production.
    pad_id: int = 0
    bos_id: int = 2
    eos_id: int = 3
    
    # Model dimensions
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    
    # Encoder
    n_encoder_layers: int = 6
    
    # Decoder
    n_decoder_layers: int = 6
    
    # Probabilistic encoder
    use_probabilistic_encoder: bool = True
    n_latent_classes: Optional[int] = None  # If None, use vocab_size
    prior_alpha: float = 0.5  # Weight for n-gram prior
    posterior_beta: float = 0.3  # Weight for comprehension fusion
    use_gating: bool = True  # Use gating in decoder cross-attention
    
    # KL regularization
    kl_weight: float = 0.01  # Weight for KL(posterior || prior)
    
    # Max sequence length
    max_seq_len: int = 256
    
    def __post_init__(self):
        if self.n_latent_classes is None:
            self.n_latent_classes = self.vocab_size


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings."""
    # Data
    csv_path: str = "corpus.csv"
    text_column: str = "text"
    train_ratio: float = 0.9
    random_seed: int = 42
    
    # Training mode
    mode: Literal["causal", "taskA_prefix_suffix", "mixed"] = "mixed"
    taskA_prob: float = 0.3  # Probability of taskA in mixed mode
    
    # Validation mode (separate from training mode for stable metrics)
    val_mode: Literal["causal", "taskA_prefix_suffix", "mixed"] = "causal"
    
    # Task mixing control (for preventing collapse)
    causal_to_taskA_ratio: int = 6  # 6 causal batches per 1 taskA batch (when using alternating loaders)
    taskA_weight: float = 0.2  # Weight for taskA loss in combined loss
    
    # Causal mode configuration (encoder-decoder prompt→continuation)
    causal_mode_type: Literal["encdec"] = "encdec"  # Encoder-decoder generative objective
    min_prompt_len: int = 16  # Minimum tokens in encoder prompt (source)
    min_target_len: int = 16  # Minimum tokens in decoder target (continuation)
    prompt_sampling: Literal["random", "fixed"] = "random"  # How to split prompt/target
    fixed_prompt_len: Optional[int] = None  # If set, use this exact prompt length (overrides fixed_prompt_ratio)
    fixed_prompt_ratio: float = 0.5  # If prompt_sampling="fixed", use this ratio for prompt length
    
    # EOS collapse prevention
    min_new_tokens_train: int = 24  # Enforce targets >= this length (excluding BOS) to prevent short continuation bias
    target_len_strategy: Literal["uniform", "favor_long"] = "favor_long"  # Sampling strategy for target length
    favor_long_gamma: float = 2.0  # Exponent for favor_long strategy: p(t) ∝ (t - min_target + 1)^gamma
    eos_in_labels: bool = True  # Keep EOS supervised in labels
    eos_loss_weight: float = 0.05  # Down-weight EOS in loss (0.05-0.2 works well; 1.0 = no down-weighting)
    max_target_len: Optional[int] = None  # Optional cap for target length
    
    # Validation-specific split strategy (for deterministic validation)
    validation_prompt_sampling: Literal["random", "fixed"] = "fixed"  # Recommended: "fixed" for determinism
    validation_fixed_prompt_len: Optional[int] = None  # If set, use this exact prompt length for validation
    validation_fixed_prompt_ratio: float = 0.5  # If validation_prompt_sampling="fixed", use this ratio
    
    # Debug mode
    debug: bool = False  # Enable debug prints and safety checks
    debug_sanity: bool = True  # Enable sanity checks for causal LM (CRITICAL for validation)
    
    # Noising for causal mode (to prevent leakage)
    noise_type: Literal["none", "mask", "drop"] = "mask"
    noise_ratio: float = 0.15  # Ratio of tokens to noise
    
    # Optimization
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 10
    warmup_steps: int = 1500
    min_lr_ratio: float = 0.1  # Cosine LR floor: final lr = base_lr * min_lr_ratio
    gradient_clip: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    
    # Two-phase training
    phase1_enabled: bool = True  # Enable phase 1 (TaskA warm-start)
    phase1_epochs: int = 2  # Max epochs for TaskA warm-start
    phase1_early_stop_patience: int = 1  # Short patience for phase 1
    phase1_early_stop_min_delta: float = 0.01
    phase1_lr_multiplier: float = 1.0  # LR multiplier for phase 1 (1.0 = same as base)
    
    phase2_epochs: int = 10  # Max epochs for causal fine-tune
    phase2_early_stop_patience: int = 3  # Patience for phase 2
    phase2_early_stop_min_delta: float = 0.01
    phase2_lr_multiplier: float = 0.5  # LR multiplier for phase 2 (safer to start with lower LR)
    
    # Checkpointing
    save_best_on_phase2_only: bool = True  # Only update best_model.pt during phase 2
    best_metric: str = "val_loss_causal"  # Metric for best checkpoint selection
    
    # Early stopping (legacy - kept for backward compatibility)
    early_stop_patience: int = 3  # Stop if no improvement for N epochs
    early_stop_min_delta: float = 0.01  # Minimum change to count as improvement
    early_stop_metric: str = "val_loss_causal"  # Metric to monitor for early stopping
    
    # Checkpointing
    save_interval: int = 1  # Save every N epochs
    keep_last_n: int = 3  # Keep last N checkpoints
    
    # Artifact paths (set by train.py, used for checkpoint copying)
    spm_model_path: Optional[str] = None  # Path to SentencePiece model file
    ngram_prior_path: Optional[str] = None  # Path to n-gram prior file
    
    # Logging
    log_interval: int = 50  # Log every N steps
    
    # Device
    device: str = "cuda"  # Will be set automatically


@dataclass
class GenerationConfig:
    """Generation-specific configuration."""
    # Debug mode
    debug: bool = False  # Enable verbose generation diagnostics
    
    # Prompt length warnings
    generation_min_prompt_len: int = 8  # Warn if prompt is shorter (model trained on longer prompts)


@dataclass
class PreprocessConfig:
    """Preprocessing configuration."""
    # Digraph handling
    use_digraph_repetition: bool = True  # Enable zhh/chh/shh for repeated digraphs
    
    # Output
    initials_corpus_path: str = "initials_corpus.txt"
    
    # Validation
    max_hanzi_per_line: int = 1000  # Skip very long lines


if __name__ == "__main__":
    # Smoke test
    tok_cfg = TokenizerConfig()
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    prep_cfg = PreprocessConfig()
    
    print("[OK] Config classes initialized successfully")
    print(f"  Tokenizer vocab: {tok_cfg.vocab_size}")
    print(f"  Model d_model: {model_cfg.d_model}, layers: {model_cfg.n_encoder_layers}/{model_cfg.n_decoder_layers}")
    print(f"  Training mode: {train_cfg.mode}, batch: {train_cfg.batch_size}")
