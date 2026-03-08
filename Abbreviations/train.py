"""
Main training script with CLI interface.

End-to-end pipeline:
1. Preprocess CSV (Hanzi → initials)
2. Train SentencePiece on initials corpus
3. Validate vocab contains no CJK
4. Compute n-gram priors on TRAIN split
5. Train encoder-decoder model
6. Save checkpoints
"""
import argparse
from pathlib import Path
import sys

import torch

from config import TokenizerConfig, ModelConfig, TrainingConfig, PreprocessConfig
from preprocess import preprocess_csv, validate_initials_corpus, hanzi_to_initials
from tokenizer_spm import train_sentencepiece, InitialsTokenizer, validate_vocab_no_cjk
from ngrams import NgramPrior
from dataset import create_dataloaders, InitialsDataset, collate_fn
from torch.utils.data import DataLoader
from model import EncoderDecoderModel
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train pinyin-initials encoder-decoder model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data
    parser.add_argument("--csv", type=str, default="corpus.csv", help="Input CSV with Hanzi text")
    parser.add_argument("--text_column", type=str, default="text", help="Column name for text")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    
    # Preprocessing
    parser.add_argument("--use_digraph_repetition", action="store_true", default=True,
                        help="Use zhh/chh/shh for repeated digraphs")
    parser.add_argument("--no_digraph_repetition", action="store_false", dest="use_digraph_repetition")
    
    # Tokenizer
    parser.add_argument("--spm_model", type=str, default=None,
                        help="Existing SentencePiece model (skip training if provided)")
    parser.add_argument("--spm_vocab", type=int, default=3900, help="SentencePiece vocab size (64 for small corpus, 2000-8000 for production)")
    parser.add_argument("--spm_type", type=str, default="unigram", 
                        choices=["unigram", "bpe", "char", "word"], help="SentencePiece model type")
    
    # Model architecture
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--n_encoder_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--n_decoder_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    
    # Probabilistic encoder
    parser.add_argument("--no_prob_encoder", action="store_false", dest="use_probabilistic_encoder",
                        help="Disable probabilistic encoder")
    parser.add_argument("--prior_alpha", type=float, default=0.5, help="Weight for n-gram prior")
    parser.add_argument("--posterior_beta", type=float, default=0.3, help="Weight for comprehension fusion")
    parser.add_argument("--kl_weight", type=float, default=0.01, help="KL regularization weight")
    parser.add_argument("--no_gating", action="store_false", dest="use_gating",
                        help="Disable gating in decoder")
    
    # Training
    parser.add_argument("--mode", type=str, default="mixed", 
                        choices=["causal", "taskA_prefix_suffix", "mixed"],
                        help="Training mode")
    parser.add_argument("--val_mode", type=str, default="causal",
                        choices=["causal", "taskA_prefix_suffix", "mixed"],
                        help="Validation mode (separate from training for stable metrics)")
    parser.add_argument("--causal_to_taskA_ratio", type=int, default=6,
                        help="Ratio of causal to taskA batches for alternating training (e.g., 6 means 6:1)")
    parser.add_argument("--taskA_weight", type=float, default=0.2,
                        help="Weight for taskA loss in combined training loss")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with extra prints and safety checks")
    parser.add_argument("--taskA_prob", type=float, default=0.3,
                        help="Probability of taskA in mixed mode")
    parser.add_argument("--noise_type", type=str, default="mask",
                        choices=["none", "mask", "drop"],
                        help="Noising type for causal mode")
    parser.add_argument("--noise_ratio", type=float, default=0.15,
                        help="Ratio of tokens to noise")
    
    # Causal mode configuration
    parser.add_argument("--min_prompt_len", type=int, default=16,
                        help="Minimum prompt tokens for causal mode")
    parser.add_argument("--min_target_len", type=int, default=16,
                        help="Minimum target tokens for causal mode")
    parser.add_argument("--prompt_sampling", type=str, default="random",
                        choices=["random", "fixed"],
                        help="Prompt/target split strategy for causal mode")
    parser.add_argument("--fixed_prompt_len", type=int, default=None,
                        help="Exact prompt length for fixed sampling (overrides ratio)")
    parser.add_argument("--fixed_prompt_ratio", type=float, default=0.5,
                        help="Prompt ratio for fixed sampling (0.5 = 50%% prompt)")
    parser.add_argument("--validation_prompt_sampling", type=str, default="fixed",
                        choices=["random", "fixed"],
                        help="Validation prompt sampling strategy (recommended: fixed for determinism)")
    parser.add_argument("--validation_fixed_prompt_len", type=int, default=None,
                        help="Exact prompt length for validation (overrides ratio)")
    parser.add_argument("--validation_fixed_prompt_ratio", type=float, default=0.5,
                        help="Validation prompt ratio for fixed sampling")
    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Warmup steps")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1, help="Minimum LR ratio for cosine schedule")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Train/val split ratio")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    
    # Two-phase training
    parser.add_argument("--phase1_enabled", action="store_true", default=True,
                        help="Enable phase 1 (TaskA warm-start)")
    parser.add_argument("--no_phase1", action="store_false", dest="phase1_enabled",
                        help="Disable phase 1")
    parser.add_argument("--phase1_epochs", type=int, default=2,
                        help="Max epochs for phase 1 (TaskA warm-start)")
    parser.add_argument("--phase1_early_stop_patience", type=int, default=1,
                        help="Early stop patience for phase 1")
    parser.add_argument("--phase1_lr_multiplier", type=float, default=1.0,
                        help="LR multiplier for phase 1 (1.0 = same as base LR)")
    parser.add_argument("--phase2_epochs", type=int, default=10,
                        help="Max epochs for phase 2 (causal fine-tune)")
    parser.add_argument("--phase2_early_stop_patience", type=int, default=3,
                        help="Early stop patience for phase 2")
    parser.add_argument("--phase2_lr_multiplier", type=float, default=0.5,
                        help="LR multiplier for phase 2 (0.5 = half base LR)")
    
    # System
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--no_amp", action="store_false", dest="use_amp",
                        help="Disable mixed precision training")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers")
    
    # Checkpointing
    parser.add_argument("--save_interval", type=int, default=1, help="Save every N epochs")
    parser.add_argument("--keep_last_n", type=int, default=3, help="Keep last N checkpoints")
    parser.add_argument("--log_interval", type=int, default=50, help="Log every N steps")
    
    # N-gram
    parser.add_argument("--ngram_k", type=float, default=1.0, help="N-gram smoothing parameter")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠ CUDA not available, using CPU")
        args.device = "cpu"
    
    print("="*70)
    print("Pinyin-Initials Encoder-Decoder Model Training")
    print("="*70)
    print(f"Device: {args.device}")
    print(f"Output directory: {args.out_dir}")
    print("="*70 + "\n")
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create training config early (needed for dataset initialization)
    # This will be used throughout the pipeline
    config = TrainingConfig(
        csv_path=args.csv,
        text_column=args.text_column,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        mode=args.mode,
        taskA_prob=args.taskA_prob,
        val_mode=args.val_mode,
        causal_to_taskA_ratio=args.causal_to_taskA_ratio,
        taskA_weight=args.taskA_weight,
        debug=args.debug,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=getattr(args, 'min_lr_ratio', 0.1),
        gradient_clip=args.gradient_clip,
        use_amp=args.use_amp,
        save_interval=args.save_interval,
        keep_last_n=args.keep_last_n,
        log_interval=args.log_interval,
        device=args.device,
        min_prompt_len=args.min_prompt_len,
        min_target_len=args.min_target_len,
        prompt_sampling=args.prompt_sampling,
        fixed_prompt_len=args.fixed_prompt_len,
        fixed_prompt_ratio=args.fixed_prompt_ratio,
        validation_prompt_sampling=args.validation_prompt_sampling,
        validation_fixed_prompt_len=args.validation_fixed_prompt_len,
        validation_fixed_prompt_ratio=args.validation_fixed_prompt_ratio,
        phase1_enabled=getattr(args, 'phase1_enabled', True),
        phase1_epochs=getattr(args, 'phase1_epochs', 2),
        phase1_early_stop_patience=getattr(args, 'phase1_early_stop_patience', 1),
        phase1_lr_multiplier=getattr(args, 'phase1_lr_multiplier', 1.0),
        phase2_epochs=getattr(args, 'phase2_epochs', 10),
        phase2_early_stop_patience=getattr(args, 'phase2_early_stop_patience', 3),
        phase2_lr_multiplier=getattr(args, 'phase2_lr_multiplier', 0.5),
    )
    
    # Step 1: Preprocess CSV to initials corpus
    initials_corpus_path = out_dir / "initials_corpus.txt"
    
    if not initials_corpus_path.exists():
        print("Step 1: Preprocessing Hanzi → Initials")
        print("-" * 70)
        
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f"❌ ERROR: CSV file not found: {csv_path}")
            print("\nCreating a demo CSV for testing...")
            
            # Create demo CSV
            import csv
            demo_csv = out_dir / "demo_corpus.csv"
            with open(demo_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["text"])
                writer.writeheader()
                # Some demo Hanzi sentences
                demo_texts = [
                    "中国是一个伟大的国家",
                    "我爱学习中文",
                    "今天天气很好",
                    "北京是中国的首都",
                    "上海是一个现代化的城市",
                    "长江是中国最长的河流",
                    "我喜欢吃中国菜",
                    "汉语是世界上使用人数最多的语言",
                ] * 10  # Repeat for more data
                
                for text in demo_texts:
                    writer.writerow({"text": text})
            
            print(f"[OK] Demo CSV created: {demo_csv}")
            csv_path = demo_csv
            args.csv = str(demo_csv)
        
        lines_processed, lines_skipped = preprocess_csv(
            csv_path=str(csv_path),
            output_path=str(initials_corpus_path),
            text_column=args.text_column,
            use_digraph_repetition=args.use_digraph_repetition,
        )
        
        print(f"[OK] Preprocessing complete:")
        print(f"  Processed: {lines_processed} lines")
        print(f"  Skipped: {lines_skipped} lines")
        print(f"  Output: {initials_corpus_path}")
        
        # Validate initials corpus
        validate_initials_corpus(str(initials_corpus_path))
        print()
    else:
        print(f"Step 1: Using existing initials corpus: {initials_corpus_path}\n")
    
    # Step 2: Train SentencePiece tokenizer
    spm_model_path = out_dir / "spm_model.model"
    
    if args.spm_model:
        spm_model_path = Path(args.spm_model)
        print(f"Step 2: Using existing SentencePiece model: {spm_model_path}")
    elif not spm_model_path.exists():
        print("Step 2: Training SentencePiece Tokenizer")
        print("-" * 70)
        
        tokenizer_config = TokenizerConfig(
            vocab_size=args.spm_vocab,
            model_type=args.spm_type,
        )
        
        train_sentencepiece(
            corpus_path=str(initials_corpus_path),
            model_prefix=str(out_dir / "spm_model"),
            config=tokenizer_config,
        )
        print()
    else:
        print(f"Step 2: Using existing SentencePiece model: {spm_model_path}\n")
    
    # Validate SentencePiece vocab (CRITICAL)
    print("Step 2b: Validating SentencePiece vocabulary")
    print("-" * 70)
    validate_vocab_no_cjk(str(spm_model_path))
    print()
    
    # Load tokenizer
    tokenizer = InitialsTokenizer(str(spm_model_path))
    
    # Store artifact path in config for checkpoint saving
    config.spm_model_path = str(spm_model_path)
    
    # Step 3: Compute n-gram priors
    print("Step 3: Computing N-gram Priors")
    print("-" * 70)
    
    ngram_path = out_dir / "ngram_prior.npz"
    
    if ngram_path.exists():
        print(f"Loading existing n-gram model: {ngram_path}")
        try:
            ngram_prior = NgramPrior.load(str(ngram_path))
        except (KeyError, ValueError) as e:
            print(f"[WARNING] Failed to load n-gram model (format may be outdated): {e}")
            print("Recomputing n-gram statistics...")
            ngram_path.unlink()  # Delete old file
            # Fall through to recomputation below
    
    if not ngram_path.exists():
        print("Computing n-gram statistics on TRAIN split...")
        
        # Create temporary dataset to extract train sequences
        train_dataset = InitialsDataset(
            corpus_path=str(initials_corpus_path),
            tokenizer=tokenizer,
            max_seq_len=args.max_seq_len,
            mode="causal",
            split="train",
            train_ratio=args.train_ratio,
            random_seed=args.random_seed,
            config=config,
        )
        
        # Debug: Verify config was passed
        print("[DEBUG] dataset has config:", hasattr(train_dataset, "config"))
        
        # Extract token sequences
        print("Extracting token sequences from train split...")
        train_sequences = []
        for i in range(len(train_dataset)):
            item = train_dataset[i]
            token_ids = item["decoder_input_ids"].tolist()
            train_sequences.append(token_ids)
        
        # Fit n-gram model
        ngram_prior = NgramPrior(vocab_size=tokenizer.vocab_size, k=args.ngram_k)
        ngram_prior.fit(train_sequences)
        
        # Save
        ngram_prior.save(str(ngram_path))
    
    # Store artifact path in config for checkpoint saving
    config.ngram_prior_path = str(ngram_path)
    
    print()
    
    # Step 4: Create dataloaders
    print("Step 4: Creating Dataloaders")
    print("-" * 70)
    
    train_loader, val_loader = create_dataloaders(
        corpus_path=str(initials_corpus_path),
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        mode=args.mode,
        taskA_prob=args.taskA_prob,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        num_workers=args.num_workers,
        val_mode=args.val_mode,  # Pass separate validation mode
        min_prompt_len=args.min_prompt_len,
        min_target_len=args.min_target_len,
        prompt_sampling=args.prompt_sampling,
        fixed_prompt_ratio=args.fixed_prompt_ratio,
        config=config,
        fixed_prompt_len=args.fixed_prompt_len,
        validation_prompt_sampling=args.validation_prompt_sampling,
        validation_fixed_prompt_len=args.validation_fixed_prompt_len,
        validation_fixed_prompt_ratio=args.validation_fixed_prompt_ratio,
    )
    
    print(f"[OK] Dataloaders created")
    print(f"  Train mode: {args.mode}")
    print(f"  Val mode: {args.val_mode}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    
    # Create separate task-specific loaders for two-phase training
    # These are always created for flexibility (phase 1 and phase 2)
    print(f"  Creating task-specific loaders for two-phase training...")
    
    # Create causal-only loader (for phase 2)
    train_dataset_causal = InitialsDataset(
        corpus_path=str(initials_corpus_path),
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        mode="causal",
        taskA_prob=0.0,
        noise_type=args.noise_type,
        noise_ratio=args.noise_ratio,
        split="train",
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        min_prompt_len=args.min_prompt_len,
        min_target_len=args.min_target_len,
        prompt_sampling=args.prompt_sampling,
        fixed_prompt_ratio=args.fixed_prompt_ratio,
        fixed_prompt_len=args.fixed_prompt_len,
        config=config,
    )
    train_loader_causal = DataLoader(
        train_dataset_causal,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_id),
        num_workers=args.num_workers,
    )
    
    # Create taskA-only loader (for phase 1)
    train_dataset_taskA = InitialsDataset(
        corpus_path=str(initials_corpus_path),
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        mode="taskA_prefix_suffix",
        taskA_prob=1.0,
        noise_type="none",
        noise_ratio=0.0,
        split="train",
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        min_prompt_len=args.min_prompt_len,
        min_target_len=args.min_target_len,
        prompt_sampling=args.prompt_sampling,
        fixed_prompt_ratio=args.fixed_prompt_ratio,
        fixed_prompt_len=args.fixed_prompt_len,
        config=config,
    )
    train_loader_taskA = DataLoader(
        train_dataset_taskA,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_id),
        num_workers=args.num_workers,
    )
    
    # Create taskA validation loader for phase 1 early stopping
    val_dataset_taskA = InitialsDataset(
        corpus_path=str(initials_corpus_path),
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        mode="taskA_prefix_suffix",
        taskA_prob=1.0,
        noise_type="none",
        noise_ratio=0.0,
        split="val",
        train_ratio=args.train_ratio,
        random_seed=args.random_seed,
        min_prompt_len=args.min_prompt_len,
        min_target_len=args.min_target_len,
        prompt_sampling=args.prompt_sampling,
        fixed_prompt_ratio=args.fixed_prompt_ratio,
        fixed_prompt_len=args.fixed_prompt_len,
        validation_prompt_sampling=args.validation_prompt_sampling,
        validation_fixed_prompt_len=args.validation_fixed_prompt_len,
        validation_fixed_prompt_ratio=args.validation_fixed_prompt_ratio,
        config=config,
    )
    val_loader_taskA = DataLoader(
        val_dataset_taskA,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_id),
        num_workers=args.num_workers,
    )
    
    print(f"  Causal train loader: {len(train_loader_causal)} batches")
    print(f"  TaskA train loader: {len(train_loader_taskA)} batches")
    print(f"  TaskA val loader: {len(val_loader_taskA)} batches")
    
    print()
    
    # Step 5: Create model
    print("Step 5: Creating Model")
    print("-" * 70)
    
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
        bos_id=tokenizer.bos_id,
        eos_id=tokenizer.eos_id,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        use_probabilistic_encoder=args.use_probabilistic_encoder,
        prior_alpha=args.prior_alpha,
        posterior_beta=args.posterior_beta,
        use_gating=args.use_gating,
        kl_weight=args.kl_weight,
        max_seq_len=args.max_seq_len,
    )
    
    model = EncoderDecoderModel(model_config, ngram_prior=ngram_prior)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[OK] Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Architecture: {args.n_encoder_layers}-layer encoder, {args.n_decoder_layers}-layer decoder")
    print(f"  d_model: {args.d_model}, n_heads: {args.n_heads}, d_ff: {args.d_ff}")
    print()
    
    # ACCEPTANCE TEST: Verify causal mode data construction
    if args.mode in ["causal", "mixed"] or args.val_mode == "causal":
        print("="*70)
        print("CAUSAL MODE ACCEPTANCE TEST")
        print("="*70)
        print("Verifying prompt→continuation split and leakage prevention...\n")
        
        # Get a batch from causal loader
        test_batch = next(iter(train_loader_causal))
        
        enc_ids = test_batch["encoder_input_ids"]
        dec_ids = test_batch["decoder_input_ids"]
        labels = test_batch["labels"]
        task_id = test_batch["task_id"]
        
        print(f"Test batch shape: encoder={enc_ids.shape}, decoder={dec_ids.shape}, labels={labels.shape}")
        
        # Test 1: Verify encoder != decoder (prompt vs continuation)
        print("\n[Test 1] Encoder ≠ Decoder check:")
        print(f"  encoder shape: {tuple(enc_ids.shape)}")
        print(f"  decoder shape: {tuple(dec_ids.shape)}")
        
        # Check if tensors are identical (handling different shapes gracefully)
        if enc_ids.shape != dec_ids.shape:
            all_equal = False
            print(f"  Shapes differ (expected for prompt→continuation)")
        else:
            all_equal = torch.equal(enc_ids, dec_ids)
            print(f"  Shapes match, checking content equality...")
        
        print(f"  All equal (tensor-wise): {all_equal}")
        
        if all_equal:
            print("  ❌ FAIL: encoder_input_ids == decoder_input_ids (OLD BEHAVIOR - LEAKAGE!)")
            print("  The encoder is seeing the full sequence including future tokens.")
            raise AssertionError("LEAKAGE: encoder_input_ids equals decoder_input_ids (old broken behavior)")
        else:
            print("  ✓ PASS: encoder_input_ids ≠ decoder_input_ids (correct prompt→continuation)")
        
        # Test 2: Verify labels are shifted relative to decoder
        print("\n[Test 2] Decoder-labels shift verification:")
        for i in range(min(3, enc_ids.shape[0])):
            dec_sample = dec_ids[i, :10].tolist()
            lab_sample = labels[i, :10].tolist()
            print(f"  Sample {i}:")
            print(f"    decoder_input_ids[:10]: {dec_sample}")
            print(f"    labels[:10]:            {lab_sample}")
            if dec_sample == lab_sample:
                print(f"    ❌ FAIL: No shifting detected!")
            else:
                print(f"    ✓ PASS: Properly shifted")
        
        # Test 3: Leakage check - token set overlap
        print("\n[Test 3] Token leakage check (encoder tokens ∩ label tokens):")
        for i in range(min(3, enc_ids.shape[0])):
            enc_tokens = set(enc_ids[i].tolist())
            lab_tokens = set(labels[i].tolist())
            
            # Remove special tokens
            special = {0, 2, 3, -100}
            enc_tokens -= special
            lab_tokens -= special
            
            if len(enc_tokens) > 0 and len(lab_tokens) > 0:
                overlap = enc_tokens & lab_tokens
                overlap_ratio = len(overlap) / min(len(enc_tokens), len(lab_tokens))
                print(f"  Sample {i}: {len(overlap)}/{min(len(enc_tokens), len(lab_tokens))} = {overlap_ratio:.1%} overlap")
                if overlap_ratio > 0.5:
                    print(f"    ⚠️ High overlap (may indicate leakage or small vocab)")
                else:
                    print(f"    ✓ Low overlap (expected for prompt→continuation)")
        
        print("\n" + "="*70)
        print("Acceptance test complete. Training will begin...\n")
    
    # Step 6: Train
    print("Step 6: Training")
    print("-" * 70)
    
    # Config was already created at the beginning - just update csv_path to use processed corpus
    config.csv_path = str(initials_corpus_path)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device,
        train_loader_causal=train_loader_causal,
        train_loader_taskA=train_loader_taskA,
        val_loader_taskA=val_loader_taskA,
    )
    
    # Override output directory
    trainer.out_dir = out_dir / "checkpoints"
    trainer.out_dir.mkdir(exist_ok=True)
    
    # Train
    trainer.train()
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Checkpoints saved to: {trainer.out_dir}")
    print(f"Best model: {trainer.out_dir / 'best_model.pt'}")
    print("\nTo generate text, run:")
    print(f"  python generate.py --checkpoint {trainer.out_dir / 'best_model.pt'} --interactive")
    print("="*70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[WARNING] Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
