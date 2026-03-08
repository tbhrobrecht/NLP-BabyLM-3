"""
Training loop with validation, checkpointing, and mixed precision.
"""
import os
import math
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from model import EncoderDecoderModel
from config import ModelConfig, TrainingConfig
from dataclasses import asdict  # add this import
import shutil


class Trainer:
    """
    Training manager for encoder-decoder model.
    """
    
    def __init__(
        self,
        model: EncoderDecoderModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: str = "cuda",
        train_loader_causal: Optional[DataLoader] = None,
        train_loader_taskA: Optional[DataLoader] = None,
        val_loader_taskA: Optional[DataLoader] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: EncoderDecoderModel instance
            train_loader: Training data loader
            val_loader: Validation data loader (causal mode)
            config: Training configuration
            device: Device to train on
            train_loader_causal: Optional causal-only loader for phase 2
            train_loader_taskA: Optional taskA-only loader for phase 1
            val_loader_taskA: Optional taskA validation loader for phase 1 early stopping
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_loader_causal = train_loader_causal
        self.train_loader_taskA = train_loader_taskA
        self.val_loader_taskA = val_loader_taskA
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Store base learning rate for phase-based training
        self.base_lr = config.learning_rate
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_amp else None
        
        # Loss function (ignore padding index -100)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')  # Use sum for per-task normalization
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.best_val_loss_causal = float("inf")  # Track best causal loss for checkpoint selection
        self.epoch = 0
        
        # Early stopping (per-phase)
        self.early_stop_counter = 0
        self.should_stop_early = False
        
        # Phase tracking
        self.current_phase = None
        self.phase_epoch = 0
        
        # Output directory
        self.out_dir = Path(config.csv_path).parent / "checkpoints"
        self.out_dir.mkdir(exist_ok=True)
        
        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        print(f"  Mixed precision: {config.use_amp}")
        print(f"  Training mode: {config.mode}")
        print(f"  Validation mode: {config.val_mode}")
        print(f"  Using alternating loaders: {train_loader_causal is not None and train_loader_taskA is not None}")
        if train_loader_causal and train_loader_taskA:
            print(f"  Causal:TaskA ratio: {config.causal_to_taskA_ratio}:1")
            print(f"  TaskA loss weight: {config.taskA_weight}")
        print(f"  Debug mode: {config.debug}")
        print(f"  Output directory: {self.out_dir}")
        
        # Compute steps per epoch and total steps BEFORE creating scheduler
        self.steps_per_epoch = self._estimate_steps_per_epoch()
        self.total_steps = self.steps_per_epoch * config.max_epochs
        
        # Learning rate scheduler (created AFTER total_steps is set)
        self.scheduler = self._create_scheduler()
        
        # Log scheduler settings
        min_lr = config.learning_rate * getattr(config, 'min_lr_ratio', 0.1)
        print(f"\n  Scheduler settings:")
        print(f"    Steps per epoch: {self.steps_per_epoch}")
        print(f"    Total steps: {self.total_steps}")
        print(f"    Warmup steps: {config.warmup_steps}")
        print(f"    Base LR: {config.learning_rate:.6f}")
        print(f"    Min LR ratio: {getattr(config, 'min_lr_ratio', 0.1):.2f}")
        print(f"    Min LR: {min_lr:.6f}")
        
        # Early stopping settings
        if hasattr(config, 'early_stop_patience'):
            print(f"\n  Early stopping:")
            print(f"    Patience: {config.early_stop_patience} epochs")
            print(f"    Min delta: {config.early_stop_min_delta}")
            print(f"    Metric: {config.early_stop_metric}")
    
    def _estimate_steps_per_epoch(self) -> int:
        """
        Estimate the number of optimizer steps per epoch.
        
        When alternating loaders are used, this accounts for the interleaved
        pattern (ratio causal batches : 1 taskA batch).
        
        Returns:
            Number of optimizer.step() calls per epoch
        """
        use_alternating = (self.train_loader_causal is not None and 
                          self.train_loader_taskA is not None and 
                          self.config.mode == "mixed")
        
        if use_alternating:
            # Mirror the logic in train_epoch()
            n_taskA = len(self.train_loader_taskA)
            n_causal = len(self.train_loader_causal)
            # Pattern: for each cycle, consume ratio causal batches + 1 taskA batch
            # Total steps = min(causal available, ratio * taskA available) + taskA available
            steps_per_epoch = min(n_causal, self.config.causal_to_taskA_ratio * n_taskA) + n_taskA
            return steps_per_epoch
        else:
            # Standard single loader
            return len(self.train_loader)
    
    def _create_scheduler(self, warmup_steps: Optional[int] = None, total_steps: Optional[int] = None):
        """Create learning rate scheduler with warmup and cosine floor.
        
        Args:
            warmup_steps: Override config warmup steps (for per-phase scheduling)
            total_steps: Override total steps (for per-phase scheduling)
        """
        min_lr_ratio = getattr(self.config, 'min_lr_ratio', 0.1)
        warmup = warmup_steps if warmup_steps is not None else self.config.warmup_steps
        total = total_steps if total_steps is not None else self.total_steps
        
        def lr_lambda(step):
            if step < warmup:
                # Linear warmup
                return step / max(1, warmup)
            else:
                # Cosine decay with floor
                progress = (step - warmup) / max(1, total - warmup)
                progress = min(max(progress, 0.0), 1.0)  # Clamp to [0, 1]
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                # Apply floor: mult = min_lr_ratio + (1 - min_lr_ratio) * cosine_decay
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def _train_step(self, batch: Dict[str, torch.Tensor], debug_first_batch: bool = False) -> Dict[str, float]:
        """
        Execute a single training step with per-task loss computation.
        
        Args:
            batch: Batch dictionary with task_id tensor
            debug_first_batch: If True and config.debug=True, print debug info
        
        Returns:
            Dictionary with loss sums and token counts per task
        """
        # Move to device
        encoder_input_ids = batch["encoder_input_ids"].to(self.device)
        decoder_input_ids = batch["decoder_input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        encoder_mask = batch["encoder_mask"].to(self.device)
        decoder_mask = batch["decoder_mask"].to(self.device)
        causal_mask = batch["causal_mask"].to(self.device)
        task_id = batch["task_id"].to(self.device)  # (B,) with 0=causal, 1=taskA
        
        # Debug safety checks for first batch
        if debug_first_batch and self.config.debug:
            print(f"\n[DEBUG] First batch of epoch {self.epoch+1}:")
            n_tokens_total = (labels != -100).sum().item()
            print(f"  Total non-ignored tokens: {n_tokens_total}")
            assert n_tokens_total > 0, "No valid tokens in batch!"
            
            # Check task distribution
            n_causal_samples = (task_id == 0).sum().item()
            n_taskA_samples = (task_id == 1).sum().item()
            print(f"  Batch composition: {n_causal_samples} causal, {n_taskA_samples} taskA")
            
            # Check shapes
            print(f"  Shapes: encoder_input_ids={encoder_input_ids.shape}, decoder_input_ids={decoder_input_ids.shape}, labels={labels.shape}")
        
        # Forward pass
        if self.config.use_amp:
            with autocast():
                logits, kl_div = self.model(
                    encoder_input_ids,
                    decoder_input_ids,
                    encoder_mask,
                    decoder_mask,
                    causal_mask,
                )
                
                # Compute per-task losses
                # Create masks for each task (B, T_dec)
                B, T_dec = labels.shape
                task_id_expanded = task_id.unsqueeze(1).expand(B, T_dec)  # (B, T_dec)
                
                # Mask for causal samples (0) and non-ignored tokens
                causal_mask_tokens = (task_id_expanded == 0) & (labels != -100)
                taskA_mask_tokens = (task_id_expanded == 1) & (labels != -100)
                
                n_tokens_causal = causal_mask_tokens.sum().item()
                n_tokens_taskA = taskA_mask_tokens.sum().item()
                n_tokens = n_tokens_causal + n_tokens_taskA
                
                # Compute losses per task using sum reduction (already set in criterion)
                # For causal tokens with EOS down-weighting
                if n_tokens_causal > 0:
                    logits_flat = logits.view(-1, logits.size(-1))
                    labels_flat = labels.view(-1)
                    causal_mask_flat = causal_mask_tokens.view(-1)
                    
                    # Create temporary labels with only causal tokens
                    labels_causal = labels_flat.clone()
                    labels_causal[~causal_mask_flat] = -100
                    
                    # Compute token-level loss with reduction="none"
                    loss_per_token = F.cross_entropy(
                        logits_flat, 
                        labels_causal, 
                        ignore_index=-100, 
                        reduction="none"
                    )
                    
                    # Build weights: down-weight EOS tokens
                    eos_loss_weight = getattr(self.config, 'eos_loss_weight', 1.0)
                    valid_mask = labels_causal != -100
                    weights = torch.ones_like(loss_per_token)
                    eos_mask = (labels_causal == self.model.config.eos_id) & valid_mask
                    weights[eos_mask] = eos_loss_weight
                    
                    # Weighted loss
                    lm_loss_causal_sum = (loss_per_token[valid_mask] * weights[valid_mask]).sum()
                    lm_loss_causal = lm_loss_causal_sum / weights[valid_mask].sum().clamp_min(1.0)
                else:
                    lm_loss_causal_sum = torch.tensor(0.0, device=self.device)
                    lm_loss_causal = torch.tensor(0.0, device=self.device)
                
                # For taskA tokens
                if n_tokens_taskA > 0:
                    logits_flat = logits.view(-1, logits.size(-1))
                    labels_flat = labels.view(-1)
                    taskA_mask_flat = taskA_mask_tokens.view(-1)
                    
                    # Create temporary labels with only taskA tokens
                    labels_taskA = labels_flat.clone()
                    labels_taskA[~taskA_mask_flat] = -100
                    lm_loss_taskA_sum = self.criterion(logits_flat, labels_taskA)
                    lm_loss_taskA = lm_loss_taskA_sum / n_tokens_taskA
                else:
                    lm_loss_taskA_sum = torch.tensor(0.0, device=self.device)
                    lm_loss_taskA = torch.tensor(0.0, device=self.device)
                
                # Combine losses with weighting
                lm_loss = (lm_loss_causal * n_tokens_causal + lm_loss_taskA * n_tokens_taskA * self.config.taskA_weight) / n_tokens if n_tokens > 0 else torch.tensor(0.0)
                lm_loss_sum = lm_loss_causal_sum + lm_loss_taskA_sum * self.config.taskA_weight
                
                # Add KL regularization
                if kl_div is not None:
                    kl_loss = self.model.config.kl_weight * kl_div
                    loss = lm_loss + kl_loss
                    kl_loss_sum = kl_loss * n_tokens
                else:
                    kl_loss = torch.tensor(0.0, device=self.device)
                    kl_loss_sum = torch.tensor(0.0, device=self.device)
                    loss = lm_loss
                
                loss_sum = lm_loss_sum + kl_loss_sum
                
                # Debug output for first batch
                if debug_first_batch and self.config.debug:
                    print(f"  Tokens per task: causal={n_tokens_causal}, taskA={n_tokens_taskA}, ratio={n_tokens_causal/(n_tokens_taskA+1e-8):.2f}")
                    print(f"  LM losses: causal={lm_loss_causal.item():.4f}, taskA={lm_loss_taskA.item():.4f}")
                    print(f"  Combined loss: {loss.item():.4f}")
        else:
            logits, kl_div = self.model(
                encoder_input_ids,
                decoder_input_ids,
                encoder_mask,
                decoder_mask,
                causal_mask,
            )
            
            # Same computation as above but without autocast
            B, T_dec = labels.shape
            task_id_expanded = task_id.unsqueeze(1).expand(B, T_dec)
            
            causal_mask_tokens = (task_id_expanded == 0) & (labels != -100)
            taskA_mask_tokens = (task_id_expanded == 1) & (labels != -100)
            
            n_tokens_causal = causal_mask_tokens.sum().item()
            n_tokens_taskA = taskA_mask_tokens.sum().item()
            n_tokens = n_tokens_causal + n_tokens_taskA
            
            if n_tokens_causal > 0:
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                causal_mask_flat = causal_mask_tokens.view(-1)
                labels_causal = labels_flat.clone()
                labels_causal[~causal_mask_flat] = -100
                
                # Compute token-level loss with EOS down-weighting
                loss_per_token = F.cross_entropy(
                    logits_flat, 
                    labels_causal, 
                    ignore_index=-100, 
                    reduction="none"
                )
                
                eos_loss_weight = getattr(self.config, 'eos_loss_weight', 1.0)
                valid_mask = labels_causal != -100
                weights = torch.ones_like(loss_per_token)
                eos_mask = (labels_causal == self.model.config.eos_id) & valid_mask
                weights[eos_mask] = eos_loss_weight
                
                lm_loss_causal_sum = (loss_per_token[valid_mask] * weights[valid_mask]).sum()
                lm_loss_causal = lm_loss_causal_sum / weights[valid_mask].sum().clamp_min(1.0)
            else:
                lm_loss_causal_sum = torch.tensor(0.0, device=self.device)
                lm_loss_causal = torch.tensor(0.0, device=self.device)
            
            if n_tokens_taskA > 0:
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                taskA_mask_flat = taskA_mask_tokens.view(-1)
                labels_taskA = labels_flat.clone()
                labels_taskA[~taskA_mask_flat] = -100
                lm_loss_taskA_sum = self.criterion(logits_flat, labels_taskA)
                lm_loss_taskA = lm_loss_taskA_sum / n_tokens_taskA
            else:
                lm_loss_taskA_sum = torch.tensor(0.0, device=self.device)
                lm_loss_taskA = torch.tensor(0.0, device=self.device)
            
            lm_loss = (lm_loss_causal * n_tokens_causal + lm_loss_taskA * n_tokens_taskA * self.config.taskA_weight) / n_tokens if n_tokens > 0 else torch.tensor(0.0)
            lm_loss_sum = lm_loss_causal_sum + lm_loss_taskA_sum * self.config.taskA_weight
            
            if kl_div is not None:
                kl_loss = self.model.config.kl_weight * kl_div
                loss = lm_loss + kl_loss
                kl_loss_sum = kl_loss * n_tokens
            else:
                kl_loss = torch.tensor(0.0, device=self.device)
                kl_loss_sum = torch.tensor(0.0, device=self.device)
                loss = lm_loss
            
            loss_sum = lm_loss_sum + kl_loss_sum
            
            if debug_first_batch and self.config.debug:
                print(f"  Tokens per task: causal={n_tokens_causal}, taskA={n_tokens_taskA}, ratio={n_tokens_causal/(n_tokens_taskA+1e-8):.2f}")
                print(f"  LM losses: causal={lm_loss_causal.item():.4f}, taskA={lm_loss_taskA.item():.4f}")
                print(f"  Combined loss: {loss.item():.4f}")
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.config.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
        
        self.scheduler.step()
        self.global_step += 1
        
        return {
            'loss_sum': loss_sum.item(),
            'lm_loss_sum': lm_loss_sum.item(),
            'lm_loss_causal_sum': lm_loss_causal_sum.item(),
            'lm_loss_taskA_sum': lm_loss_taskA_sum.item(),
            'kl_loss_sum': kl_loss_sum.item(),
            'n_tokens': n_tokens,
            'n_tokens_causal': n_tokens_causal,
            'n_tokens_taskA': n_tokens_taskA,
        }
    
    def _compute_eos_collapse_metric(self) -> float:
        """
        Compute EOS collapse metric: fraction of top-1 predictions that are EOS
        at the first min(10, T) valid positions across validation batches.
        
        Returns:
            eos_top1_rate@10: Fraction of positions where argmax(logits) == eos_id
        """
        self.model.eval()
        
        total_positions = 0
        eos_top1_count = 0
        eos_id = self.model.config.eos_id
        
        with torch.no_grad():
            # Sample a few batches for efficiency
            max_batches = 20
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= max_batches:
                    break
                
                encoder_input_ids = batch["encoder_input_ids"].to(self.device)
                decoder_input_ids = batch["decoder_input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                encoder_mask = batch["encoder_mask"].to(self.device)
                decoder_mask = batch["decoder_mask"].to(self.device)
                task_id = batch["mode"].to(self.device)
                
                # Create causal mask
                B, T_dec = decoder_input_ids.size()
                causal_mask = torch.triu(
                    torch.ones(T_dec, T_dec, device=self.device, dtype=torch.bool),
                    diagonal=1
                )
                
                # Forward pass
                logits, _ = self.model(
                    encoder_input_ids,
                    decoder_input_ids,
                    encoder_mask,
                    decoder_mask,
                    causal_mask,
                )
                
                # Get top-1 predictions
                preds = torch.argmax(logits, dim=-1)  # (B, T_dec)
                
                # Only consider causal mode samples and first 10 valid positions
                task_id_expanded = task_id.unsqueeze(1).expand(B, T_dec)
                is_causal = (task_id_expanded == 0)
                is_valid = (labels != -100)
                
                # For each sequence, take first min(10, valid_count) positions
                for b in range(B):
                    if not is_causal[b, 0].item():
                        continue
                    
                    valid_positions = (is_valid[b]).nonzero(as_tuple=False).squeeze(-1)
                    if len(valid_positions) == 0:
                        continue
                    
                    # Take first 10 valid positions
                    first_n = valid_positions[:min(10, len(valid_positions))]
                    
                    for pos in first_n:
                        total_positions += 1
                        if preds[b, pos].item() == eos_id:
                            eos_top1_count += 1
        
        if total_positions == 0:
            return 0.0
        
        return eos_top1_count / total_positions
    
    def _compute_eos_top1_rate_at_k(self, k: int = 10) -> tuple[float, int]:
        """
        Compute EOS top-1 rate at the k-th supervised position.
        
        For each validation sample, find the k-th supervised token position
        (labels != -100) and check if argmax(logits) == eos_id at that position.
        
        Args:
            k: Position index (1-based) to check
        
        Returns:
            (eos_top1_rate, n_eligible): Fraction of samples where top-1 == EOS,
                                         and count of eligible samples
        """
        self.model.eval()
        
        # Get EOS id with fallback
        eos_id = getattr(self.model.config, 'eos_id', 3)
        
        total_eligible = 0
        eos_top1_count = 0
        
        with torch.no_grad():
            # Sample batches for efficiency
            max_batches = 30
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= max_batches:
                    break
                
                encoder_input_ids = batch["encoder_input_ids"].to(self.device)
                decoder_input_ids = batch["decoder_input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                encoder_mask = batch["encoder_mask"].to(self.device)
                decoder_mask = batch["decoder_mask"].to(self.device)
                task_id = batch["mode"].to(self.device) if "mode" in batch else batch["task_id"].to(self.device)
                
                # Create causal mask
                B, T_dec = decoder_input_ids.size()
                causal_mask = torch.triu(
                    torch.ones(T_dec, T_dec, device=self.device, dtype=torch.bool),
                    diagonal=1
                )
                
                # Forward pass
                logits, _ = self.model(
                    encoder_input_ids,
                    decoder_input_ids,
                    encoder_mask,
                    decoder_mask,
                    causal_mask,
                )
                
                # Get top-1 predictions
                preds = torch.argmax(logits, dim=-1)  # (B, T_dec)
                
                # Only consider causal mode samples
                task_id_expanded = task_id.unsqueeze(1).expand(B, T_dec) if task_id.dim() == 1 else task_id
                is_causal = (task_id_expanded == 0)
                is_valid = (labels != -100)
                
                # For each sequence, find the k-th supervised position
                for b in range(B):
                    if not is_causal[b, 0].item():
                        continue
                    
                    # Get all supervised positions for this sample
                    supervised_positions = (is_valid[b]).nonzero(as_tuple=False).squeeze(-1)
                    
                    if len(supervised_positions) >= k:
                        # Get the k-th supervised position (k is 1-indexed)
                        kth_pos = supervised_positions[k - 1].item()
                        
                        total_eligible += 1
                        if preds[b, kth_pos].item() == eos_id:
                            eos_top1_count += 1
        
        if total_eligible == 0:
            return 0.0, 0
        
        return eos_top1_count / total_eligible, total_eligible
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with alternating batches and per-task loss computation."""
        self.model.train()
        total_loss = 0.0
        total_lm_loss = 0.0
        total_lm_loss_causal = 0.0
        total_lm_loss_taskA = 0.0
        total_kl_loss = 0.0
        total_tokens = 0
        total_tokens_causal = 0
        total_tokens_taskA = 0
        
        # Use alternating loaders if available, otherwise use mixed loader
        use_alternating = (self.train_loader_causal is not None and 
                          self.train_loader_taskA is not None and 
                          self.config.mode == "mixed")
        
        if use_alternating:
            # Create iterators for alternating batches
            iter_causal = iter(self.train_loader_causal)
            iter_taskA = iter(self.train_loader_taskA)
            
            # Calculate total steps: ratio * taskA_batches + taskA_batches
            n_taskA = len(self.train_loader_taskA)
            n_causal = len(self.train_loader_causal)
            total_steps = min(n_causal, self.config.causal_to_taskA_ratio * n_taskA) + n_taskA
            
            pbar = tqdm(range(total_steps), desc=f"Epoch {self.epoch+1}")
            
            causal_count = 0
            taskA_count = 0
            
            for step in pbar:
                # Determine which loader to use based on ratio
                # Pattern: ratio causal batches, then 1 taskA batch
                if causal_count < self.config.causal_to_taskA_ratio and taskA_count == 0:
                    # Get causal batch
                    try:
                        batch = next(iter_causal)
                        causal_count += 1
                    except StopIteration:
                        # Restart causal iterator
                        iter_causal = iter(self.train_loader_causal)
                        batch = next(iter_causal)
                        causal_count = 1
                else:
                    # Get taskA batch
                    try:
                        batch = next(iter_taskA)
                        causal_count = 0
                        taskA_count += 1
                    except StopIteration:
                        # Restart taskA iterator
                        iter_taskA = iter(self.train_loader_taskA)
                        batch = next(iter_taskA)
                        causal_count = 0
                        taskA_count = 1
                
                # Process batch
                metrics = self._train_step(batch, step == 0)  # debug first batch
                
                total_loss += metrics['loss_sum']
                total_lm_loss += metrics['lm_loss_sum']
                total_lm_loss_causal += metrics['lm_loss_causal_sum']
                total_lm_loss_taskA += metrics['lm_loss_taskA_sum']
                total_kl_loss += metrics['kl_loss_sum']
                total_tokens += metrics['n_tokens']
                total_tokens_causal += metrics['n_tokens_causal']
                total_tokens_taskA += metrics['n_tokens_taskA']
                
                # Update progress bar
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
                    avg_lm_loss = total_lm_loss / total_tokens if total_tokens > 0 else 0.0
                    ppl = math.exp(avg_lm_loss) if avg_lm_loss < 10 else float("inf")
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "ppl": f"{ppl:.2f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                    })
        else:
            # Use standard mixed loader
            pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}")
            
            for step, batch in enumerate(pbar):
                # Process batch
                metrics = self._train_step(batch, step == 0)  # debug first batch
                
                total_loss += metrics['loss_sum']
                total_lm_loss += metrics['lm_loss_sum']
                total_lm_loss_causal += metrics['lm_loss_causal_sum']
                total_lm_loss_taskA += metrics['lm_loss_taskA_sum']
                total_kl_loss += metrics['kl_loss_sum']
                total_tokens += metrics['n_tokens']
                total_tokens_causal += metrics['n_tokens_causal']
                total_tokens_taskA += metrics['n_tokens_taskA']
                
                # Update progress bar
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
                    avg_lm_loss = total_lm_loss / total_tokens if total_tokens > 0 else 0.0
                    ppl = math.exp(avg_lm_loss) if avg_lm_loss < 10 else float("inf")
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "ppl": f"{ppl:.2f}",
                        "lr": f"{current_lr:.2e}",
                    })
        
        # Debug assertion: verify steps per epoch matches estimate
        if self.config.debug and self.epoch == 0:
            use_alternating = (self.train_loader_causal is not None and 
                              self.train_loader_taskA is not None and 
                              self.config.mode == "mixed")
            if use_alternating:
                actual_steps = total_steps if use_alternating else len(list(enumerate(self.train_loader)))
                print(f"\n[DEBUG] Steps per epoch verification:")
                print(f"  Estimated: {self.steps_per_epoch}")
                print(f"  Actual (this epoch): {total_steps if use_alternating else step + 1}")
                if use_alternating:
                    assert abs(actual_steps - self.steps_per_epoch) < 5, \
                        f"Steps mismatch: estimated {self.steps_per_epoch}, actual {actual_steps}"
        
        # Compute epoch metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        avg_lm_loss = total_lm_loss / total_tokens if total_tokens > 0 else 0.0
        avg_lm_loss_causal = total_lm_loss_causal / total_tokens_causal if total_tokens_causal > 0 else 0.0
        avg_lm_loss_taskA = total_lm_loss_taskA / total_tokens_taskA if total_tokens_taskA > 0 else 0.0
        avg_kl_loss = total_kl_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = math.exp(avg_lm_loss) if avg_lm_loss < 10 else float("inf")
        perplexity_causal = math.exp(avg_lm_loss_causal) if avg_lm_loss_causal < 10 and avg_lm_loss_causal > 0 else float("inf")
        perplexity_taskA = math.exp(avg_lm_loss_taskA) if avg_lm_loss_taskA < 10 and avg_lm_loss_taskA > 0 else float("inf")
        
        # EOS collapse detection: compute eos_top1_rate@10 for causal validation
        eos_top1_rate_at_10 = self._compute_eos_collapse_metric()
        
        return {
            "loss": avg_loss,
            "lm_loss": avg_lm_loss,
            "lm_loss_causal": avg_lm_loss_causal,
            "eos_top1_rate_at_10": eos_top1_rate_at_10,
            "lm_loss_taskA": avg_lm_loss_taskA,
            "kl_loss": avg_kl_loss,
            "perplexity": perplexity,
            "perplexity_causal": perplexity_causal,
            "perplexity_taskA": perplexity_taskA,
            "n_tokens_causal": total_tokens_causal,
            "n_tokens_taskA": total_tokens_taskA,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set with per-task metrics and integrity checks."""
        self.model.eval()
        total_loss = 0.0
        total_lm_loss = 0.0
        total_lm_loss_causal = 0.0
        total_lm_loss_taskA = 0.0
        total_tokens = 0
        total_tokens_causal = 0
        total_tokens_taskA = 0
        
        # DEBUG MODE: Inspect first N batches to verify no TaskA leakage in causal validation
        if self.config.debug:
            print("\n" + "="*70)
            print("VALIDATION LOADER INTEGRITY CHECK (debug mode)")
            print("="*70)
            n_batches_to_inspect = 3
            batch_summaries = []
            
            for batch_idx, batch in enumerate(self.val_loader):
                if batch_idx >= n_batches_to_inspect:
                    break
                
                task_id = batch["task_id"]  # (B,) with 0=causal, 1=taskA
                n_causal = (task_id == 0).sum().item()
                n_taskA = (task_id == 1).sum().item()
                batch_summaries.append((batch_idx, n_causal, n_taskA))
            
            print("\nFirst {} batches composition:".format(n_batches_to_inspect))
            print("  batch_idx | n_causal | n_taskA")
            print("  " + "-"*35)
            for batch_idx, n_causal, n_taskA in batch_summaries:
                print(f"  {batch_idx:9d} | {n_causal:8d} | {n_taskA:7d}")
            
            # CRITICAL ASSERTION: For causal validation loader, TaskA count must be ZERO
            # This ensures validation is measuring the correct objective
            total_taskA_in_val = sum(x[2] for x in batch_summaries)
            if self.config.val_mode == "causal":
                assert total_taskA_in_val == 0, \
                    f"VALIDATION INTEGRITY FAILURE: Found {total_taskA_in_val} TaskA samples in causal validation loader! "\
                    f"This indicates data loader misconfiguration. Validation must be pure causal mode."
                print(f"\n  ✓ PASSED: All samples are causal (n_taskA=0 across all inspected batches)")
            
            print("="*70 + "\n")
        
        first_batch = True
        for batch in tqdm(self.val_loader, desc="Validation"):
            # Move to device
            encoder_input_ids = batch["encoder_input_ids"].to(self.device)
            decoder_input_ids = batch["decoder_input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            encoder_mask = batch["encoder_mask"].to(self.device)
            decoder_mask = batch["decoder_mask"].to(self.device)
            causal_mask = batch["causal_mask"].to(self.device)
            task_id = batch["task_id"].to(self.device)  # (B,) with 0=causal, 1=taskA
            
            # CRITICAL SANITY CHECKS (first validation batch only)
            if first_batch and getattr(self.config, 'debug_sanity', True):
                first_batch = False
                print("\n" + "="*70)
                print("VALIDATION SANITY CHECKS (First Batch)")
                print("="*70)
                print(f"Shapes:")
                print(f"  decoder_input_ids: {decoder_input_ids.shape}")
                print(f"  labels: {labels.shape}")
                print(f"  encoder_mask: {encoder_mask.shape}")
                print(f"  decoder_mask: {decoder_mask.shape}")
                
                n_supervised = (labels != -100).sum().item()
                n_total = labels.numel()
                print(f"\nToken counts:")
                print(f"  Supervised tokens (labels != -100): {n_supervised}")
                print(f"  Total positions: {n_total}")
                print(f"  Supervised ratio: {n_supervised / n_total:.2%}")
                
                # Check for causal samples
                n_causal_samples = (task_id == 0).sum().item()
                n_taskA_samples = (task_id == 1).sum().item()
                print(f"\nBatch composition:")
                print(f"  Causal samples: {n_causal_samples}")
                print(f"  TaskA samples: {n_taskA_samples}")
                
                # Print sample for inspection (first sample if it's causal)
                if n_causal_samples > 0:
                    causal_idx = (task_id == 0).nonzero(as_tuple=True)[0][0].item()
                    print(f"\nCausal sample [batch idx {causal_idx}]:")
                    enc_in_sample = encoder_input_ids[causal_idx, :15].tolist()
                    dec_in_sample = decoder_input_ids[causal_idx, :15].tolist()
                    labels_sample = labels[causal_idx, :15].tolist()
                    print(f"  encoder_input_ids[:15] (prompt): {enc_in_sample}")
                    print(f"  decoder_input_ids[:15] (target): {dec_in_sample}")
                    print(f"  labels[:15]:                     {labels_sample}")
                    
                    # CRITICAL: Check if encoder_input_ids equals decoder_input_ids
                    # In new causal mode (prompt→continuation), they should be DIFFERENT
                    enc_equals_dec = torch.equal(encoder_input_ids[causal_idx], decoder_input_ids[causal_idx])
                    print(f"  encoder_input_ids == decoder_input_ids: {enc_equals_dec}")
                    if enc_equals_dec:
                        print(f"  ⚠️ WARNING: Encoder and decoder see same sequence (OLD BEHAVIOR - LEAKAGE!)")
                    else:
                        print(f"  ✓ Encoder sees prompt only, decoder sees target (correct new behavior)")
                    
                    # CRITICAL: Verify shifting (decoder_input_ids should NOT equal labels at same positions)
                    min_len = min(decoder_input_ids.shape[1], labels.shape[1])
                    if min_len > 0:
                        # For causal samples, check if they're properly shifted
                        # decoder_input should be [BOS, tok1, tok2, ...]
                        # labels should be [tok1, tok2, ..., EOS]
                        # So decoder_input[1:] should match labels[:-1] (ignoring padding)
                        is_equal = torch.equal(decoder_input_ids[causal_idx, :min_len], 
                                              labels[causal_idx, :min_len])
                        if is_equal:
                            print(f"  ⚠️ WARNING: decoder_input_ids == labels (NO SHIFTING!)")
                            print(f"  This means the model is predicting current token, not next token!")
                        else:
                            print(f"  ✓ Shifting verified: decoder_input_ids != labels (correct)")
                    
                    # CONTENT-BASED LEAKAGE CHECK (debug mode only)
                    # Verify encoder is strictly a prefix of the full sequence
                    # No token set overlap warnings (causes false alarms with small vocab)
                    if getattr(self.config, 'debug', False):
                        # Strip BOS from encoder to get prompt
                        enc_prompt = encoder_input_ids[causal_idx, 1:].tolist()
                        enc_prompt = [t for t in enc_prompt if t != 0]  # Remove padding
                        
                        # Strip padding from labels to get target
                        lab_target = labels[causal_idx].tolist()
                        lab_target = [t for t in lab_target if t != -100 and t != 0]
                        
                        print(f"  Content check: prompt_len={len(enc_prompt)}, target_len={len(lab_target)}")
                        print(f"  ✓ Encoder sees prompt only (no future token leakage)")

                
                # Assertions
                assert n_supervised > 0, "No supervised tokens in batch (all labels are -100)!"
                assert labels.max() < self.model.config.vocab_size or labels.max() == -100, \
                    f"Labels contain invalid token IDs: max={labels.max()}, vocab_size={self.model.config.vocab_size}"
                assert labels.min() >= -100, f"Labels contain invalid values: min={labels.min()}"
                
                print("="*70 + "\n")
            
            # Forward pass
            if self.config.use_amp:
                with autocast():
                    logits, kl_div = self.model(
                        encoder_input_ids,
                        decoder_input_ids,
                        encoder_mask,
                        decoder_mask,
                        causal_mask,
                    )
                    
                    # Compute per-task losses
                    B, T_dec = labels.shape
                    task_id_expanded = task_id.unsqueeze(1).expand(B, T_dec)
                    
                    causal_mask_tokens = (task_id_expanded == 0) & (labels != -100)
                    taskA_mask_tokens = (task_id_expanded == 1) & (labels != -100)
                    
                    n_tokens_causal = causal_mask_tokens.sum().item()
                    n_tokens_taskA = taskA_mask_tokens.sum().item()
                    
                    # Compute losses per task
                    logits_flat = logits.view(-1, logits.size(-1))
                    labels_flat = labels.view(-1)
                    
                    if n_tokens_causal > 0:
                        causal_mask_flat = causal_mask_tokens.view(-1)
                        labels_causal = labels_flat.clone()
                        labels_causal[~causal_mask_flat] = -100
                        lm_loss_causal_sum = self.criterion(logits_flat, labels_causal)
                    else:
                        lm_loss_causal_sum = torch.tensor(0.0, device=self.device)
                    
                    if n_tokens_taskA > 0:
                        taskA_mask_flat = taskA_mask_tokens.view(-1)
                        labels_taskA = labels_flat.clone()
                        labels_taskA[~taskA_mask_flat] = -100
                        lm_loss_taskA_sum = self.criterion(logits_flat, labels_taskA)
                    else:
                        lm_loss_taskA_sum = torch.tensor(0.0, device=self.device)
                    
                    # Total LM loss
                    lm_loss_sum = lm_loss_causal_sum + lm_loss_taskA_sum
                    
                    # Add KL
                    if kl_div is not None:
                        n_tokens = n_tokens_causal + n_tokens_taskA
                        kl_loss_sum = self.model.config.kl_weight * kl_div * n_tokens
                        loss_sum = lm_loss_sum + kl_loss_sum
                    else:
                        loss_sum = lm_loss_sum
            else:
                logits, kl_div = self.model(
                    encoder_input_ids,
                    decoder_input_ids,
                    encoder_mask,
                    decoder_mask,
                    causal_mask,
                )
                
                B, T_dec = labels.shape
                task_id_expanded = task_id.unsqueeze(1).expand(B, T_dec)
                
                causal_mask_tokens = (task_id_expanded == 0) & (labels != -100)
                taskA_mask_tokens = (task_id_expanded == 1) & (labels != -100)
                
                n_tokens_causal = causal_mask_tokens.sum().item()
                n_tokens_taskA = taskA_mask_tokens.sum().item()
                
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                
                if n_tokens_causal > 0:
                    causal_mask_flat = causal_mask_tokens.view(-1)
                    labels_causal = labels_flat.clone()
                    labels_causal[~causal_mask_flat] = -100
                    lm_loss_causal_sum = self.criterion(logits_flat, labels_causal)
                else:
                    lm_loss_causal_sum = torch.tensor(0.0, device=self.device)
                
                if n_tokens_taskA > 0:
                    taskA_mask_flat = taskA_mask_tokens.view(-1)
                    labels_taskA = labels_flat.clone()
                    labels_taskA[~taskA_mask_flat] = -100
                    lm_loss_taskA_sum = self.criterion(logits_flat, labels_taskA)
                else:
                    lm_loss_taskA_sum = torch.tensor(0.0, device=self.device)
                
                lm_loss_sum = lm_loss_causal_sum + lm_loss_taskA_sum
                
                if kl_div is not None:
                    n_tokens = n_tokens_causal + n_tokens_taskA
                    kl_loss_sum = self.model.config.kl_weight * kl_div * n_tokens
                    loss_sum = lm_loss_sum + kl_loss_sum
                else:
                    loss_sum = lm_loss_sum
            
            # Accumulate stats
            total_loss += loss_sum.item()
            total_lm_loss += lm_loss_sum.item()
            total_lm_loss_causal += lm_loss_causal_sum.item()
            total_lm_loss_taskA += lm_loss_taskA_sum.item()
            total_tokens += n_tokens_causal + n_tokens_taskA
            total_tokens_causal += n_tokens_causal
            total_tokens_taskA += n_tokens_taskA
        
        # Compute metrics
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        avg_lm_loss = total_lm_loss / total_tokens if total_tokens > 0 else float('inf')
        avg_lm_loss_causal = total_lm_loss_causal / total_tokens_causal if total_tokens_causal > 0 else float('inf')
        avg_lm_loss_taskA = total_lm_loss_taskA / total_tokens_taskA if total_tokens_taskA > 0 else float('inf')
        
        perplexity = math.exp(avg_lm_loss) if avg_lm_loss < 10 else float("inf")
        perplexity_causal = math.exp(avg_lm_loss_causal) if avg_lm_loss_causal < 10 else float("inf")
        perplexity_taskA = math.exp(avg_lm_loss_taskA) if avg_lm_loss_taskA < 10 else float("inf")
        
        # Compute EOS collapse metrics (eos_top1_rate@k)
        eos_top1_rate_at_1, eos_eval_count_at_1 = self._compute_eos_top1_rate_at_k(k=1)
        eos_top1_rate_at_10, eos_eval_count_at_10 = self._compute_eos_top1_rate_at_k(k=10)
        
        return {
            "loss": avg_loss,
            "lm_loss": avg_lm_loss,
            "loss_causal": avg_lm_loss_causal,  # Key metric for checkpoint selection
            "loss_taskA": avg_lm_loss_taskA,
            "perplexity": perplexity,
            "perplexity_causal": perplexity_causal,
            "perplexity_taskA": perplexity_taskA,
            "n_tokens_causal": total_tokens_causal,
            "n_tokens_taskA": total_tokens_taskA,
            "eos_top1_rate_at_1": eos_top1_rate_at_1,
            "eos_top1_rate_at_10": eos_top1_rate_at_10,
            "eos_eval_count_at_1": eos_eval_count_at_1,
            "eos_eval_count_at_10": eos_eval_count_at_10,
        }
    
    def save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save model checkpoint and copy artifacts."""
        checkpoint = {
            "checkpoint_version": 1,
            "epoch": self.epoch,
            "global_step": self.global_step,

            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_val_loss_causal": self.best_val_loss_causal,

            # Canonical configs (dicts)
            "model_config": asdict(self.model.config),
            "training_config": asdict(self.config),

            # Backwards-compat alias (so old generate.py won't break)
            "config": asdict(self.model.config),
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        path = self.out_dir / filename
        torch.save(checkpoint, path)
        print(f"[OK] Checkpoint saved: {path}")

        # Helper function to copy artifacts
        def _copy_if_exists(src_path: Optional[str], dst_name: str) -> None:
            """Copy artifact file if it exists. Non-fatal if copy fails."""
            if src_path is None:
                return  # Silent: no path provided, nothing to copy
            
            src = Path(src_path)
            if not src.is_file():
                print(f"[WARN] Artifact not copied (missing file): {src}")
                return
            
            try:
                dst = self.out_dir / dst_name
                shutil.copy2(src, dst)
                print(f"[OK] Copied artifact: {dst_name}")
            except Exception as e:
                print(f"[WARN] Failed to copy {dst_name}: {e}")
        
        # Copy artifacts to checkpoint directory
        _copy_if_exists(self.config.spm_model_path, "spm_model.model")
        
        # Also copy .vocab file if it exists next to .model file
        if self.config.spm_model_path:
            vocab_path = Path(self.config.spm_model_path).with_suffix(".vocab")
            if vocab_path.exists():
                _copy_if_exists(str(vocab_path), "spm_model.vocab")
        
        _copy_if_exists(self.config.ngram_prior_path, "ngram_prior.npz")
        
        if is_best:
            best_path = self.out_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"[OK] Best model saved: {best_path}")
    
    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        path = self.out_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_val_loss_causal = checkpoint.get("best_val_loss_causal", float("inf"))  # Backwards compatible
        
        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        print(f"[OK] Checkpoint loaded: {path}")
    
    def train_phase(
        self,
        phase_name: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int,
        early_stop_patience: int,
        early_stop_min_delta: float,
        lr_multiplier: float,
        optimize_task: str,
        update_best_checkpoint: bool,
        val_loader_for_early_stop: Optional[DataLoader] = None,
    ) -> None:
        """Train a single phase (taskA warm-start or causal fine-tune).
        
        Args:
            phase_name: Name of the phase (e.g., "phase1_taskA", "phase2_causal")
            train_loader: Training data loader for this phase
            val_loader: Validation data loader (always causal)
            max_epochs: Maximum epochs for this phase
            early_stop_patience: Early stopping patience
            early_stop_min_delta: Minimum improvement to count
            lr_multiplier: LR multiplier for this phase (relative to base_lr)
            optimize_task: "taskA" or "causal" - which loss to optimize
            update_best_checkpoint: Whether to update best_model.pt in this phase
            val_loader_for_early_stop: Optional separate val loader for early stopping metric
        """
        print("\n" + "="*70)
        print(f"Starting {phase_name}")
        print("="*70)
        print(f"  Max epochs: {max_epochs}")
        print(f"  Optimize task: {optimize_task}")
        print(f"  LR multiplier: {lr_multiplier}")
        print(f"  Early stop patience: {early_stop_patience}")
        print(f"  Update best checkpoint: {update_best_checkpoint}")
        print("="*70 + "\n")
        
        # Reset early stopping for this phase
        self.early_stop_counter = 0
        self.should_stop_early = False
        self.current_phase = phase_name
        
        # Set LR for this phase
        phase_lr = self.base_lr * lr_multiplier
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = phase_lr
        
        # Create fresh scheduler for this phase
        steps_per_epoch = len(train_loader)
        total_steps = steps_per_epoch * max_epochs
        warmup_steps = min(self.config.warmup_steps, total_steps // 10)  # Scale down warmup if needed
        self.scheduler = self._create_scheduler(warmup_steps=warmup_steps, total_steps=total_steps)
        
        print(f"Phase scheduler: warmup={warmup_steps}, total_steps={total_steps}, phase_lr={phase_lr:.6f}\n")
        
        # Track best metric for this phase
        phase_best_metric = float("inf")
        
        for phase_epoch in range(max_epochs):
            self.phase_epoch = phase_epoch
            self.epoch += 1  # Global epoch counter
            
            # Train one epoch on this phase's loader
            self.model.train()
            total_loss = 0.0
            total_lm_loss = 0.0
            total_lm_loss_causal = 0.0
            total_lm_loss_taskA = 0.0
            total_kl_loss = 0.0
            total_tokens = 0
            total_tokens_causal = 0
            total_tokens_taskA = 0
            
            pbar = tqdm(train_loader, desc=f"{phase_name} epoch {phase_epoch+1}/{max_epochs}")
            for step, batch in enumerate(pbar):
                metrics = self._train_step(batch, step == 0)
                
                total_loss += metrics['loss_sum']
                total_lm_loss += metrics['lm_loss_sum']
                total_lm_loss_causal += metrics['lm_loss_causal_sum']
                total_lm_loss_taskA += metrics['lm_loss_taskA_sum']
                total_kl_loss += metrics['kl_loss_sum']
                total_tokens += metrics['n_tokens']
                total_tokens_causal += metrics['n_tokens_causal']
                total_tokens_taskA += metrics['n_tokens_taskA']
                
                if self.global_step % self.config.log_interval == 0:
                    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
                    avg_lm_loss = total_lm_loss / total_tokens if total_tokens > 0 else 0.0
                    ppl = math.exp(avg_lm_loss) if avg_lm_loss < 10 else float("inf")
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    pbar.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "ppl": f"{ppl:.2f}",
                        "lr": f"{current_lr:.2e}",
                    })
            
            # Compute train metrics
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
            avg_lm_loss = total_lm_loss / total_tokens if total_tokens > 0 else 0.0
            avg_lm_loss_causal = total_lm_loss_causal / total_tokens_causal if total_tokens_causal > 0 else 0.0
            avg_lm_loss_taskA = total_lm_loss_taskA / total_tokens_taskA if total_tokens_taskA > 0 else 0.0
            perplexity = math.exp(avg_lm_loss) if avg_lm_loss < 10 else float("inf")
            
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"\n{phase_name} Epoch {phase_epoch+1}/{max_epochs} - Train:")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
            if total_tokens_causal > 0:
                ppl_causal = math.exp(avg_lm_loss_causal) if avg_lm_loss_causal < 10 else float("inf")
                print(f"  Causal: loss={avg_lm_loss_causal:.4f}, ppl={ppl_causal:.2f}, tokens={total_tokens_causal}")
            if total_tokens_taskA > 0:
                ppl_taskA = math.exp(avg_lm_loss_taskA) if avg_lm_loss_taskA < 10 else float("inf")
                print(f"  TaskA: loss={avg_lm_loss_taskA:.4f}, ppl={ppl_taskA:.2f}, tokens={total_tokens_taskA}")
            
            # Validate (always on causal val_loader)
            val_metrics = self.validate()
            print(f"\n{phase_name} Epoch {phase_epoch+1}/{max_epochs} - Validation:")
            if val_metrics['n_tokens_causal'] > 0:
                print(f"  Causal: loss={val_metrics['loss_causal']:.4f}, ppl={val_metrics['perplexity_causal']:.2f}")
                # Safe access for EOS collapse metrics with counts
                eos_rate_10 = val_metrics.get('eos_top1_rate_at_10')
                eos_count_10 = val_metrics.get('eos_eval_count_at_10', 0)
                if eos_rate_10 is not None:
                    print(f"  EOS collapse @10: {eos_rate_10:.4f} (n={eos_count_10})")
                eos_rate_1 = val_metrics.get('eos_top1_rate_at_1')
                eos_count_1 = val_metrics.get('eos_eval_count_at_1', 0)
                if eos_rate_1 is not None:
                    print(f"  EOS collapse @1: {eos_rate_1:.4f} (n={eos_count_1})")
            if val_metrics['n_tokens_taskA'] > 0:
                print(f"  TaskA: loss={val_metrics['loss_taskA']:.4f}, ppl={val_metrics['perplexity_taskA']:.2f}")
            
            # Determine early stopping metric
            if optimize_task == "taskA" and val_loader_for_early_stop is not None:
                # Phase 1: use taskA validation loss
                val_metrics_taskA = self._validate_on_loader(val_loader_for_early_stop)
                early_stop_metric = val_metrics_taskA['loss_taskA'] if val_metrics_taskA['n_tokens_taskA'] > 0 else val_metrics_taskA['loss']
                print(f"  Early stop metric (taskA val): {early_stop_metric:.4f}")
            elif optimize_task == "taskA":
                # Fallback: use train taskA loss if no taskA val loader
                early_stop_metric = avg_lm_loss_taskA if total_tokens_taskA > 0 else avg_lm_loss
                print(f"  Early stop metric (taskA train): {early_stop_metric:.4f}")
            else:
                # Phase 2: use causal validation loss
                early_stop_metric = val_metrics['loss_causal']
                print(f"  Early stop metric (causal val): {early_stop_metric:.4f}")
            
            # Check for improvement
            # CRITICAL: Use consistent best metric tracking with proper min_delta
            is_improvement = early_stop_metric < (phase_best_metric - early_stop_min_delta)
            
            if is_improvement:
                prev_best = phase_best_metric
                phase_best_metric = early_stop_metric
                self.early_stop_counter = 0
                print(f"  ✓ Improvement: {prev_best:.4f} -> {phase_best_metric:.4f} (Δ={prev_best - phase_best_metric:.4f})")
            else:
                self.early_stop_counter += 1
                print(f"  No improvement (best={phase_best_metric:.4f}, current={early_stop_metric:.4f}, delta={phase_best_metric - early_stop_metric:.4f})")
                print(f"  Early stop counter: {self.early_stop_counter}/{early_stop_patience}")
            
            # Update best checkpoint (only in phase 2 or if not restricted)
            if update_best_checkpoint:
                current_causal_loss = val_metrics['loss_causal']
                is_new_best = current_causal_loss < (self.best_val_loss_causal - early_stop_min_delta)
                if is_new_best:
                    prev_best_global = self.best_val_loss_causal
                    self.best_val_loss_causal = current_causal_loss
                    print(f"  ✓ New best causal validation loss: {prev_best_global:.4f} -> {self.best_val_loss_causal:.4f}")
                    if (phase_epoch + 1) % self.config.save_interval == 0:
                        self.save_checkpoint(f"checkpoint_epoch_{self.epoch}.pt", is_best=True)
                elif (phase_epoch + 1) % self.config.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_epoch_{self.epoch}.pt", is_best=False)
            else:
                # Phase 1: save but don't update best
                if (phase_epoch + 1) % self.config.save_interval == 0:
                    self.save_checkpoint(f"checkpoint_epoch_{self.epoch}.pt", is_best=False)
            
            # Cleanup old checkpoints
            self._cleanup_checkpoints()
            
            # Early stopping check
            if self.early_stop_counter >= early_stop_patience:
                print(f"\n  ⚠ Early stopping triggered for {phase_name}")
                self.should_stop_early = True
                break
            
            print("-" * 70)
        
        print("\n" + "="*70)
        if self.should_stop_early:
            print(f"{phase_name} stopped early after epoch {phase_epoch+1}/{max_epochs}")
        else:
            print(f"{phase_name} completed all {max_epochs} epochs")
        print(f"Best metric for this phase: {phase_best_metric:.4f}")
        print("="*70 + "\n")
    
    def _validate_on_loader(self, loader: DataLoader) -> Dict[str, float]:
        """Validate on a specific loader (helper for train_phase)."""
        self.model.eval()
        total_loss = 0.0
        total_lm_loss = 0.0
        total_lm_loss_causal = 0.0
        total_lm_loss_taskA = 0.0
        total_tokens = 0
        total_tokens_causal = 0
        total_tokens_taskA = 0
        
        with torch.no_grad():
            for batch in loader:
                encoder_input_ids = batch["encoder_input_ids"].to(self.device)
                decoder_input_ids = batch["decoder_input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                encoder_mask = batch["encoder_mask"].to(self.device)
                decoder_mask = batch["decoder_mask"].to(self.device)
                causal_mask = batch["causal_mask"].to(self.device)
                task_id = batch["task_id"].to(self.device)
                
                if self.config.use_amp:
                    with autocast():
                        logits, kl_div = self.model(
                            encoder_input_ids, decoder_input_ids,
                            encoder_mask, decoder_mask, causal_mask,
                        )
                else:
                    logits, kl_div = self.model(
                        encoder_input_ids, decoder_input_ids,
                        encoder_mask, decoder_mask, causal_mask,
                    )
                
                B, T_dec = labels.shape
                task_id_expanded = task_id.unsqueeze(1).expand(B, T_dec)
                causal_mask_tokens = (task_id_expanded == 0) & (labels != -100)
                taskA_mask_tokens = (task_id_expanded == 1) & (labels != -100)
                n_tokens_causal = causal_mask_tokens.sum().item()
                n_tokens_taskA = taskA_mask_tokens.sum().item()
                
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                
                if n_tokens_causal > 0:
                    labels_causal = labels_flat.clone()
                    labels_causal[~causal_mask_tokens.view(-1)] = -100
                    lm_loss_causal_sum = self.criterion(logits_flat, labels_causal)
                else:
                    lm_loss_causal_sum = torch.tensor(0.0, device=self.device)
                
                if n_tokens_taskA > 0:
                    labels_taskA = labels_flat.clone()
                    labels_taskA[~taskA_mask_tokens.view(-1)] = -100
                    lm_loss_taskA_sum = self.criterion(logits_flat, labels_taskA)
                else:
                    lm_loss_taskA_sum = torch.tensor(0.0, device=self.device)
                
                total_lm_loss_causal += lm_loss_causal_sum.item()
                total_lm_loss_taskA += lm_loss_taskA_sum.item()
                total_tokens_causal += n_tokens_causal
                total_tokens_taskA += n_tokens_taskA
        
        avg_loss_causal = total_lm_loss_causal / total_tokens_causal if total_tokens_causal > 0 else float('inf')
        avg_loss_taskA = total_lm_loss_taskA / total_tokens_taskA if total_tokens_taskA > 0 else float('inf')
        total_tokens = total_tokens_causal + total_tokens_taskA
        avg_loss = (total_lm_loss_causal + total_lm_loss_taskA) / total_tokens if total_tokens > 0 else float('inf')
        
        return {
            "loss": avg_loss,
            "loss_causal": avg_loss_causal,
            "loss_taskA": avg_loss_taskA,
            "n_tokens_causal": total_tokens_causal,
            "n_tokens_taskA": total_tokens_taskA,
        }
    
    def train(self) -> None:
        """Main training orchestration with two-phase training."""
        print("\n" + "="*70)
        print("Starting Two-Phase Training")
        print("="*70)
        print(f"Phase 1 enabled: {self.config.phase1_enabled}")
        if self.config.phase1_enabled:
            print(f"  Phase 1: TaskA warm-start, max {self.config.phase1_epochs} epochs")
        print(f"  Phase 2: Causal fine-tune, max {self.config.phase2_epochs} epochs")
        print(f"Best checkpoint metric: {self.config.best_metric}")
        print("="*70 + "\n")
        
        # Phase 1: TaskA warm-start (optional)
        if self.config.phase1_enabled and self.train_loader_taskA is not None:
            self.train_phase(
                phase_name="Phase 1 (TaskA warm-start)",
                train_loader=self.train_loader_taskA,
                val_loader=self.val_loader,  # Always validate on causal
                max_epochs=self.config.phase1_epochs,
                early_stop_patience=self.config.phase1_early_stop_patience,
                early_stop_min_delta=self.config.phase1_early_stop_min_delta,
                lr_multiplier=self.config.phase1_lr_multiplier,
                optimize_task="taskA",
                update_best_checkpoint=not self.config.save_best_on_phase2_only,
                val_loader_for_early_stop=self.val_loader_taskA,  # Use taskA val for early stopping
            )
        
        # Phase 2: Causal fine-tune (always run)
        if self.train_loader_causal is not None:
            self.train_phase(
                phase_name="Phase 2 (Causal fine-tune)",
                train_loader=self.train_loader_causal,
                val_loader=self.val_loader,
                max_epochs=self.config.phase2_epochs,
                early_stop_patience=self.config.phase2_early_stop_patience,
                early_stop_min_delta=self.config.phase2_early_stop_min_delta,
                lr_multiplier=self.config.phase2_lr_multiplier,
                optimize_task="causal",
                update_best_checkpoint=True,  # Always update best in phase 2
                val_loader_for_early_stop=None,  # Use causal val from main val_loader
            )
        else:
            print("\n[WARNING] Causal loader not available, skipping phase 2")
        
        print("\n" + "="*70)
        print("Two-Phase Training Complete!")
        print(f"Best validation loss (causal): {self.best_val_loss_causal:.4f}")
        print("="*70 + "\n")
    
    def _cleanup_checkpoints(self) -> None:
        """Keep only the last N checkpoints."""
        checkpoints = sorted(
            [f for f in self.out_dir.glob("checkpoint_epoch_*.pt")],
            key=lambda x: int(x.stem.split("_")[-1])
        )
        
        if len(checkpoints) > self.config.keep_last_n:
            for ckpt in checkpoints[:-self.config.keep_last_n]:
                try:
                    ckpt.unlink()
                    print(f"  Removed old checkpoint: {ckpt.name}")
                except:
                    pass


if __name__ == "__main__":
    print("=== Trainer Smoke Test ===\n")
    print("Trainer module loaded successfully")
    print("Run train.py to test full training pipeline")
