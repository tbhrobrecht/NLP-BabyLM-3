"""
Encoder-Decoder model with probabilistic encoder and n-gram priors.

Architecture:
- Encoder: Bidirectional Transformer (BERT-like)
- Probabilistic Encoder: Posterior head + n-gram prior integration
- Decoder: Causal Transformer (GPT-like) with cross-attention
- Optional gating for probabilistic comprehension
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from ngrams import NgramPrior


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        return x + self.pe[:, :x.size(1), :]


class ProbabilisticEncoder(nn.Module):
    """
    Probabilistic encoder head that combines posterior and n-gram prior.
    
    Computes:
    - Posterior q(z | h): distribution over vocab given encoder hidden states
    - Prior p(z): from n-gram statistics
    - Fused comprehension: weighted combination
    """
    
    def __init__(self, config: ModelConfig, ngram_prior: Optional[NgramPrior] = None):
        super().__init__()
        self.config = config
        self.ngram_prior = ngram_prior
        
        # Posterior head: hidden -> vocab distribution
        self.posterior_proj = nn.Linear(config.d_model, config.vocab_size)
        
        # Embedding layer for comprehension fusion
        self.vocab_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Fusion layer norm
        self.fusion_norm = nn.LayerNorm(config.d_model)
        
        # OPTIMIZATION: Register n-gram tensors as buffers
        # This ensures they move with .to(device) and are not recreated every forward pass
        if ngram_prior is not None and ngram_prior.is_fitted:
            self._register_ngram_buffers()
    
    def _register_ngram_buffers(self) -> None:
        """
        Register n-gram log-probability tensors as buffers.
        
        Buffers are automatically moved with .to(device) but not trained.
        NUMERICAL EQUIVALENCE: Same values, just stored as torch tensors.
        """
        # Unigram log-probs: (V,)
        unigram_logprobs = torch.from_numpy(self.ngram_prior.unigram_logprobs).float()
        self.register_buffer('unigram_logprobs', unigram_logprobs, persistent=False)
        
        # Bigram log-probs: (V, V)
        bigram_logprobs = torch.from_numpy(self.ngram_prior.bigram_logprobs_np).float()
        self.register_buffer('bigram_logprobs', bigram_logprobs, persistent=False)
    
    def forward(
        self,
        encoder_hidden: torch.Tensor,
        input_ids: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute probabilistic comprehension.
        
        Args:
            encoder_hidden: (B, T, D) encoder hidden states
            input_ids: (B, T) input token IDs (for n-gram prior context)
            mask: (B, T) attention mask
        
        Returns:
            - fused_hidden: (B, T, D) fused encoder memory
            - posterior_dist: (B, T, V) posterior distribution
            - kl_div: Optional KL divergence between posterior and prior
        """
        B, T, D = encoder_hidden.shape
        V = self.config.vocab_size
        
        # Compute posterior q(z | h)
        posterior_logits = self.posterior_proj(encoder_hidden)  # (B, T, V)
        posterior_dist = F.softmax(posterior_logits, dim=-1)  # (B, T, V)
        
        # Compute prior p(z) from n-grams
        if self.ngram_prior is not None and self.ngram_prior.is_fitted:
            # OPTIMIZATION: Use bigram prior with tensorized indexing
            # P(token_t | token_{t-1}) computed via tensor indexing
            # NUMERICAL EQUIVALENCE: Same smoothing, same probabilities
            
            # Shift input_ids to get prev tokens
            prev_ids = torch.cat([
                torch.full((B, 1), self.config.bos_id, dtype=torch.long, device=input_ids.device),
                input_ids[:, :-1]
            ], dim=1)  # (B, T)
            
            # TENSORIZED: Use registered buffers for fast indexing
            # Clamp to valid range
            prev_ids_clamped = torch.clamp(prev_ids, 0, V - 1)  # (B, T)
            
            # Index into bigram_logprobs buffer: (V, V)[prev_ids] -> (B, T, V)
            log_prior = self.bigram_logprobs[prev_ids_clamped]  # (B, T, V)
            
            # Apply softmax to get probabilities (same as before)
            prior_dist = F.softmax(log_prior, dim=-1)  # (B, T, V)
            
            # Combine posterior and prior (log-space mixing - numerically equivalent)
            alpha = self.config.prior_alpha
            log_posterior = torch.log(posterior_dist + 1e-10)
            log_prior_stable = torch.log(prior_dist + 1e-10)
            log_combined = (1 - alpha) * log_posterior + alpha * log_prior_stable
            combined_dist = F.softmax(log_combined, dim=-1)  # (B, T, V)
            
            # Compute KL divergence for regularization (same as before)
            kl_div = F.kl_div(
                log_posterior,
                prior_dist,
                reduction="none"
            ).sum(dim=-1)  # (B, T)
            kl_div = (kl_div * mask).sum() / mask.sum()  # Masked mean
        else:
            # No prior available, use posterior only
            combined_dist = posterior_dist
            kl_div = None
        
        # Compute comprehension embedding: E[embedding under combined dist]
        # OPTIMIZATION: Flatten batch*time for efficient matmul, then reshape
        # NUMERICAL EQUIVALENCE: Same as sum_v combined_dist[:,:,v] * vocab_embedding[v]
        # comp = combined_dist @ vocab_embedding.weight
        
        # Ensure tensors are contiguous and dtype matches
        combined_dist = combined_dist.contiguous()  # (B, T, V)
        
        # Flatten: (B, T, V) -> (B*T, V)
        BT = B * T
        combined_dist_flat = combined_dist.view(BT, V)
        
        # Efficient matmul: (B*T, V) @ (V, D) -> (B*T, D)
        comp_embedding_flat = torch.matmul(
            combined_dist_flat.to(self.vocab_embedding.weight.dtype),
            self.vocab_embedding.weight
        )
        
        # Reshape back: (B*T, D) -> (B, T, D)
        comp_embedding = comp_embedding_flat.view(B, T, D)
        
        # Fuse with encoder hidden states (same as before)
        beta = self.config.posterior_beta
        fused_hidden = self.fusion_norm(encoder_hidden + beta * comp_embedding)
        
        return fused_hidden, posterior_dist, kl_div


class TransformerEncoder(nn.Module):
    """Bidirectional Transformer encoder."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_encoder_layers)
    
    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: (B, T) token IDs
            mask: (B, T) attention mask (1 for real tokens, 0 for padding)
        
        Returns:
            (B, T, D) encoder hidden states
        """
        # Embed
        x = self.token_embedding(input_ids)  # (B, T, D)
        x = self.pos_encoding(x)
        
        # Transformer expects mask in different format: True for positions to IGNORE
        # Our mask: 1 for real, 0 for padding -> invert for Transformer
        src_key_padding_mask = (mask == 0)  # (B, T)
        
        # Encode
        hidden = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        return hidden


class TransformerDecoder(nn.Module):
    """Causal Transformer decoder with cross-attention."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.pos_encoding = PositionalEncoding(config.d_model, config.max_seq_len)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_decoder_layers)
        
        # Optional gating for probabilistic comprehension
        if config.use_gating:
            self.gate = nn.Sequential(
                nn.Linear(config.d_model * 2, config.d_model),
                nn.Sigmoid()
            )
        else:
            self.gate = None
        
        # Output projection
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        memory_key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: (B, T_dec) decoder input token IDs
            encoder_memory: (B, T_enc, D) encoder memory
            tgt_mask: (T_dec, T_dec) causal mask (lower triangular)
            tgt_key_padding_mask: (B, T_dec) decoder padding mask
            memory_key_padding_mask: (B, T_enc) encoder padding mask
        
        Returns:
            (B, T_dec, V) logits over vocabulary
        """
        # Embed
        x = self.token_embedding(input_ids)  # (B, T_dec, D)
        x = self.pos_encoding(x)
        
        # Decode with cross-attention
        hidden = self.transformer_decoder(
            tgt=x,
            memory=encoder_memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, T_dec, D)
        
        # Optional gating
        if self.gate is not None:
            # Gate context from encoder
            # Compute attention-weighted context (simple mean pooling over encoder memory)
            # More sophisticated: use attention scores, but for simplicity, use mean
            context = encoder_memory.mean(dim=1, keepdim=True).expand_as(hidden)  # (B, T_dec, D)
            gate_input = torch.cat([hidden, context], dim=-1)  # (B, T_dec, 2D)
            gate_value = self.gate(gate_input)  # (B, T_dec, D)
            hidden = gate_value * hidden + (1 - gate_value) * context
        
        # Project to vocabulary
        logits = self.output_proj(hidden)  # (B, T_dec, V)
        
        return logits


class EncoderDecoderModel(nn.Module):
    """
    Full encoder-decoder model with probabilistic encoder.
    """
    
    def __init__(self, config: ModelConfig, ngram_prior: Optional[NgramPrior] = None):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = TransformerEncoder(config)
        
        # Probabilistic encoder head
        if config.use_probabilistic_encoder:
            self.prob_encoder = ProbabilisticEncoder(config, ngram_prior)
        else:
            self.prob_encoder = None
        
        # Decoder
        self.decoder = TransformerDecoder(config)
        
        # OPTIMIZATION: Pre-create causal mask buffer (static, reusable)
        # This avoids recreating it on every forward pass
        max_len = config.max_seq_len
        causal_mask = torch.tril(torch.ones(max_len, max_len))
        self.register_buffer('causal_mask_cache', causal_mask, persistent=False)
    
    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        encoder_mask: torch.Tensor,
        decoder_mask: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            encoder_input_ids: (B, T_enc)
            decoder_input_ids: (B, T_dec)
            encoder_mask: (B, T_enc)
            decoder_mask: (B, T_dec)
            causal_mask: (1, T_dec, T_dec) or (B, T_dec, T_dec)
        
        Returns:
            - logits: (B, T_dec, V)
            - kl_div: Optional KL divergence (scalar)
        """
        # Encode
        encoder_hidden = self.encoder(encoder_input_ids, encoder_mask)  # (B, T_enc, D)
        
        # Probabilistic encoder
        kl_div = None
        if self.prob_encoder is not None:
            encoder_memory, posterior_dist, kl_div = self.prob_encoder(
                encoder_hidden,
                encoder_input_ids,
                encoder_mask
            )
        else:
            encoder_memory = encoder_hidden
        
        # Prepare masks for decoder
        tgt_key_padding_mask = (decoder_mask == 0)  # (B, T_dec)
        memory_key_padding_mask = (encoder_mask == 0)  # (B, T_enc)
        
        # CRITICAL: Enforce causal mask (decoder cannot peek at future tokens)
        # PyTorch Transformer expects tgt_mask with shape (T_dec, T_dec) where:
        # - True = position is MASKED (cannot attend)
        # - False = position is ALLOWED (can attend)
        # For causal LM, we need upper triangular (mask future positions):
        # [[False, True,  True,  ...],  # pos 0 can only see pos 0
        #  [False, False, True,  ...],  # pos 1 can see 0,1
        #  [False, False, False, ...],  # pos 2 can see 0,1,2
        #  ...]
        T_dec = decoder_input_ids.size(1)
        device = decoder_input_ids.device
        # Create upper triangular mask (diagonal=1 means exclude diagonal, keeping only strict future)
        tgt_mask = torch.triu(torch.ones(T_dec, T_dec, device=device, dtype=torch.bool), diagonal=1)
        # tgt_mask[i,j] = True if j > i (future position, should be masked)
        
        # Decode
        logits = self.decoder(
            input_ids=decoder_input_ids,
            encoder_memory=encoder_memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (B, T_dec, V)
        
        return logits, kl_div
    
    def generate(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_mask: torch.Tensor,
        max_len: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        bos_id: int = 2,
        eos_id: int = 3,
        min_len: int = 0,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        
        Args:
            encoder_input_ids: (B, T_enc) encoder context
            encoder_mask: (B, T_enc)
            max_len: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling
            bos_id: BOS token ID
            eos_id: EOS token ID
            min_len: Minimum generation length (blocks EOS until reached)
        
        Returns:
            (B, max_len) generated token IDs
        """
        self.eval()
        device = encoder_input_ids.device
        B = encoder_input_ids.size(0)
        
        # Encode once
        with torch.no_grad():
            encoder_hidden = self.encoder(encoder_input_ids, encoder_mask)
            
            if self.prob_encoder is not None:
                encoder_memory, _, _ = self.prob_encoder(
                    encoder_hidden,
                    encoder_input_ids,
                    encoder_mask
                )
            else:
                encoder_memory = encoder_hidden
        
        # Initialize decoder input with BOS
        decoder_input_ids = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        
        # Generate autoregressively
        for _ in range(max_len):
            T_dec = decoder_input_ids.size(1)
            
            # CRITICAL: Causal mask - upper triangular (mask future tokens)
            tgt_mask = torch.triu(torch.ones(T_dec, T_dec, device=device, dtype=torch.bool), diagonal=1)
            
            # Decoder masks
            decoder_mask = torch.ones(B, T_dec, dtype=torch.long, device=device)
            tgt_key_padding_mask = (decoder_mask == 0)
            memory_key_padding_mask = (encoder_mask == 0)
            
            # Forward pass
            with torch.no_grad():
                logits = self.decoder(
                    input_ids=decoder_input_ids,
                    encoder_memory=encoder_memory,
                    tgt_mask=tgt_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )  # (B, T_dec, V)
            
            # Get logits for last position
            next_token_logits = logits[:, -1, :] / temperature  # (B, V)
            
            # Block EOS token if current length < min_len (prevents premature stopping)
            current_len = decoder_input_ids.size(1) - 1  # Subtract BOS
            if current_len < min_len:
                next_token_logits[:, eos_id] = float("-inf")
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("inf")
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float("inf")
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append to decoder input
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
            
            # Check if all sequences have generated EOS
            if (next_token == eos_id).all():
                break
        
        return decoder_input_ids


if __name__ == "__main__":
    print("=== Model Smoke Test ===\n")
    
    # Create config
    config = ModelConfig(
        vocab_size=1000,
        d_model=256,
        n_heads=4,
        d_ff=1024,
        n_encoder_layers=2,
        n_decoder_layers=2,
        max_seq_len=128,
    )
    
    # Create model (without n-gram prior for now)
    model = EncoderDecoderModel(config, ngram_prior=None)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    B, T_enc, T_dec = 4, 20, 15
    encoder_input_ids = torch.randint(0, config.vocab_size, (B, T_enc))
    decoder_input_ids = torch.randint(0, config.vocab_size, (B, T_dec))
    encoder_mask = torch.ones(B, T_enc)
    decoder_mask = torch.ones(B, T_dec)
    causal_mask = torch.tril(torch.ones(T_dec, T_dec)).unsqueeze(0)
    
    print("\nForward pass test:")
    logits, kl_div = model(
        encoder_input_ids,
        decoder_input_ids,
        encoder_mask,
        decoder_mask,
        causal_mask,
    )
    print(f"  Output logits shape: {logits.shape}")
    print(f"  Expected: ({B}, {T_dec}, {config.vocab_size})")
    print(f"  KL divergence: {kl_div}")
    
    # Test generation
    print("\nGeneration test:")
    encoder_input_ids = torch.randint(0, config.vocab_size, (2, 10))
    encoder_mask = torch.ones(2, 10)
    
    generated = model.generate(
        encoder_input_ids,
        encoder_mask,
        max_len=20,
        temperature=1.0,
        bos_id=config.bos_id,
        eos_id=config.eos_id,
    )
    print(f"  Generated shape: {generated.shape}")
    print(f"  Sample generation: {generated[0, :10].tolist()}")
    
    print("\n[OK] Model module smoke test passed")
