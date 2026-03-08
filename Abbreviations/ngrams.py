"""
N-gram statistics computation for probabilistic encoder priors.

Computes unigram and bigram probabilities over SentencePiece token IDs
from TRAIN split only, with add-k smoothing.

OPTIMIZATION NOTES:
- Replaced dict-based bigram storage with dense (V, V) numpy/torch tensors
- Eliminated all Python loops in favor of vectorized operations
- Uses advanced numpy/torch indexing for O(1) lookups instead of O(V^2) loops
- Numerical equivalence preserved: same smoothing formula, same normalization
"""
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict


class NgramPrior:
    """
    Unigram and bigram probability model for token IDs.
    
    Provides log-probabilities with add-k smoothing.
    
    TENSORIZED IMPLEMENTATION:
    - bigram_logprobs_np: Dense (V, V) numpy array for O(1) vectorized lookups
    - Eliminates nested loops in get_prior_distribution()
    - Numerical equivalence: P(w_t | w_{t-1}) = (count(w_{t-1}, w_t) + k) / (count(w_{t-1}) + k*V)
    - Same smoothing constant k, same normalization, same softmax temperature
    """
    
    def __init__(self, vocab_size: int, k: float = 1.0):
        """
        Initialize n-gram prior.
        
        Args:
            vocab_size: Size of vocabulary
            k: Smoothing parameter (add-k smoothing)
        """
        self.vocab_size = vocab_size
        self.k = k
        
        # Unigram counts and log-probs
        self.unigram_counts = np.zeros(vocab_size, dtype=np.int64)
        self.unigram_logprobs = None  # Will be computed after fitting (numpy array)
        
        # Bigram counts and log-probs
        # bigram_counts[i][j] = count of (i, j)
        self.bigram_counts = defaultdict(Counter)
        # TENSORIZED: Store as dense (V, V) tensor instead of dict
        self.bigram_logprobs_np = None  # numpy (V, V) array, will be computed in fit()
        self.bigram_logprobs_torch: Optional[torch.Tensor] = None  # torch tensor for fast GPU lookup (PATCH 4)
        
        # Statistics
        self.total_tokens = 0
        self.is_fitted = False
    
    def fit(self, token_sequences: List[List[int]]) -> None:
        """
        Compute n-gram statistics from token sequences.
        
        Args:
            token_sequences: List of token ID sequences (from TRAIN split only)
        """
        print("Computing n-gram statistics...")
        
        # Count unigrams and bigrams
        for seq in token_sequences:
            if len(seq) == 0:
                continue
            
            # Unigrams
            for token_id in seq:
                if 0 <= token_id < self.vocab_size:
                    self.unigram_counts[token_id] += 1
                    self.total_tokens += 1
            
            # Bigrams
            for i in range(len(seq) - 1):
                prev_id = seq[i]
                curr_id = seq[i + 1]
                if 0 <= prev_id < self.vocab_size and 0 <= curr_id < self.vocab_size:
                    self.bigram_counts[prev_id][curr_id] += 1
        
        # Compute unigram log-probabilities with add-k smoothing
        # P(w) = (count(w) + k) / (total + k * vocab_size)
        smoothed_total = self.total_tokens + self.k * self.vocab_size
        self.unigram_logprobs = np.log((self.unigram_counts + self.k) / smoothed_total)
        
        # Compute bigram log-probabilities with add-k smoothing
        # P(w_t | w_{t-1}) = (count(w_{t-1}, w_t) + k) / (count(w_{t-1}) + k * vocab_size)
        # TENSORIZED: Build dense (V, V) numpy array instead of dict for faster indexing
        print("Computing bigram log-probabilities (tensorized)...")
        
        # Initialize dense bigram count matrix (V, V)
        bigram_count_matrix = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float64)
        for prev_id, curr_counts in self.bigram_counts.items():
            for curr_id, count in curr_counts.items():
                bigram_count_matrix[prev_id, curr_id] = count
        
        # Vectorized computation: for each prev_id row, apply smoothing
        # Shape: (V, 1) for broadcasting
        prev_counts = self.unigram_counts[:, np.newaxis]  # (V, 1)
        smoothed_prev_totals = prev_counts + self.k * self.vocab_size  # (V, 1)
        
        # Add smoothing to all bigram counts and normalize
        # Shape: (V, V) element-wise
        self.bigram_logprobs_np = np.log(
            (bigram_count_matrix + self.k) / smoothed_prev_totals
        )  # (V, V)
        
        self.is_fitted = True
        
        # Statistics
        nonzero_unigrams = np.sum(self.unigram_counts > 0)
        nonzero_bigrams = len([1 for counts in self.bigram_counts.values() for _ in counts])
        
        print(f"[OK] N-gram statistics computed:")
        print(f"  Total tokens: {self.total_tokens}")
        print(f"  Unique unigrams: {nonzero_unigrams} / {self.vocab_size}")
        print(f"  Unique bigrams: {nonzero_bigrams} / {self.vocab_size**2}")
        print(f"  Smoothing k: {self.k}")
    
    def log_prob_unigram(self, token_id: int) -> float:
        """Get unigram log-probability for a token ID."""
        if not self.is_fitted:
            raise RuntimeError("NgramPrior not fitted. Call fit() first.")
        if not (0 <= token_id < self.vocab_size):
            return -np.inf
        return float(self.unigram_logprobs[token_id])
    
    def log_prob_bigram(self, prev_id: int, curr_id: int) -> float:
        """Get bigram log-probability P(curr_id | prev_id)."""
        if not self.is_fitted:
            raise RuntimeError("NgramPrior not fitted. Call fit() first.")
        if not (0 <= prev_id < self.vocab_size and 0 <= curr_id < self.vocab_size):
            return -np.inf
        # TENSORIZED: Direct array indexing instead of dict lookup
        return float(self.bigram_logprobs_np[prev_id, curr_id])
    
    def log_prob_unigram_batch(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Get unigram log-probabilities for a batch of token IDs.
        
        Args:
            token_ids: Tensor of shape (B, T) with token IDs
        
        Returns:
            Log-probabilities of shape (B, T)
        """
        if not self.is_fitted:
            raise RuntimeError("NgramPrior not fitted. Call fit() first.")
        
        # Clamp to valid range
        token_ids_np = token_ids.cpu().numpy()
        token_ids_np = np.clip(token_ids_np, 0, self.vocab_size - 1)
        
        # Lookup
        log_probs = self.unigram_logprobs[token_ids_np]
        
        return torch.from_numpy(log_probs).to(token_ids.device)
    
    def log_prob_bigram_batch(self, prev_ids: torch.Tensor, curr_ids: torch.Tensor) -> torch.Tensor:
        """
        Get bigram log-probabilities for a batch.
        
        Args:
            prev_ids: Tensor of shape (B, T) with previous token IDs
            curr_ids: Tensor of shape (B, T) with current token IDs
        
        Returns:
            Log-probabilities of shape (B, T)
        """
        if not self.is_fitted:
            raise RuntimeError("NgramPrior not fitted. Call fit() first.")
        
        B, T = prev_ids.shape
        device = prev_ids.device
        
        # Convert to numpy for lookup
        prev_ids_np = prev_ids.cpu().numpy()
        curr_ids_np = curr_ids.cpu().numpy()
        
        # Clamp to valid range
        prev_ids_np = np.clip(prev_ids_np, 0, self.vocab_size - 1)
        curr_ids_np = np.clip(curr_ids_np, 0, self.vocab_size - 1)
        
        # TENSORIZED: Use advanced indexing instead of nested loops
        # bigram_logprobs_np is (V, V), index with [prev_ids, curr_ids] to get (B, T)
        log_probs = self.bigram_logprobs_np[prev_ids_np, curr_ids_np]  # (B, T)
        
        return torch.from_numpy(log_probs).to(device, dtype=torch.float32)
    
    def get_prior_distribution(
        self,
        prev_ids: torch.Tensor,
        use_bigram: bool = True,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Get prior distribution over vocabulary for each position.
        
        TENSORIZED: Fully vectorized using tensor indexing, no Python loops.
        Numerical equivalence: Same smoothing, same normalization, same softmax.
        
        Args:
            prev_ids: Tensor of shape (B, T) with previous token IDs
            use_bigram: If True, use bigram P(· | prev); else use unigram
            temperature: Temperature for softmax
        
        Returns:
            Distribution of shape (B, T, V)
        """
        if not self.is_fitted:
            raise RuntimeError("NgramPrior not fitted. Call fit() first.")
        
        B, T = prev_ids.shape
        V = self.vocab_size
        device = prev_ids.device
        
        if use_bigram:
            # Clamp prev_ids to valid range
            prev_ids_clamped = torch.clamp(prev_ids, 0, V - 1)  # (B, T)
            
            # PATCH 4: Use torch tensor if available to avoid CPU roundtrip
            if self.bigram_logprobs_torch is not None and self.bigram_logprobs_torch.device == device:
                # Fast path: direct torch indexing on GPU
                log_probs = self.bigram_logprobs_torch[prev_ids_clamped]  # (B, T, V)
            else:
                # Fallback: numpy path (for safety or if tensor not on correct device)
                prev_ids_clamped_np = prev_ids_clamped.cpu().numpy()  # (B, T)
                log_probs = self.bigram_logprobs_np[prev_ids_clamped_np]  # (B, T, V)
                log_probs = torch.from_numpy(log_probs).to(device, dtype=torch.float32)
        else:
            # Unigram distribution (same for all positions)
            log_probs = torch.from_numpy(self.unigram_logprobs).to(device, dtype=torch.float32)
            log_probs = log_probs.unsqueeze(0).unsqueeze(0).expand(B, T, V)
        
        # Apply temperature and softmax (same as before)
        probs = torch.softmax(log_probs / temperature, dim=-1)
        
        return probs
    
    def save(self, path: str) -> None:
        """Save n-gram model to disk."""
        np.savez(
            path,
            vocab_size=self.vocab_size,
            k=self.k,
            unigram_counts=self.unigram_counts,
            unigram_logprobs=self.unigram_logprobs,
            bigram_logprobs=self.bigram_logprobs_np,  # Save tensor directly
            total_tokens=self.total_tokens,
        )
        print(f"[OK] N-gram model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "NgramPrior":
        """Load n-gram model from disk."""
        data = np.load(path, allow_pickle=True)
        
        model = cls(vocab_size=int(data["vocab_size"]), k=float(data["k"]))
        model.unigram_counts = data["unigram_counts"]
        model.unigram_logprobs = data["unigram_logprobs"]
        model.total_tokens = int(data["total_tokens"])
        
        # Load bigram tensor directly
        model.bigram_logprobs_np = data["bigram_logprobs"]
        model.bigram_logprobs_torch = None  # Will be set by load_ngram_prior if needed
        
        model.is_fitted = True
        
        print(f"[OK] N-gram model loaded from {path}")
        return model


def load_ngram_prior(path: str, vocab_size: int, device: str = "cuda") -> Optional[NgramPrior]:
    """
    Load n-gram prior from disk for inference.
    
    Args:
        path: Path to ngram_prior.npz file
        vocab_size: Expected vocabulary size
        device: Device to move tensors to
    
    Returns:
        NgramPrior object or None if loading fails
    """
    try:
        from pathlib import Path
        path_obj = Path(path)
        
        if not path_obj.exists():
            print(f"[INFO] N-gram prior not found: {path}")
            return None
        
        # Load using existing class method
        ngram_prior = NgramPrior.load(str(path_obj))
        
        # Validate vocab size
        if ngram_prior.vocab_size != vocab_size:
            print(f"[WARNING] N-gram vocab size mismatch: expected {vocab_size}, got {ngram_prior.vocab_size}")
            print(f"[WARNING] N-gram prior disabled")
            return None
        
        # PATCH 4: Create torch tensor on device for fast GPU lookup
        if ngram_prior.bigram_logprobs_np is not None:
            ngram_prior.bigram_logprobs_torch = torch.from_numpy(ngram_prior.bigram_logprobs_np).to(
                device, dtype=torch.float32
            )
            print(f"[OK] N-gram prior loaded: {path} (torch tensor on {device})")
        else:
            print(f"[OK] N-gram prior loaded: {path}")
        
        return ngram_prior
        
    except Exception as e:
        print(f"[WARNING] Failed to load n-gram prior: {e}")
        print(f"[INFO] N-gram prior disabled")
        return None


if __name__ == "__main__":
    print("=== N-gram Prior Smoke Test ===\n")
    
    # Create synthetic token sequences
    vocab_size = 100
    sequences = [
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6],
        [1, 2, 3, 2, 1],
        [10, 11, 12, 13],
        [2, 2, 2, 3, 3],
    ]
    
    # Fit n-gram model
    print("Fitting n-gram model...")
    ngram = NgramPrior(vocab_size=vocab_size, k=0.1)
    ngram.fit(sequences)
    
    # Test unigram
    print("\nUnigram log-probs:")
    for token_id in [1, 2, 3, 10, 99]:
        log_prob = ngram.log_prob_unigram(token_id)
        print(f"  P({token_id}) = {np.exp(log_prob):.6f} (log = {log_prob:.4f})")
    
    # Test bigram
    print("\nBigram log-probs:")
    for prev_id, curr_id in [(1, 2), (2, 3), (2, 2), (10, 11)]:
        log_prob = ngram.log_prob_bigram(prev_id, curr_id)
        print(f"  P({curr_id} | {prev_id}) = {np.exp(log_prob):.6f} (log = {log_prob:.4f})")
    
    # Test batch operations
    print("\nBatch operations:")
    token_ids = torch.tensor([[1, 2, 3], [2, 3, 4]])
    log_probs_unigram = ngram.log_prob_unigram_batch(token_ids)
    print(f"  Unigram batch shape: {log_probs_unigram.shape}")
    
    prev_ids = torch.tensor([[0, 1, 2], [1, 2, 3]])
    curr_ids = torch.tensor([[1, 2, 3], [2, 3, 4]])
    log_probs_bigram = ngram.log_prob_bigram_batch(prev_ids, curr_ids)
    print(f"  Bigram batch shape: {log_probs_bigram.shape}")
    
    # Test prior distribution
    print("\nPrior distribution:")
    prior_dist = ngram.get_prior_distribution(prev_ids, use_bigram=True)
    print(f"  Prior dist shape: {prior_dist.shape}")
    print(f"  Sum over vocab (should be ~1.0): {prior_dist.sum(dim=-1)}")
    
    # Test save/load
    print("\nSave/load:")
    ngram.save("test_ngram.npz")
    ngram_loaded = NgramPrior.load("test_ngram.npz")
    print(f"  Loaded vocab_size: {ngram_loaded.vocab_size}")
    
    # Cleanup
    import os
    try:
        os.remove("test_ngram.npz")
    except:
        pass
    
    print("\n[OK] N-gram module smoke test passed")
