"""
BREWA Utilities
===============

Helper functions for Bit-Optimal REWA Attention:
- Hadamard transform (Fast Walsh-Hadamard)
- Hamming similarity (popcount-based)
- Capacity calculations
- Bit quantization
- Tropical semiring operations
"""

import torch
import torch.nn.functional as F
import math
from typing import Tuple, Optional


# ============================================================
# 1. Fast Walsh-Hadamard Transform
# ============================================================

def hadamard_transform(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Fast Walsh-Hadamard Transform (FWHT).
    
    Complexity: O(n log n) where n = x.shape[-1]
    
    Args:
        x: Input tensor [..., n] where n must be power of 2
        normalize: If True, divide by sqrt(n)
    
    Returns:
        Transformed tensor [..., n]
    """
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"Size must be power of 2, got {n}"
    
    h = x.clone()
    log_n = int(math.log2(n))
    
    for i in range(log_n):
        stride = 2 ** (i + 1)
        half_stride = stride // 2
        
        for j in range(0, n, stride):
            for k in range(half_stride):
                idx1 = j + k
                idx2 = j + k + half_stride
                
                a = h[..., idx1]
                b = h[..., idx2]
                
                h[..., idx1] = a + b
                h[..., idx2] = a - b
    
    if normalize:
        h = h / math.sqrt(n)
    
    return h


def hadamard_transform_batched(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Batched Hadamard transform for efficiency.
    
    Args:
        x: [B, N, D] where D is power of 2
    
    Returns:
        [B, N, D] transformed
    """
    return hadamard_transform(x, normalize=normalize)


# ============================================================
# 2. Bit Quantization
# ============================================================

def bit_quantize(x: torch.Tensor, m_bits: int = 32, method: str = 'sign') -> torch.Tensor:
    """
    Quantize continuous values to m-bit representation.
    
    Args:
        x: Input tensor [..., D]
        m_bits: Number of bits (must be <= D)
        method: 'sign' (±1 → 0/1) or 'threshold' (median-based)
    
    Returns:
        Binary tensor [..., m_bits] with values in {0, 1}
    """
    if method == 'sign':
        # Use sign: positive → 1, negative → 0
        bits = (x[..., :m_bits] > 0).float()
    elif method == 'threshold':
        # Use median threshold
        threshold = x[..., :m_bits].median(dim=-1, keepdim=True)[0]
        bits = (x[..., :m_bits] > threshold).float()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return bits


def bit_dequantize(bits: torch.Tensor) -> torch.Tensor:
    """
    Convert bits back to ±1 representation.
    
    Args:
        bits: [..., m] with values in {0, 1}
    
    Returns:
        [..., m] with values in {-1, +1}
    """
    return 2.0 * bits - 1.0


# ============================================================
# 3. Hamming Similarity
# ============================================================

def hamming_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute Hamming similarity (number of matching bits).
    
    For binary tensors x, y ∈ {0,1}^m:
        sim(x, y) = m - hamming_distance(x, y)
                  = number of matching bits
    
    Args:
        x: [B, N, m] binary tensor
        y: [B, M, m] binary tensor
    
    Returns:
        [B, N, M] similarity scores (higher = more similar)
    """
    # XOR gives 1 where bits differ, 0 where they match
    # We want matches, so use XNOR (1 - XOR)
    x_expanded = x.unsqueeze(2)  # [B, N, 1, m]
    y_expanded = y.unsqueeze(1)  # [B, 1, M, m]
    
    # Count matching bits
    matches = (x_expanded == y_expanded).float()  # [B, N, M, m]
    similarity = matches.sum(dim=-1)  # [B, N, M]
    
    return similarity


def hamming_similarity_efficient(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Efficient Hamming similarity using matrix operations.
    
    For x, y ∈ {0,1}^m:
        matches = m - |x - y|₁ = m - 2·|x ⊕ y|₁
    
    Since x, y ∈ {0,1}: x·y counts 1-1 matches
    We can compute: matches = x·y + (1-x)·(1-y) = 2·x·y - x - y + m
    
    Simplified: matches = m - x - y + 2·x·y
    """
    m = x.shape[-1]
    
    # x: [B, N, m], y: [B, M, m]
    # Compute x·y^T: [B, N, M]
    xy = torch.bmm(x, y.transpose(1, 2))  # [B, N, M]
    
    # Sum of x and y
    x_sum = x.sum(dim=-1, keepdim=True)  # [B, N, 1]
    y_sum = y.sum(dim=-1, keepdim=True)  # [B, M, 1]
    
    # matches = m - x_sum - y_sum^T + 2·xy
    similarity = m - x_sum - y_sum.transpose(1, 2) + 2 * xy
    
    return similarity


# ============================================================
# 4. Capacity Calculations
# ============================================================

def channel_capacity(m_bits: int, noise_std: float = 0.1) -> float:
    """
    Calculate channel capacity for REWA encoding.
    
    For binary symmetric channel with error probability p:
        C = 1 - H(p)
    
    For Gaussian noise σ²:
        C ≈ log₂(m) bits
    
    Args:
        m_bits: Number of bits in encoding
        noise_std: Standard deviation of noise
    
    Returns:
        Capacity in bits
    """
    # Simplified: C = log₂(m) for low noise
    return math.log2(m_bits)


def max_context_length(d_model: int, m_bits: int = 32) -> int:
    """
    Calculate maximum context length from Theorem 6.1.
    
    n_max ≈ exp(√d)
    
    Args:
        d_model: Model dimension
        m_bits: Bits per encoding
    
    Returns:
        Maximum context length before capacity degradation
    """
    return int(math.exp(math.sqrt(d_model)))


def required_dimension(context_length: int) -> int:
    """
    Calculate required dimension for target context length.
    
    From scaling law: d ≥ (log N)²
    
    Args:
        context_length: Target context length N
    
    Returns:
        Minimum required dimension d
    """
    log_n = math.log2(context_length)
    return int(math.ceil(log_n ** 2))


# ============================================================
# 5. Tropical Semiring Operations
# ============================================================

def tropical_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Tropical matrix multiplication (min-plus semiring).
    
    (A ⊙ B)_ij = min_k (A_ik + B_kj)
    
    This is equivalent to shortest path computation.
    
    Args:
        A: [B, N, D] tensor
        B: [B, D, M] tensor (note: already transposed)
    
    Returns:
        [B, N, M] tropical product
    """
    # A: [B, N, D], B: [B, D, M]
    # We need to compute min_d (A[:, i, d] + B[:, d, j]) for all i, j
    # Expand: A: [B, N, D, 1], B: [B, D, 1, M] -> then transpose B to [B, 1, D, M]
    A_exp = A.unsqueeze(3)  # [B, N, D, 1]
    B_exp = B.unsqueeze(1)  # [B, 1, D, M]
    
    # Sum in tropical = regular sum
    tropical_sum = A_exp + B_exp  # [B, N, D, M]
    
    # Product in tropical = min over D dimension
    tropical_prod = tropical_sum.min(dim=2)[0]  # [B, N, M]
    
    return tropical_prod


def tropical_attention_weights(Q: torch.Tensor, K: torch.Tensor, 
                               temperature: float = 1.0) -> torch.Tensor:
    """
    Compute attention weights using tropical semiring.
    
    Instead of softmax(Q·K^T), we use:
        min-softmax(tropical_matmul(Q, K^T))
    
    Args:
        Q: [B, N, D] queries
        K: [B, M, D] keys
        temperature: Scaling factor
    
    Returns:
        [B, N, M] attention weights
    """
    # Tropical distance (min-plus)
    scores = tropical_matmul(Q, K.transpose(1, 2))  # [B, N, M]
    
    # Convert to probabilities (negative distance → higher probability)
    scores = -scores / temperature
    
    # Use softmax (or could use tropical softmax)
    weights = F.softmax(scores, dim=-1)
    
    return weights


# ============================================================
# 6. Noise Injection
# ============================================================

def add_diagonal_noise(x: torch.Tensor, noise_std: float = 0.1) -> torch.Tensor:
    """
    Add diagonal Gaussian noise to witness encoding.
    
    This is the key difference from random projections:
    - Random projection: Dense random matrix
    - REWA: Hadamard (deterministic) + diagonal noise
    
    Args:
        x: Input tensor [..., D]
        noise_std: Standard deviation of noise
    
    Returns:
        Noisy tensor [..., D]
    """
    noise = torch.randn_like(x) * noise_std
    return x + noise


# ============================================================
# 7. Information-Theoretic Metrics
# ============================================================

def mutual_information_estimate(x: torch.Tensor, y: torch.Tensor, 
                                bins: int = 50) -> float:
    """
    Estimate mutual information I(X; Y) using histograms.
    
    This measures how much information witness similarity
    carries about true similarity.
    
    Args:
        x: [N] witness similarities
        y: [N] true similarities
        bins: Number of histogram bins
    
    Returns:
        Estimated mutual information in bits
    """
    # Convert to numpy for histogram
    x_np = x.detach().cpu().numpy().flatten()
    y_np = y.detach().cpu().numpy().flatten()
    
    # 2D histogram
    hist_2d, _, _ = torch.histogram2d(
        torch.from_numpy(x_np), 
        torch.from_numpy(y_np), 
        bins=bins
    )
    
    # Normalize to probabilities
    p_xy = hist_2d / hist_2d.sum()
    p_x = p_xy.sum(dim=1, keepdim=True)
    p_y = p_xy.sum(dim=0, keepdim=True)
    
    # MI = sum p(x,y) log(p(x,y) / (p(x)p(y)))
    # Avoid log(0)
    p_xy_safe = p_xy + 1e-10
    p_x_safe = p_x + 1e-10
    p_y_safe = p_y + 1e-10
    
    mi = (p_xy * torch.log2(p_xy_safe / (p_x_safe * p_y_safe))).sum()
    
    return mi.item()


# ============================================================
# 8. Visualization Helpers
# ============================================================

def bits_to_string(bits: torch.Tensor) -> str:
    """
    Convert bit tensor to binary string for visualization.
    
    Args:
        bits: [m] binary tensor
    
    Returns:
        Binary string like "10110101"
    """
    return ''.join(['1' if b > 0.5 else '0' for b in bits.cpu().numpy()])


def print_capacity_table():
    """Print the theoretical capacity table."""
    print("\n" + "="*60)
    print("BREWA Theoretical Capacity Limits")
    print("="*60)
    print(f"{'d':<10} {'n_max':<15} {'Context Length':<20}")
    print("-"*60)
    
    for d in [64, 128, 256, 512, 1024]:
        n_max = max_context_length(d)
        if n_max < 1000:
            n_str = f"{n_max}"
        elif n_max < 1_000_000:
            n_str = f"{n_max/1000:.1f}K"
        else:
            n_str = f"{n_max/1_000_000:.1f}M"
        
        print(f"{d:<10} {n_max:<15} {n_str:<20}")
    
    print("="*60)
    print()


def print_scaling_law_table():
    """Print the scaling law table."""
    print("\n" + "="*70)
    print("BREWA Scaling Law: d ≥ (log N)²")
    print("="*70)
    print(f"{'Context N':<15} {'Required d':<15} {'Heads (d_h=64)':<20}")
    print("-"*70)
    
    for n in [1_000, 8_000, 32_000, 128_000, 1_000_000, 10_000_000]:
        d_req = required_dimension(n)
        heads = math.ceil(d_req / 64)
        
        if n < 1_000_000:
            n_str = f"{n//1000}K"
        else:
            n_str = f"{n//1_000_000}M"
        
        print(f"{n_str:<15} {d_req:<15} {heads:<20}")
    
    print("="*70)
    print()


if __name__ == "__main__":
    # Test Hadamard transform
    print("Testing Hadamard Transform...")
    x = torch.randn(2, 10, 64)
    h = hadamard_transform_batched(x)
    print(f"Input shape: {x.shape}, Output shape: {h.shape}")
    
    # Test bit quantization
    print("\nTesting Bit Quantization...")
    bits = bit_quantize(x, m_bits=32)
    print(f"Bits shape: {bits.shape}, Values: {bits[0, 0, :8]}")
    
    # Test Hamming similarity
    print("\nTesting Hamming Similarity...")
    x_bits = torch.randint(0, 2, (2, 5, 32)).float()
    y_bits = torch.randint(0, 2, (2, 7, 32)).float()
    sim = hamming_similarity_efficient(x_bits, y_bits)
    print(f"Similarity shape: {sim.shape}, Sample: {sim[0, 0, :]}")
    
    # Print capacity tables
    print_capacity_table()
    print_scaling_law_table()
