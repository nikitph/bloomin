# rewa_gpt_v4_utils.py
# ---------------------------------------------------------------
# Shared utilities for REWA-GPT v4
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

# ============================================================
# 1. Deterministic 64-bit Stable Hash
# ============================================================

def torch_stable_hash_int64(arr: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """
    A stable 64-bit integer hash implemented entirely in PyTorch operations.
    Works on CPU and MPS/GPU.
    """
    x = arr.to(torch.int64)
    x = (x ^ seed) & 0xFFFFFFFFFFFFFFFF

    x ^= (x >> 30)
    x = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF

    x ^= (x >> 27)
    x = (x * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF

    x ^= (x >> 31)
    return x


# ============================================================
# 2. Rotary Positional Embeddings (RoPE)
# ============================================================

def apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
    """
    x: [B, N, H, D], sin/cos: [N, D]
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    sin = sin.unsqueeze(0).unsqueeze(0)
    cos = cos.unsqueeze(0).unsqueeze(0)
    x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_rot.flatten(-2)


def build_rope_freqs(max_len: int, dim: int, device):
    """
    Build sinusoidal frequency matrix for RoPE.
    Returns sin, cos tensors of shape [max_len, dim].
    """
    theta = 10000 ** (-torch.arange(0, dim, 2, device=device).float() / dim)
    pos = torch.arange(max_len, device=device).float().unsqueeze(1)
    angles = pos * theta.unsqueeze(0)
    sin = torch.sin(angles)
    cos = torch.cos(angles)
    # restore interleaved shape
    sin_full = torch.zeros(max_len, dim, device=device)
    cos_full = torch.zeros(max_len, dim, device=device)
    sin_full[..., ::2] = sin
    cos_full[..., ::2] = cos
    sin_full[..., 1::2] = sin
    cos_full[..., 1::2] = cos
    return sin_full, cos_full


# ============================================================
# 3. Exact Match Utility for Needle Retrieval
# ============================================================

def exact_match_candidates(input_ids: torch.Tensor,
                           needle_ids: torch.Tensor,
                           window: int = 16) -> torch.Tensor:
    """
    Return indices where the n-gram matches the needle exactly.
    """
    N = input_ids.shape[0]
    L = needle_ids.shape[0]

    if L > N:
        return torch.tensor([], dtype=torch.long)

    cands = []
    needle = needle_ids.tolist()
    seq = input_ids.tolist()

    for i in range(0, N - L + 1):
        if seq[i:i+L] == needle:
            cands.append(i)

    if not cands:
        return torch.tensor([], dtype=torch.long)

    return torch.tensor(cands, dtype=torch.long)


# ============================================================
# 4. Vectorized Local Window Similarity
# ============================================================

def local_similarity(hidden: torch.Tensor,
                     query: torch.Tensor,
                     window: int = 32) -> torch.Tensor:
    """
    Compute dot-product similarity ONLY in windows around the sequence.
    hidden: [N, E]
    query: [E]
    returns: [N]
    """
    N, E = hidden.shape
    sims = hidden @ query

    # Optionally, restrict to windows (not enforced in v4)
    return sims


# ============================================================
# 5. Chunking Utilities
# ============================================================

def chunk_tokens(t: torch.Tensor, chunk_size: int):
    """
    Yield slices of t in chunks.
    """
    N = t.shape[0]
    for i in range(0, N, chunk_size):
        yield t[i:i+chunk_size]


# ============================================================
# 6. Vectorized Arg-Sort Grouping
# ============================================================

def group_by_buckets(q, k, v, bucket_idx, chunk_size):
    """
    Vectorized grouping helper for chunked attention.
    """
    B, N, D = q.shape
    sorted_idx = torch.argsort(bucket_idx, dim=1)
    unsort_idx = torch.argsort(sorted_idx, dim=1)

    q_sorted = torch.gather(q, 1, sorted_idx.unsqueeze(-1).expand(-1,-1,D))
    k_sorted = torch.gather(k, 1, sorted_idx.unsqueeze(-1).expand(-1,-1,D))
    v_sorted = torch.gather(v, 1, sorted_idx.unsqueeze(-1).expand(-1,-1,D))

    return q_sorted, k_sorted, v_sorted, unsort_idx