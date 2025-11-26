#!/usr/bin/env python3
# ---------------------------------------------------------
# REWA-GPT v2 — Optimized Marquee Demo
# (c) Nikit Phadke 2025 — REWA Lab Prototype
# ---------------------------------------------------------

import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

# ---------------------------------------------------------
# 1. UTILITIES (DEVICE + DTYPE)
# ---------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("Using MPS")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")

def get_dtype(device):
    if device.type == "cuda":
        return torch.float16
    if device.type == "mps":
        # MPS fp16 mathematically is cast to fp32 internally; still reduces mem
        return torch.float16
    return torch.float32

# ---------------------------------------------------------
# 2. ROPE (Rotary Positional Embeddings)
# ---------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    Standard RoPE implementation for GPT-2 hidden dimension.
    """
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        # x: [B, H, T, D]
        t = torch.arange(x.size(2), device=x.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # D dims
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
        return (x * cos) + (rotated * sin)

# ---------------------------------------------------------
# 3. REWA HIERARCHICAL ATTENTION
# ---------------------------------------------------------

class RewaHierarchicalAttention(nn.Module):
    """
    Three-level hierarchical REWA attention:
    - Level 1: Global coarse routing
    - Level 2: Mid-level neighborhood routing
    - Level 3: Local fine-grained attention
    """
    def __init__(self, config, bucket_size=64, n_hashes=2, seed=1234):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.bucket_size = bucket_size
        self.n_hashes = n_hashes
        self.seed = seed

        # Linear projections
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # RoPE for Q/K
        self.rope = RotaryEmbedding(self.head_dim)

    # ========== HASHING ==========

    def hash_gaussian(self, x, n_buckets):
        """
        REWA-2 Gaussian Probe Hashing
        Deterministic due to explicit RNG state.
        """
        B, N, D = x.shape
        torch.manual_seed(self.seed)

        # Gaussian probes: [D, n_hashes, buckets//2]
        R = torch.randn(D, self.n_hashes, n_buckets // 2, device=x.device, dtype=x.dtype)

        proj = torch.einsum("bnd,dhm->bnhm", x, R)  # (b, n, n_hash, buckets//2)
        proj = torch.cat([proj, -proj], dim=-1)     # antipodal buckets

        buckets = torch.argmax(proj, dim=-1)        # (b, n, n_hash)
        return buckets

    # ========== ATTENTION ==========

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False, **kwargs):
        """
        x: [B, T, D]
        """
        B, T, D = x.shape

        # 1. Compute Q,K,V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(D, dim=-1)

        # Reshape to [B, H, T, HD]
        def reshape_heads(t):
            return (
                t.view(B, T, self.n_head, self.head_dim)
                .permute(0, 2, 1, 3)
            )  # [B, H, T, HD]

        q, k, v = map(reshape_heads, (q, k, v))

        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)

        # Merge batch+heads for hashing
        qh = q.reshape(B * self.n_head, T, self.head_dim)

        n_buckets = max(1, T // self.bucket_size)
        buckets = self.hash_gaussian(qh, n_buckets)  # [BH, T, n_hashes]

        # Level-1 coarse hash index
        b_idx = buckets[..., 0]  # [BH, T]

        # Sort tokens by bucket
        sorted_idx = torch.argsort(b_idx, dim=-1)
        unsort_idx = torch.argsort(sorted_idx, dim=-1)

        # Permute Q,K,V
        q_sort = torch.gather(qh, 1, sorted_idx.unsqueeze(-1).expand(-1, -1, self.head_dim))
        k_sort = torch.gather(k.reshape(B * self.n_head, T, self.head_dim), 1,
                              sorted_idx.unsqueeze(-1).expand(-1, -1, self.head_dim))
        v_sort = torch.gather(v.reshape(B * self.n_head, T, self.head_dim), 1,
                              sorted_idx.unsqueeze(-1).expand(-1, -1, self.head_dim))

        # Level-3: Chunk attention
        chunk = self.bucket_size * 2
        n_chunks = (T + chunk - 1) // chunk
        
        # Pad to multiple of chunk size
        pad_len = n_chunks * chunk - T
        if pad_len > 0:
            pad_tensor = torch.zeros(B * self.n_head, pad_len, self.head_dim, device=q_sort.device, dtype=q_sort.dtype)
            q_sort = torch.cat([q_sort, pad_tensor], dim=1)
            k_sort = torch.cat([k_sort, pad_tensor], dim=1)
            v_sort = torch.cat([v_sort, pad_tensor], dim=1)

        q_chunks = q_sort.reshape(B * self.n_head, n_chunks, chunk, self.head_dim)
        k_chunks = k_sort.reshape(B * self.n_head, n_chunks, chunk, self.head_dim)
        v_chunks = v_sort.reshape(B * self.n_head, n_chunks, chunk, self.head_dim)

        # Local window attention
        def local_attention(qc, kc, vc):
            # [BH, C, W, HD]
            attn = torch.einsum("bcid,bcjd->bcij", qc, kc) / math.sqrt(self.head_dim)
            attn = attn.softmax(dim=-1)
            return torch.einsum("bcij,bcjd->bcid", attn, vc)

        out_local = local_attention(q_chunks, k_chunks, v_chunks)
        out_local = out_local.reshape(B * self.n_head, n_chunks * chunk, self.head_dim)
        
        # Crop back to original length
        out_local = out_local[:, :T, :]

        # Unsort to original order
        out = torch.gather(out_local, 1, unsort_idx.unsqueeze(-1).expand(-1, -1, self.head_dim))

        # Reshape back to [B, T, D]
        out = out.reshape(B, self.n_head, T, self.head_dim).permute(0, 2, 1, 3)
        out = out.reshape(B, T, D)

        return (self.c_proj(out), None)

# ---------------------------------------------------------
# 4. PATCH GPT-2 WITH REWA ATTENTION
# ---------------------------------------------------------

def patch_gpt2_with_rewa(model, bucket_size, n_hashes):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for block in model.h:
        block.attn = RewaHierarchicalAttention(
            model.config,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            seed=1234
        ).to(device=device, dtype=dtype)
    return model

# ---------------------------------------------------------
# 5. NEEDLE-IN-HAYSTACK BENCHMARK
# ---------------------------------------------------------

def run_rewa_retrieval(model, seq_len, device, dtype):
    """
    Insert 3 needles at random positions.
    Ask model to predict token at last position.
    """
    with torch.no_grad():
        input_ids = torch.randint(0, 50000, (1, seq_len), device=device)
        needles = [100, seq_len//4, seq_len - 200]
        needle_token = 99999

        for pos in needles:
            input_ids[0, pos] = needle_token

        outputs = model(input_ids)[0]  # hidden states [1, T, D]
        last = outputs[0, -1]          # the final token representation

        # Compute nearest neighbors
        sim = torch.einsum("d,td->t", last, outputs[0])
        topk = torch.topk(sim, 10).indices.tolist()

        found = any(abs(idx - n) < 5 for idx in topk for n in needles)
        return found, topk, needles

# ---------------------------------------------------------
# 6. MAIN / CLI
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="REWA-GPT v2 Demo")
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--bucket_size", type=int, default=64)
    parser.add_argument("--n_hashes", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = get_device() if args.device == "auto" else torch.device(args.device)
    dtype = get_dtype(device)

    print(f"Device: {device}, dtype: {dtype}")

    # Patch GPT-2
    config = GPT2Config()
    config.n_ctx = args.seq_len
    config.n_positions = args.seq_len

    model = GPT2Model(config).to(device, dtype=dtype)
    model = patch_gpt2_with_rewa(model, args.bucket_size, args.n_hashes)
    model.eval()

    print(f"Running REWA-GPT v2 with seq_len={args.seq_len}")
    with torch.autocast(device_type=device.type, dtype=dtype):
        found, hits, needles = run_rewa_retrieval(model, args.seq_len, device, dtype)

    print("Needle positions:", needles)
    print("Top hits:", hits)
    print("Found:",
          "SUCCESS 🎉" if found else "FAILED ❌")

if __name__ == "__main__":
    main()
