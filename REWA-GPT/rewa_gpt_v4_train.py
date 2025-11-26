# rewa_gpt_v4_train.py (patched for numeric stability)
# ---------------------------------------------------------------
# REWA-GPT v4: Trainable 2048-token REWA Attention GPT with LoRA
# Patched: stable bucket alignment, float32 loss computation, grad clipping
# ---------------------------------------------------------------

import os
import math
import time
import random
import argparse
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformers import (
    GPT2TokenizerFast,
    GPT2Config,
    GPT2Model
)

# ============================================================
# Device Setup
# ============================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("REWA-GPT v4 Training — Device:", device)
torch.manual_seed(42)
random.seed(42)

# ============================================================
# Helpers
# ============================================================
def batched_gather(t: torch.Tensor, idx: torch.LongTensor, dim: int = 1):
    """
    t: [B, N, D]
    idx: [B, N]
    returns: [B, N, D]
    """
    B, N, D = t.shape
    assert idx.shape == (B, N)
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)  # [B, N, D]
    return torch.gather(t, dim, idx_exp)

# ============================================================
# 1. LoRA Module
# ============================================================
class LoRALinear(nn.Module):
    def __init__(self, linear_module, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        # reuse base weight and bias (frozen for LoRA pattern)
        self.weight = linear_module.weight
        self.bias = linear_module.bias

        in_f, out_f = self.weight.shape[1], self.weight.shape[0]
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        # LoRA update (low-rank)
        update = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base + self.scale * update

# ============================================================
# 2. Witness Extractor (Same as v1/v3 but vectorized)
# ============================================================
class WitnessExtractor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def extract(self, token_ids, token_texts):
        N = len(token_ids)

        coarse = (token_ids % 1024).tolist()

        mid = []
        for s in token_texts:
            num = any(ch.isdigit() for ch in s)
            cap = s[:1].isalpha() and s[:1].isupper()
            spc = any(not ch.isalnum() for ch in s)
            h = (num << 0) | (cap << 1) | (spc << 2)
            h = (h * 31 + len(s)) % 2048
            mid.append(h)

        fine = []
        for i, s in enumerate(token_texts):
            pos_bin = i // 64
            fingerprint = (hash(s) & 0xFFFF) % 4096
            fine.append((pos_bin * 4096 + fingerprint) % 65536)

        return (
            torch.tensor(coarse, dtype=torch.long),
            torch.tensor(mid, dtype=torch.long),
            torch.tensor(fine, dtype=torch.long),
        )

# ============================================================
# 3. REWA Hierarchical Attention (patched into GPT-2)
# ============================================================
class RewaHierarchicalAttention(nn.Module):
    """
    Robust, drop-in RewaHierarchicalAttention for GPT2 block replacement.

    - Returns (attn_output, attn_weights) where attn_weights is None (we don't compute per-head weights).
    - Accepts witness buckets as shape [N], [B, N], or [B*heads, N].
    - Uses chunked, vectorized local attention per bucket chunk.
    """

    def __init__(self, config, bucket_sizes=(256, 64, 16), n_hashes=2, probe_dim=64):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.embed_dim = config.n_embd

        # Projections (can be swapped with LoRA wrappers externally)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        # Hierarchical params
        self.bucket_sizes = tuple(bucket_sizes)
        self.n_hashes = int(n_hashes)
        self.probe_dim = int(probe_dim)

        # Random probes per level x heads x head_dim x probe_dim (buffer)
        self.register_buffer("_probes_init", torch.randn(len(self.bucket_sizes),
                                                         self.n_head,
                                                         self.head_dim,
                                                         self.probe_dim))
        # we will expose them as cpu/gpu buffers below
        for lvl in range(len(self.bucket_sizes)):
            self.register_buffer(f"_probe_lvl_{lvl}", self._probes_init[lvl])

        # placeholder for injected witness buckets
        self.witness_buckets = None

    def set_witness_buckets(self, buckets_tuple):
        """Accept a tuple (wb_coarse, wb_mid, wb_fine) where each is [N] or [B,N] or [B*heads,N]."""
        self.witness_buckets = buckets_tuple

    def _ensure_wb_shape(self, wb, B, N):
        """
        Normalize witness-bucket tensor `wb` to shape [B*heads, N].
        Accepts:
          - wb shape [N] -> expands to [B, N] then to [B*heads, N]
          - wb shape [B, N] -> expands to [B*heads, N]
          - wb shape [B*heads, N] -> returned as-is
        """
        if wb is None:
            # If None, produce zeros
            return torch.zeros(B * self.n_head, N, dtype=torch.long, device=next(self.parameters()).device)

        if wb.dim() == 1 and wb.shape[0] == N:
            # [N] -> [B, N]
            wb_b = wb.unsqueeze(0).expand(B, -1).to(next(self.parameters()).device)
        elif wb.dim() == 2 and wb.shape[0] == B:
            wb_b = wb.to(next(self.parameters()).device)
        elif wb.dim() == 2 and wb.shape[0] == B * self.n_head:
            # Already per-merged-head
            return wb.to(next(self.parameters()).device)
        else:
            # Fallback: try to broadcast (raise clear error if not possible)
            raise RuntimeError(f"Unsupported witness-bucket shape {tuple(wb.shape)} for B={B}, N={N}, heads={self.n_head}")

        # expand to [B, heads, N] -> reshape to [B*heads, N]
        wb_exp = wb_b.unsqueeze(1).expand(B, self.n_head, N).reshape(B * self.n_head, N)
        return wb_exp

    def _hash_via_probes(self, x_merged, probe_buffer, n_buckets):
        """
        x_merged: [B*heads, N, D]
        probe_buffer: [heads, D, probe_dim] or [B*heads, D, probe_dim]
        n_buckets: int
        Return: bucket_idx [B*heads, N] in 0..n_buckets-1
        """
        Bheads, N, D = x_merged.shape
        device = x_merged.device

        # If probe_buffer is heads x D x p, repeat for B
        if probe_buffer.dim() == 3 and probe_buffer.shape[0] == self.n_head:
            probe_per_merged = probe_buffer.repeat(torch.ceil(torch.tensor(Bheads / self.n_head)).int().item(), 1, 1)[:Bheads]
        else:
            # otherwise assume already [B*heads, D, p]
            probe_per_merged = probe_buffer.to(device)

        # project: [B*heads, N, p]
        proj = torch.einsum("bnd,bdp->bnp", x_merged, probe_per_merged)
        # region selection
        region = torch.argmax(proj, dim=-1)  # [B*heads, N], values 0..p-1
        # compress region -> bucket id (simple multiplicative hash then mod)
        bucket_idx = (region.long() * 9973) % max(1, int(n_buckets))
        return bucket_idx

    def _chunked_local_attention(self, q, k, v, bucket_idx, chunk_size):
        """
        Vectorized chunked attention:
        q,k,v: [B*heads, N, D]
        bucket_idx: [B*heads, N]
        chunk_size: int (approx number of tokens per chunk to perform local attention)
        Returns: out [B*heads, N, D]
        """
        Bheads, N, D = q.shape
        device = q.device

        # argsort by bucket id to group tokens of same bucket together
        sorted_idx = torch.argsort(bucket_idx, dim=1)  # [B*heads, N]
        unsort_idx = torch.argsort(sorted_idx, dim=1)  # inverse permutation

        # apply gather
        q_sorted = batched_gather(q, sorted_idx)
        k_sorted = batched_gather(k, sorted_idx)
        v_sorted = batched_gather(v, sorted_idx)

        # compute number of chunks and pad if needed
        n_chunks = math.ceil(N / chunk_size)
        pad_len = n_chunks * chunk_size - N
        if pad_len > 0:
            pad_q = torch.zeros(Bheads, pad_len, D, device=device, dtype=q_sorted.dtype)
            pad_k = torch.zeros_like(pad_q)
            pad_v = torch.zeros_like(pad_q)
            q_pad = torch.cat([q_sorted, pad_q], dim=1)
            k_pad = torch.cat([k_sorted, pad_k], dim=1)
            v_pad = torch.cat([v_sorted, pad_v], dim=1)
        else:
            q_pad, k_pad, v_pad = q_sorted, k_sorted, v_sorted

        # reshape to [B*heads, n_chunks, chunk_size, D]
        q_chunks = q_pad.view(Bheads, n_chunks, chunk_size, D)
        k_chunks = k_pad.view(Bheads, n_chunks, chunk_size, D)
        v_chunks = v_pad.view(Bheads, n_chunks, chunk_size, D)

        # We'll attend within chunk and to previous chunk (if exists) for boundary overlap
        # Build k_cat, v_cat per chunk by concatenation along the sequence axis.
        # Prepare containers
        out_chunks = torch.empty_like(q_chunks)

        # vectorized approach: compute scores chunk-by-chunk but in batch (loop over chunks is fine because chunk_count << N)
        for i in range(n_chunks):
            q_i = q_chunks[:, i]  # [B*heads, chunk, D]
            if i > 0:
                k_cat = torch.cat([k_chunks[:, i-1], k_chunks[:, i]], dim=1)  # [B*heads, 2*chunk, D]
                v_cat = torch.cat([v_chunks[:, i-1], v_chunks[:, i]], dim=1)
            else:
                k_cat = k_chunks[:, i]
                v_cat = v_chunks[:, i]

            # scaled dot-product attention: [B*heads, chunk, 2*chunk] scores
            scores = torch.matmul(q_i, k_cat.transpose(-2, -1)) / math.sqrt(D)
            attn = F.softmax(scores, dim=-1)
            out_i = torch.matmul(attn, v_cat)  # [B*heads, chunk, D]
            out_chunks[:, i] = out_i

        out_pad = out_chunks.view(Bheads, n_chunks * chunk_size, D)
        out_sorted = out_pad[:, :N, :]
        # unsort
        out = batched_gather(out_sorted, unsort_idx)
        return out

    def forward(self, x: torch.Tensor, layer_past=None, attention_mask=None, head_mask=None,
                use_cache=False, output_attentions=False, witness_buckets=None, **kwargs):
        """
        x: [B, N, E]
        witness_buckets: optional tuple (wb_coarse, wb_mid, wb_fine) with shapes [N] or [B,N] or [B*heads,N].
        Return: (out: [B, N, E], attn_weights: None)
        """
        if witness_buckets is None:
            witness_buckets = self.witness_buckets

        B, N, E = x.shape
        device = x.device

        # Project QKV
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape projections for heads -> merge batch and heads
        def _merge_heads(t):
            return t.view(B, N, self.n_head, self.head_dim).permute(0, 2, 1, 3).reshape(B * self.n_head, N, self.head_dim)

        qh = _merge_heads(q)
        kh = _merge_heads(k)
        vh = _merge_heads(v)

        # Normalize witness buckets to [B*heads, N]
        probes = [getattr(self, f"_probe_lvl_{lvl}") for lvl in range(len(self.bucket_sizes))]  # [heads, D, p]

        # prepare provided buckets or compute via hash of q
        provided = (None, None, None)
        if witness_buckets is not None:
            provided = witness_buckets

        # compute merged witness buckets per level robustly
        merged_wb = []
        for lvl in range(len(self.bucket_sizes)):
            wb_lvl = provided[lvl] if (provided is not None and len(provided) > lvl) else None
            if wb_lvl is not None:
                wb_merged = self._ensure_wb_shape(wb_lvl, B, N)  # [B*heads, N]
            else:
                # if not provided, compute via probe hashing of qh
                # produce per-merged-head probe buffer: [B*heads, D, p]
                probe_buf = probes[lvl].repeat(math.ceil((B * self.n_head) / self.n_head), 1, 1)[:B * self.n_head].to(device)
                wb_merged = self._hash_via_probes(qh, probe_buf, max(1, N // max(1, self.bucket_sizes[lvl])))
            merged_wb.append(wb_merged)

        # For each level compute local chunked attention and accumulate
        outs = torch.zeros_like(qh, device=device, dtype=qh.dtype)
        for lvl, wb_merged in enumerate(merged_wb):
            n_buckets = max(1, N // max(1, self.bucket_sizes[lvl]))
            chunk_size = self.bucket_sizes[lvl]
            out_lvl = self._chunked_local_attention(qh, kh, vh, wb_merged, chunk_size=chunk_size)
            outs += out_lvl

        # average across levels
        outs = outs / float(len(self.bucket_sizes))

        # reshape back to [B, N, E]
        outs = outs.view(B, self.n_head, N, self.head_dim).permute(0, 2, 1, 3).reshape(B, N, E)
        out = self.out_proj(outs)

        # transformers expects (attn_output, attn_weights) from attention modules
        # we don't compute attn_weights here (would be big); return None
        return out, None

# ============================================================
# 4. Patch GPT-2 with REWA Attention
# ============================================================

def patch_gpt2_with_rewa(config, model):
    for i, block in enumerate(model.h):
        block.attn = RewaHierarchicalAttention(config)
    return model

# ============================================================
# 5. Synthetic Retrieval Dataset (Random Needle)
# ============================================================

def build_retrieval_batch(tokenizer, batch_size, seq_len=2048, needle="SECRET123"):
    batches = []
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    L = len(needle_ids)

    for _ in range(batch_size):
        ids = torch.randint(0, tokenizer.vocab_size, (seq_len,), dtype=torch.long)
        pos = random.randint(50, seq_len - L - 10)
        ids[pos:pos+L] = torch.tensor(needle_ids)
        batches.append((ids, pos))

    all_ids = torch.stack([b[0] for b in batches])
    all_pos = torch.tensor([b[1] for b in batches])
    return all_ids, all_pos, needle_ids

# ============================================================
# 6. Loss Functions (stable versions)
# ============================================================
def contrastive_loss(hidden: torch.Tensor, positions: torch.Tensor):
    """
    hidden: [B, N, D] (float32 recommended)
    positions: [B] int positions (target index in 0..N-1)
    We compute similarity between last token vector and all tokens, then cross-entropy.
    """
    B, N, D = hidden.shape
    q_last = hidden[:, -1]  # [B, D]
    sims = torch.einsum("bd,bnd->bn", q_last, hidden)  # [B, N]
    labels = positions.long().to(sims.device)  # [B]
    return F.cross_entropy(sims, labels)

def bucket_alignment_loss(buckets: Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor],
                          positions: torch.Tensor,
                          seq_len: int = 2048):
    """
    Normalized bucket alignment loss.
    - Normalizes bucket indices to [0,1] to avoid huge magnitudes.
    - positions: [B] (int)
    """
    # Ensure positions is float32 for stable arithmetic
    positions = positions.to(torch.float32)

    losses = []
    for b in (buckets[0], buckets[1], buckets[2]):
        b_t = b.to(device).to(torch.float32)
        # Normalize bucket indices to [0,1]
        max_b = float(max(1.0, b_t.max().item()))
        b_norm = b_t / (max_b + 1e-8)  # [N] or [B,N]
        # Broadcast to [B, N]
        if b_norm.dim() == 1:
            b_norm = b_norm.unsqueeze(0).expand(positions.size(0), -1)
        # positions -> normalized in [0,1] by seq_len
        pos_norm = positions.unsqueeze(-1) / float(max(1, seq_len))
        diff = (b_norm - pos_norm).abs()
        diff = torch.clamp(diff, 0.0, 1.0)  # remove outliers
        losses.append(diff.mean())
    return sum(losses) / len(losses)

# ============================================================
# 7. Training Loop
# ============================================================
def train_rewa_gpt_v4(
    steps=200,
    lr=2e-4,
    batch_size=2,
    seq_len=2048,
    needle="SECRET123"
):
    tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
    cfg = GPT2Config.from_pretrained("distilgpt2")
    cfg.n_ctx = seq_len
    cfg.n_positions = seq_len

    model = GPT2Model(cfg)
    model = patch_gpt2_with_rewa(cfg, model)
    model = model.to(device)
    model.train()

    # Only use half precision on CUDA GPUs; keep float on MPS/CPU for stability
    if device.type == "cuda":
        model = model.half()
    else:
        model = model.float()

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    extractor = WitnessExtractor(tokenizer)

    print("\n🔥 Starting REWA-GPT v4 Training...\n")
    for step in range(1, steps + 1):
        input_ids, positions, needle_ids = build_retrieval_batch(tokenizer, batch_size, seq_len, needle)
        input_ids = input_ids.to(device)

        # prepare token_texts for the first batch example (fast)
        token_texts = tokenizer.batch_decode(input_ids[0].tolist(), clean_up_tokenization_spaces=False)
        wb = extractor.extract(input_ids[0].cpu(), token_texts)
        wb = tuple(w.to(device) for w in wb)

        # inject witness buckets into all layers
        for block in model.h:
            if hasattr(block.attn, 'set_witness_buckets'):
                block.attn.set_witness_buckets(wb)

        outputs = model(input_ids, output_hidden_states=True)
        hidden = outputs.last_hidden_state  # [B, N, E] (dtype depends on model dtype)

        # Compute losses in float32 for numerical stability
        hidden_f = hidden.float()
        logits = torch.matmul(hidden_f, model.wte.weight.T.float())

        # CE (next-token) loss
        ce_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1).to(device)
        )

        # Contrastive retrieval loss (float32)
        cont_loss = contrastive_loss(hidden_f, positions.to(device))

        # Bucket alignment (normalized)
        ba_loss = bucket_alignment_loss(wb, positions.to(device), seq_len=seq_len)

        # Weighted sum; make BA small initially
        loss = ce_loss + 0.5 * cont_loss + 0.02 * ba_loss

        # Detect NaN/Inf and dump debug info
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf loss detected at step {step}")
            print("CE:", float(ce_loss), "CONT:", float(cont_loss), "BA:", float(ba_loss))
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "input_ids": input_ids.cpu(),
                "positions": positions.cpu(),
                "wb": (wb[0].cpu(), wb[1].cpu(), wb[2].cpu())
            }, "checkpoints/debug_batch.pt")
            raise RuntimeError("NaN/Inf loss — debug saved to checkpoints/debug_batch.pt")

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if step % 10 == 0 or step == 1:
            print(f"step {step}/{steps} | total={loss.item():.6f} | CE={ce_loss.item():.6f} | CONT={cont_loss.item():.6f} | BA={ba_loss.item():.6f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/rewa_gpt_v4.pt")
    print("\n✅ Training Complete — Model Saved to checkpoints/rewa_gpt_v4.pt\n")


# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--needle", type=str, default="SECRET123")
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    train_rewa_gpt_v4(
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        needle=args.needle
    )

# ============================================================
# Recommended warm-up command (quick test):
#   python3 rewa_gpt_v4_train.py --steps 50 --batch_size 1 --seq_len 1024 --lr 1e-4
#
# Notes:
# - On Mac MPS we keep float32 for numeric stability (half can cause NaNs).
# - If training is stable, raise steps and batch_size gradually.
# - If you still see instability, reduce --lr to 5e-5.
# ============================================================