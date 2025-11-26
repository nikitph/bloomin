# rewa_gpt_v4_eval.py
# ---------------------------------------------------------------
# Evaluate REWA-GPT v4 Retrieval on long-context sequences
# ---------------------------------------------------------------

import os
import math
import time
import random
import argparse
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, GPT2TokenizerFast

# ============================================================
# Device
# ============================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("REWA-GPT v4 Eval — Device:", device)

# ============================================================
# 1. LoRA block (same as train file)
# ============================================================

class LoRALinear(nn.Module):
    def __init__(self, linear_module, r=8, alpha=16, dropout=0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

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
        update = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base + self.scale * update

# ============================================================
# 2. WitnessExtractor
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
# 3. REWA Hierarchical Attention (same as training)
# ============================================================

def batched_gather(t, idx, dim=1):
    B, N, D = t.shape
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(t, dim, idx_exp)

class RewaHierarchicalAttention(nn.Module):
    def __init__(self, config, bucket_sizes=(256,64,16), n_hashes=2):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = LoRALinear(nn.Linear(config.n_embd, config.n_embd))
        self.k_proj = LoRALinear(nn.Linear(config.n_embd, config.n_embd))
        self.v_proj = LoRALinear(nn.Linear(config.n_embd, config.n_embd))
        self.out_proj = LoRALinear(nn.Linear(config.n_embd, config.n_embd))

        self.bucket_sizes = bucket_sizes
        self.n_hashes = n_hashes
        self.probes = None
        self.witness_buckets = None

    def set_witness_buckets(self, buckets):
        self.witness_buckets = buckets

    def _ensure_probes(self, device, levels=3, probe_dim=64):
        if self.probes is None:
            probes = torch.randn(levels, self.n_head, self.head_dim, probe_dim, device=device)
            self.probes = nn.Parameter(probes, requires_grad=False)

    def chunked_attention(self, q, k, v, bucket_idx, chunk_size):
        B, N, D = q.shape
        sorted_idx = torch.argsort(bucket_idx, dim=1)
        unsort_idx = torch.argsort(sorted_idx, dim=1)

        q_sorted = batched_gather(q, sorted_idx)
        k_sorted = batched_gather(k, sorted_idx)
        v_sorted = batched_gather(v, sorted_idx)

        out = torch.zeros_like(q_sorted)

        n_chunks = math.ceil(N / chunk_size)
        for i in range(n_chunks):
            lo = i * chunk_size
            hi = min(N, lo + chunk_size)

            q_i = q_sorted[:, lo:hi]

            if i > 0:
                k_cat = k_sorted[:, lo - chunk_size:hi]
                v_cat = v_sorted[:, lo - chunk_size:hi]
            else:
                k_cat = k_sorted[:, lo:hi]
                v_cat = v_sorted[:, lo:hi]

            scores = torch.matmul(q_i, k_cat.transpose(-2,-1)) / math.sqrt(D)
            attn = F.softmax(scores, dim=-1)
            out[:, lo:hi] = torch.matmul(attn, v_cat)

        return batched_gather(out, unsort_idx)

    def forward(self, x, **kwargs):
        if self.witness_buckets is None:
            raise RuntimeError("Witness buckets not set in eval.")

        B, N, E = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        qh = q.view(B, N, self.n_head, self.head_dim).permute(0,2,1,3).reshape(B*self.n_head, N, self.head_dim)
        kh = k.view(B, N, self.n_head, self.head_dim).permute(0,2,1,3).reshape(B*self.n_head, N, self.head_dim)
        vh = v.view(B, N, self.n_head, self.head_dim).permute(0,2,1,3).reshape(B*self.n_head, N, self.head_dim)

        wb_coarse, wb_mid, wb_fine = self.witness_buckets
        wb_coarse = wb_coarse.to(x.device)
        wb_mid = wb_mid.to(x.device)
        wb_fine = wb_fine.to(x.device)

        def expand(w):
            return w.unsqueeze(1).expand(B, self.n_head, N).reshape(B*self.n_head, N)

        wb = [expand(wb_coarse), expand(wb_mid), expand(wb_fine)]
        chunk_sizes = list(self.bucket_sizes)

        outs = torch.zeros_like(qh)
        self._ensure_probes(x.device)

        for lvl in range(3):
            bucket_idx = wb[lvl]
            out_lvl = self.chunked_attention(qh, kh, vh, bucket_idx, chunk_size=chunk_sizes[lvl])
            outs += out_lvl

        outs = outs / 3.0
        outs = outs.view(B, self.n_head, N, self.head_dim).permute(0,2,1,3).reshape(B, N, E)
        return (self.out_proj(outs), )


# ============================================================
# Patch GPT-2
# ============================================================

def patch_gpt2_with_rewa(config, model):
    for i, block in enumerate(model.h):
        block.attn = RewaHierarchicalAttention(config)
    return model


# ============================================================
# Retrieval Test
# ============================================================

tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")

def build_long_input(seq_len, needle, pos):
    ids = torch.randint(0, tokenizer.vocab_size, (seq_len,), dtype=torch.long)
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    L = len(needle_ids)
    ids[pos:pos+L] = torch.tensor(needle_ids)
    return ids, needle_ids


def run_retrieval(model, input_ids, needle_ids):
    token_texts = tokenizer.batch_decode(input_ids.tolist(), clean_up_tokenization_spaces=False)
    extractor = WitnessExtractor(tokenizer)
    wb = extractor.extract(input_ids.cpu(), token_texts)
    wb = tuple(w.to(device) for w in wb)

    for block in model.h:
         block.attn.set_witness_buckets(wb)

    with torch.no_grad():
        outputs = model(input_ids.unsqueeze(0).to(device), output_hidden_states=True)
        hidden = outputs.last_hidden_state.squeeze(0)

    q = hidden[-1]
    sims = hidden @ q
    topk = torch.topk(sims[:-1], k=10).indices.cpu().tolist()
    return topk


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq_len", type=int, default=2048)  # Match training default
    parser.add_argument("--needle", type=str, default="SECRET123")
    parser.add_argument("--pos", type=int, default=1000)  # Changed from 2000 to fit in 2048
    args = parser.parse_args()

    # Load model with matching config
    cfg = GPT2Config.from_pretrained("distilgpt2")
    cfg.n_ctx = args.seq_len
    cfg.n_positions = args.seq_len  # Important: match training
    model = GPT2Model(cfg)
    model = patch_gpt2_with_rewa(cfg, model)
    model.load_state_dict(torch.load("checkpoints/rewa_gpt_v4.pt", map_location=device))
    model = model.to(device)
    model.eval()

    if device.type != "cpu":
        model = model.half()

    print("\n📥 Building Input...")
    input_ids, needle_ids = build_long_input(args.seq_len, args.needle, args.pos)

    print("\n🔍 Running Retrieval...")
    topk = run_retrieval(model, input_ids, needle_ids)

    print("\nTop-10 Retrieved Positions:", topk)
    print("Needle inserted at:", args.pos)

    found = any(abs(idx - args.pos) < len(needle_ids)*2 for idx in topk)
    print("\n🎯 FOUND?" , found)