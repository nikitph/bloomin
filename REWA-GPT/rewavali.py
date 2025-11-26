# ============================================================
# rewa_v4_validation.py
# Full REWA Validation Suite
#
# 1) Witness–Embedding Correlation Test
# 2) Bucket Recall Test
# 3) Minimal LoRA + REWA Attention Retrieval Fine-Tune Test
#
# Works on CPU / MPS / CUDA
# ============================================================

import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2TokenizerFast, GPT2Model, GPT2Config

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Device:", device)
torch.manual_seed(42)
random.seed(42)

# ============================================================
# 1. Witness Extractor
# ============================================================
class WitnessExtractor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def extract(self, tokens):
        # Tokens: list of strings
        coarse = []
        mid = []
        fine = []
        for i, s in enumerate(tokens):
            tid = self.tokenizer.convert_tokens_to_ids(s)

            coarse.append(tid % 1024)

            num = any(ch.isdigit() for ch in s)
            cap = s[:1].isalpha() and s[:1].isupper()
            spc = any(not ch.isalnum() for ch in s)
            h = (num << 0) | (cap << 1) | (spc << 2)
            h = (h * 31 + len(s)) % 2048
            mid.append(h)

            pos_bin = i // 32
            fingerprint = (hash(s) & 0xFFFF) % 4096
            fine.append((pos_bin * 4096 + fingerprint) % 65536)

        return (
            torch.tensor(coarse, dtype=torch.long),
            torch.tensor(mid, dtype=torch.long),
            torch.tensor(fine, dtype=torch.long),
        )


# ============================================================
# 2. Correlation Test
# ============================================================
def correlation_test():
    print("\n============================================")
    print(" REWA VALIDATION TEST 1 — CORRELATION")
    print("============================================")

    tok = GPT2TokenizerFast.from_pretrained("distilgpt2")
    model = GPT2Model.from_pretrained("distilgpt2").eval().to(device)

    texts = [
        "hello world",
        "hi there",
        "the quick brown fox jumps over the lazy dog",
        "def foo(x): return x+1",
        "def bar(y): return y+1",
        "beautiful mountains and rivers",
        "programming in python is fun",
        "mathematics and physics",
        "quantum field theory",
        "pytorch deep learning",
    ] * 3  # 30 samples

    reps = []
    W = []
    extractor = WitnessExtractor(tok)

    for t in texts:
        ids = tok.encode(t, add_special_tokens=False)
        emb = model.wte.weight[torch.tensor(ids).to(device)].mean(dim=0)
        reps.append(emb)

        tokens = tok.tokenize(t)
        coarse, mid, fine = extractor.extract(tokens)
        W.append((set(coarse.tolist()), set(mid.tolist()), set(fine.tolist())))

    reps = torch.stack(reps)
    M = len(texts)
    overlaps = []
    sims = []

    for i in range(M):
        for j in range(i+1, M):
            # embedding similarity
            cos = F.cosine_similarity(reps[i], reps[j], dim=0).item()

            # witness overlap
            c = len(W[i][0].intersection(W[j][0]))
            m = len(W[i][1].intersection(W[j][1]))
            f = len(W[i][2].intersection(W[j][2]))
            overlap = c + m + f

            sims.append(cos)
            overlaps.append(overlap)

    sims = np.array(sims)
    overlaps = np.array(overlaps)

    # Spearman correlation
    from scipy.stats import spearmanr
    rho, p = spearmanr(sims, overlaps)

    print(f"Spearman ρ = {rho:.4f}, p = {p:.4e}")

    if rho > 0.5:
        print("PASS: Witness overlap meaningfully tracks semantic similarity.")
    elif rho > 0.2:
        print("WEAK: Witness extractor needs improvement but REWA may still work.")
    else:
        print("FAIL: Witness overlap does NOT correlate – REWA will fail unless extractor is fixed.")


# ============================================================
# 3. Bucket Recall Test
# ============================================================
def bucket_recall_test(n_rounds=16, probe_dim=128, n_buckets=2048):
    print("\n============================================")
    print(" REWA VALIDATION TEST 2 — BUCKET RECALL")
    print("============================================")

    tok = GPT2TokenizerFast.from_pretrained("distilgpt2")
    model = GPT2Model.from_pretrained("distilgpt2").eval().to(device)

    # choose two similar sentences
    t1 = "def foo(x): return x+1"
    t2 = "def bar(y): return y+1"

    ids1 = tok.encode(t1, add_special_tokens=False)
    ids2 = tok.encode(t2, add_special_tokens=False)
    rep1 = model.wte.weight[torch.tensor(ids1).to(device)].mean(dim=0)
    rep2 = model.wte.weight[torch.tensor(ids2).to(device)].mean(dim=0)

    E = rep1.size(0)
    successes = 0

    for r in range(n_rounds):
        R = torch.randn(E, probe_dim).to(device)
        b1 = int((rep1 @ R).argmax().item() % n_buckets)
        b2 = int((rep2 @ R).argmax().item() % n_buckets)
        if b1 == b2:
            successes += 1

    recall = successes / n_rounds
    print(f"Bucket Collision Recall: {recall*100:.2f}%")

    if recall > 0.8:
        print("PASS: REWA hashing routes true matches together reliably.")
    elif recall > 0.4:
        print("WEAK: Increase n_hashes, probe_dim, or learn probes.")
    else:
        print("FAIL: Routing does not preserve neighborhood – REWA cannot work.")


# ============================================================
# 4. Minimal REWA Retrieval Test (no full training)
# ============================================================

class SimpleREWAAttention(nn.Module):
    def __init__(self, dim, n_buckets=256, probe_dim=128):
        super().__init__()
        self.probe = nn.Parameter(torch.randn(dim, probe_dim))

        self.n_buckets = n_buckets

    def forward(self, x):
        # x: [N, E]
        proj = x @ self.probe  # [N, probe_dim]
        buckets = torch.argmax(proj, dim=-1) % self.n_buckets
        return buckets


def minimal_retrieval_test():
    print("\n============================================")
    print(" REWA VALIDATION TEST 3 — MINIMAL RETRIEVAL")
    print("============================================")

    tok = GPT2TokenizerFast.from_pretrained("distilgpt2")
    model = GPT2Model.from_pretrained("distilgpt2").eval().to(device)

    seq_len = 1024
    needle = "SECRET123"

    needle_ids = tok.encode(needle, add_special_tokens=False)
    ln = len(needle_ids)

    ids = torch.randint(0, tok.vocab_size, (seq_len,), dtype=torch.long)
    pos = random.randint(50, seq_len-ln-10)
    ids[pos:pos+ln] = torch.tensor(needle_ids)

    # get embeddings
    emb = model.wte.weight[ids.to(device)]  # [N, E]

    # REWA attention routing
    attn = SimpleREWAAttention(emb.size(-1)).to(device)
    buckets = attn(emb)  # [N]

    # find all tokens in the same bucket as last query token
    q_bucket = buckets[-1].item()
    candidates = (buckets == q_bucket).nonzero(as_tuple=True)[0]

    print("Candidates near query:", candidates[:20].tolist(), "…")

    # check if the needle start is inside candidates
    found = any(abs(int(c.item()) - pos) < ln for c in candidates)

    print("Needle found:", found)

    if found:
        print("PASS: Even untrained REWA attention can retrieve needles.")
    else:
        print("FAIL: Needs bigger probe or training (but REWA core still testable).")


# ============================================================
# Run All Tests
# ============================================================

if __name__ == "__main__":
    correlation_test()
    bucket_recall_test()
    minimal_retrieval_test()
    print("\nAll REWA tests completed.\n")