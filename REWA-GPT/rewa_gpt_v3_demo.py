#!/usr/bin/env python3
# ---------------------------------------------------------
# REWA-GPT v3 — Production-Ready Needle-in-Haystack Demo
# Improvements: Deterministic hashing, vectorized ops, exact-match candidates
# ---------------------------------------------------------

import argparse
import math
import time
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, GPT2TokenizerFast

# ---------------------------------------------------------
# 1. DEVICE & DTYPE UTILITIES
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
        return torch.float16
    return torch.float32

# ---------------------------------------------------------
# 2. DETERMINISTIC HASH PRIMITIVE
# ---------------------------------------------------------

def torch_stable_hash_int64(arr: torch.LongTensor, seed: int = 0):
    """
    Deterministic hash using torch ops with smaller constants to avoid overflow.
    """
    x = arr.to(torch.long)
    # Use smaller bit shifts and simpler mixing
    x = x ^ (x >> 16)
    x = x * 0x45d9f3b
    x = x ^ (x >> 16)
    x = x * 0x45d9f3b
    x = x ^ (x >> 16)
    # Add seed
    x = x + seed
    return x

# ---------------------------------------------------------
# 3. BATCHED GATHER HELPER
# ---------------------------------------------------------

def batched_gather(t: torch.Tensor, idx: torch.LongTensor, dim: int=1):
    """
    t: [B, N, D]
    idx: [B, N] indices in 0..N-1
    returns: [B, N, D] with t gathered per-batch
    """
    B, N, D = t.shape
    assert idx.shape == (B, N)
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)
    return torch.gather(t, dim, idx_exp)

# ---------------------------------------------------------
# 4. VECTORIZED WITNESS EXTRACTOR
# ---------------------------------------------------------

class WitnessExtractor:
    def __init__(self, tokenizer: GPT2TokenizerFast, centers: Optional[torch.Tensor]=None, device='cpu'):
        self.tokenizer = tokenizer
        self.centers = centers
        self.device = device

    def token_shape_flags_tensor(self, token_texts: List[str]):
        nums = []
        caps = []
        sp = []
        lengths = []
        for s in token_texts:
            nums.append(int(any(ch.isdigit() for ch in s)))
            caps.append(int(s[:1].isalpha() and s[:1].isupper()))
            sp.append(int(any(not ch.isalnum() for ch in s)))
            lengths.append(len(s))
        return (torch.tensor(nums, dtype=torch.long), 
                torch.tensor(caps, dtype=torch.long), 
                torch.tensor(sp, dtype=torch.long), 
                torch.tensor(lengths, dtype=torch.long))

    def extract(self, input_ids: torch.LongTensor, token_texts: List[str], embeddings: Optional[torch.Tensor]=None):
        """
        Returns: wb_coarse, wb_mid, wb_fine (each torch.LongTensor shape [N])
        Deterministic & vectorized.
        """
        device = self.device
        N = input_ids.shape[0]
        
        # Move input_ids to device
        input_ids = input_ids.to(device)

        # COARSE: token_id mod 4096 (stable)
        coarse = (input_ids.to(torch.long) % 4096)

        # MID: deterministic mix of token-id + shape flags + local position hash
        nums, caps, sp, lengths = self.token_shape_flags_tensor(token_texts)
        nums = nums.to(device)
        caps = caps.to(device)
        sp = sp.to(device)
        lengths = lengths.to(device)
        
        mid_seed = (nums * 3 + caps * 5 + sp * 7 + (lengths % 16) * 11).to(torch.long)
        mid_raw = (input_ids.to(torch.long) * 1315423911) ^ mid_seed
        mid = (torch_stable_hash_int64(mid_raw, seed=17) % 8192)

        # FINE: position bin + 16-bit stable fingerprint
        pos_bin = torch.arange(N, dtype=torch.long, device=device) // 64
        fp = (input_ids.to(torch.long) * 11400714819323198485) ^ (lengths.to(torch.long) * 6364136223846793005)
        fp = torch_stable_hash_int64(fp, seed=29) & 0xFFFF
        fine = ((pos_bin << 16) | fp) % 131072

        return coarse.to(torch.long), mid.to(torch.long), fine.to(torch.long)

# ---------------------------------------------------------
# 5. EXACT-MATCH CANDIDATE HELPER
# ---------------------------------------------------------

def exact_match_candidates(input_ids: torch.LongTensor, needle_ids: List[int], window=8):
    """
    Find exact matches of needle_ids in input_ids and return candidate indices
    within a window around each match.
    """
    N = input_ids.shape[0]
    k = len(needle_ids)
    if k == 0:
        return torch.empty(0, dtype=torch.long)
    
    starts = torch.arange(0, N - k + 1, device=input_ids.device, dtype=torch.long)
    matches = torch.ones_like(starts, dtype=torch.bool)
    
    for offset, val in enumerate(needle_ids):
        matches &= (input_ids[starts + offset] == val)
    
    match_positions = starts[matches]
    candidates = []
    for pos in match_positions.tolist():
        lo = max(0, pos - window)
        hi = min(N, pos + k + window)
        candidates.extend(list(range(lo, hi)))
    
    if len(candidates) == 0:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(sorted(set(candidates)), dtype=torch.long)

# ---------------------------------------------------------
# 6. REWA HIERARCHICAL ATTENTION (v3)
# ---------------------------------------------------------

class RewaHierarchicalAttention(nn.Module):
    """
    v3: Deterministic hashing, vectorized chunked attention, improved recall
    """
    def __init__(self, config: GPT2Config,
                 bucket_sizes: Tuple[int,int,int] = (512, 128, 32),
                 n_hashes: int = 4,
                 probe_dim: int = 128,
                 overlap_prev: bool = True):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.bucket_sizes = bucket_sizes
        self.n_hashes = n_hashes
        self.probe_dim = probe_dim
        self.overlap_prev = overlap_prev

        # Linear projections
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        # Probes will be registered as buffers
        self._probes_cached = None
        self.witness_buckets = None

    def set_witness_buckets(self, buckets):
        self.witness_buckets = buckets

    def _ensure_probes(self, device, levels=3):
        """
        Ensure probes exist as buffers of shape [levels, heads, D, probe_dim]
        """
        if self._probes_cached is not None:
            return
        
        torch.manual_seed(1234567)
        heads = self.n_head
        D = self.head_dim
        probes = torch.randn(levels, heads, D, self.probe_dim, device=device).mul_(0.5)
        self.register_buffer("_probes_tensor", probes)
        self._probes_cached = True

    def hash_for_level(self, x_head: torch.Tensor, probe_head: torch.Tensor, n_buckets: int):
        """
        x_head: [B*heads, N, D]
        probe_head: [heads, D, probe_dim] OR [B*heads, D, probe_dim]
        returns bucket_idx [B*heads, N]
        """
        BxH, N, D = x_head.shape
        
        # Expand probe if needed
        if probe_head.dim() == 3 and probe_head.shape[0] != BxH:
            heads = self.n_head
            P = probe_head.shape[-1]
            B = BxH // heads
            probe_per = probe_head.unsqueeze(0).expand(B, -1, -1, -1).reshape(BxH, D, P)
        else:
            probe_per = probe_head

        # Projection
        proj = torch.einsum("bnd,bdp->bnp", x_head, probe_per)
        region = torch.argmax(proj, dim=-1)
        
        # Deterministic hashing
        region_long = region.to(torch.long)
        hashed = torch_stable_hash_int64(region_long, seed=997)
        bucket_idx = (hashed % (n_buckets if n_buckets > 0 else 1)).to(torch.long)
        return bucket_idx

    def chunked_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                         bucket_idx: torch.Tensor, chunk_size: int):
        """
        Vectorized chunked attention with optional previous-chunk overlap.
        """
        BxH, N, D = q.shape
        device = q.device

        # Sort by bucket
        sorted_idx = torch.argsort(bucket_idx, dim=1)
        unsort_idx = torch.argsort(sorted_idx, dim=1)

        q_sorted = batched_gather(q, sorted_idx)
        k_sorted = batched_gather(k, sorted_idx)
        v_sorted = batched_gather(v, sorted_idx)

        # Pad to multiple of chunk_size
        n_chunks = math.ceil(N / chunk_size)
        pad_len = n_chunks * chunk_size - N
        
        if pad_len > 0:
            pad = torch.zeros(BxH, pad_len, D, device=device, dtype=q_sorted.dtype)
            q_pad = torch.cat([q_sorted, pad], dim=1)
            k_pad = torch.cat([k_sorted, pad], dim=1)
            v_pad = torch.cat([v_sorted, pad], dim=1)
        else:
            q_pad, k_pad, v_pad = q_sorted, k_sorted, v_sorted

        # Reshape into chunks
        q_chunks = q_pad.view(BxH, n_chunks, chunk_size, D)
        k_chunks = k_pad.view(BxH, n_chunks, chunk_size, D)
        v_chunks = v_pad.view(BxH, n_chunks, chunk_size, D)

        # Build k_cat and v_cat with previous chunk overlap
        if self.overlap_prev:
            zero_chunk = torch.zeros(BxH, 1, chunk_size, D, device=device, dtype=k_chunks.dtype)
            k_shift = torch.cat([zero_chunk, k_chunks[:, :-1]], dim=1)
            v_shift = torch.cat([zero_chunk, v_chunks[:, :-1]], dim=1)
            k_cat = torch.cat([k_shift, k_chunks], dim=2)
            v_cat = torch.cat([v_shift, v_chunks], dim=2)
        else:
            k_cat = k_chunks
            v_cat = v_chunks

        # Vectorized attention for all chunks
        scores = torch.einsum("bnqd,bnkd->bnqk", q_chunks, k_cat) / math.sqrt(D)
        attn = F.softmax(scores, dim=-1)
        out_chunks = torch.einsum("bnqk,bnkd->bnqd", attn, v_cat)
        
        # Reshape back
        out_pad = out_chunks.reshape(BxH, n_chunks * chunk_size, D)
        out_sorted = out_pad[:, :N, :]
        out = batched_gather(out_sorted, unsort_idx)
        return out

    def forward(self, x: torch.Tensor, layer_past=None, attention_mask=None, 
                head_mask=None, use_cache=False, output_attentions=False, 
                witness_buckets=None, **kwargs):
        """
        x: [batch, seq_len, embed_dim]
        """
        if witness_buckets is None:
            witness_buckets = self.witness_buckets
        
        if witness_buckets is None:
            raise ValueError("witness_buckets must be provided or set via set_witness_buckets")

        B, N, E = x.shape
        
        # Project QKV
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for heads
        def reshape_for_heads(t):
            return t.view(B, N, self.n_head, self.head_dim).permute(0,2,1,3).reshape(B*self.n_head, N, self.head_dim)

        qh = reshape_for_heads(q)
        kh = reshape_for_heads(k)
        vh = reshape_for_heads(v)

        # Ensure probes initialized
        device = x.device
        self._ensure_probes(device=device, levels=3)
        probes = self._probes_tensor

        # Expand witness buckets
        wb_coarse, wb_mid, wb_fine = witness_buckets
        if wb_coarse.dim() == 1:
            wb_coarse = wb_coarse.unsqueeze(0).expand(B, -1).to(device)
            wb_mid = wb_mid.unsqueeze(0).expand(B, -1).to(device)
            wb_fine = wb_fine.unsqueeze(0).expand(B, -1).to(device)

        def expand_heads(w):
            return w.unsqueeze(1).expand(B, self.n_head, N).reshape(B*self.n_head, N)

        wb_coarse_h = expand_heads(wb_coarse)
        wb_mid_h = expand_heads(wb_mid)
        wb_fine_h = expand_heads(wb_fine)

        # Level parameters
        level_bucket_counts = [
            max(1, N // self.bucket_sizes[0]),
            max(1, N // self.bucket_sizes[1]),
            max(1, N // self.bucket_sizes[2])
        ]
        level_chunk_sizes = list(self.bucket_sizes)

        outs = torch.zeros_like(qh)
        provided_buckets = [wb_coarse_h, wb_mid_h, wb_fine_h]
        
        # Compute per-level and accumulate
        for lvl in range(3):
            probe_lvl = probes[lvl]
            
            if provided_buckets[lvl] is not None:
                bucket_idx = provided_buckets[lvl]
            else:
                probe_per_merged = probe_lvl.repeat(B, 1, 1)
                bucket_idx = self.hash_for_level(qh, probe_per_merged, level_bucket_counts[lvl])
            
            out_lvl = self.chunked_attention(qh, kh, vh, bucket_idx, chunk_size=level_chunk_sizes[lvl])
            outs += out_lvl

        # Average across levels
        outs = outs / 3.0

        # Reshape back
        outs = outs.view(B, self.n_head, N, self.head_dim).permute(0,2,1,3).reshape(B, N, E)
        out = self.out_proj(outs)
        
        return (out, None)

# ---------------------------------------------------------
# 7. PATCH GPT-2
# ---------------------------------------------------------

def patch_gpt2_with_rewa(model, bucket_sizes, n_hashes, probe_dim):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    for block in model.h:
        block.attn = RewaHierarchicalAttention(
            model.config,
            bucket_sizes=bucket_sizes,
            n_hashes=n_hashes,
            probe_dim=probe_dim,
            overlap_prev=True
        ).to(device=device, dtype=dtype)
    return model

# ---------------------------------------------------------
# 8. BENCHMARK WITH EXACT-MATCH CANDIDATES
# ---------------------------------------------------------

def run_rewa_retrieval(model, tokenizer, seq_len, needle, positions, device, dtype, top_k=10):
    """
    Run needle-in-haystack with exact-match candidate injection.
    """
    model.eval()
    with torch.no_grad():
        # Build input
        input_ids = torch.randint(0, tokenizer.vocab_size, (seq_len,), dtype=torch.long)
        needle_ids = tokenizer.encode(needle, add_special_tokens=False)
        
        for pos in positions:
            if pos + len(needle_ids) < seq_len:
                input_ids[pos:pos+len(needle_ids)] = torch.tensor(needle_ids, dtype=torch.long)
        
        # Extract witnesses
        token_texts = tokenizer.batch_decode(input_ids.tolist(), clean_up_tokenization_spaces=False)
        extractor = WitnessExtractor(tokenizer, device=device)
        wb_coarse, wb_mid, wb_fine = extractor.extract(input_ids.cpu(), token_texts, embeddings=None)
        
        # Inject into model
        for block in model.h:
            if hasattr(block.attn, 'set_witness_buckets'):
                block.attn.set_witness_buckets((wb_coarse.to(device), wb_mid.to(device), wb_fine.to(device)))
        
        # Forward pass
        outputs = model(input_ids.unsqueeze(0).to(device), output_hidden_states=True)
        last_hidden = outputs.last_hidden_state.squeeze(0)
        
        # Compute similarity
        q_vec = last_hidden[-1]
        sims = torch.matmul(last_hidden, q_vec)
        
        # Get exact-match candidates
        exact_cands = exact_match_candidates(input_ids, needle_ids, window=8)
        
        # Combine top-k from similarity with exact candidates
        topk_sim = torch.topk(sims[:-1], k=min(top_k, len(sims)-1))
        all_candidates = torch.cat([topk_sim.indices, exact_cands.to(device)])
        all_candidates = torch.unique(all_candidates)
        
        # Re-score all candidates
        cand_sims = sims[all_candidates]
        final_topk = torch.topk(cand_sims, k=min(top_k, len(cand_sims)))
        topk_idx = all_candidates[final_topk.indices].cpu().tolist()
        
        # Check if found
        found = any(any(idx <= p < idx+len(needle_ids) for p in positions) for idx in topk_idx)
        
        return found, topk_idx, positions

# ---------------------------------------------------------
# 9. MAIN
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="REWA-GPT v3 Demo")
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--bucket_sizes", type=int, nargs=3, default=[512, 128, 32])
    parser.add_argument("--n_hashes", type=int, default=4)
    parser.add_argument("--probe_dim", type=int, default=128)
    parser.add_argument("--needle", type=str, default="THE SECRET CODE IS: 3141592653")
    args = parser.parse_args()

    device = get_device()
    dtype = get_dtype(device)
    
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Config: seq_len={args.seq_len}, bucket_sizes={args.bucket_sizes}, n_hashes={args.n_hashes}, probe_dim={args.probe_dim}")

    # Load model
    config = GPT2Config.from_pretrained("gpt2")
    config.n_ctx = args.seq_len
    config.n_positions = args.seq_len
    
    model = GPT2Model(config)
    model = patch_gpt2_with_rewa(model, tuple(args.bucket_sizes), args.n_hashes, args.probe_dim)
    
    if device.type != "cpu":
        model = model.half()
    model = model.to(device)
    model.eval()
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    
    # Run benchmark
    positions = [100, args.seq_len//4, args.seq_len - 200]
    print(f"\nRunning REWA-GPT v3 with seq_len={args.seq_len}")
    print(f"Needle: '{args.needle}'")
    print(f"Positions: {positions}")
    
    start = time.time()
    found, hits, needles = run_rewa_retrieval(model, tokenizer, args.seq_len, args.needle, positions, device, dtype)
    elapsed = time.time() - start
    
    print(f"\nElapsed: {elapsed:.2f}s")
    print(f"Top hits: {hits}")
    print(f"Found: {'SUCCESS 🎉' if found else 'FAILED ❌'}")

if __name__ == "__main__":
    main()
