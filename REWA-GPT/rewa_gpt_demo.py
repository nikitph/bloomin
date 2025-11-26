
# REWA-GPT Marquee Demo
# Implements Hierarchical REWA Attention for Long Context Retrieval

import os
import math
import time
import random
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model, GPT2TokenizerFast
import psutil
import gc

# Cell 1: environment + imports
# Device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print("Device:", device)

# Repro
torch.manual_seed(42)
random.seed(42)

# Cell 2: helpers (gather along dim with index tensor of same shape)
def batched_gather(t: torch.Tensor, idx: torch.LongTensor, dim: int=1):
    """
    t: [B, N, D]
    idx: [B, N] indices in 0..N-1
    returns: [B, N, D] with t gathered per-batch, matching PyTorch gather semantics.
    """
    B, N, D = t.shape
    assert idx.shape == (B, N)
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, D)  # [B, N, D]
    return torch.gather(t, dim, idx_exp)

# Cell 3: Witness extraction
class WitnessExtractor:
    def __init__(self, tokenizer: GPT2TokenizerFast, kmeans_centers: Optional[torch.Tensor] = None):
        self.tokenizer = tokenizer
        # If no centers provided, make small random centers based on embedding dim later.
        self.kmeans_centers = kmeans_centers  # [C, emb_dim] if provided

    @staticmethod
    def token_shape_flags(token_str: str) -> Tuple[int,int,int]:
        # Returns simple flags (is_numeric, is_capitalized, has_special)
        is_num = any(ch.isdigit() for ch in token_str)
        is_cap = token_str[:1].isalpha() and token_str[:1].isupper()
        has_special = any(not ch.isalnum() for ch in token_str)
        return int(is_num), int(is_cap), int(has_special)

    def extract(self, input_ids: torch.LongTensor, token_texts: List[str], embeddings: Optional[torch.Tensor] = None):
        """
        input_ids: [N]
        token_texts: list of N token strings
        embeddings: optional [N, emb_dim] if available (for coarse cluster)
        Returns:
            W_coarse_idx: [N] coarse bucket id
            W_mid_idx: [N] mid bucket id
            W_fine_idx: [N] fine bucket id
        """
        N = input_ids.shape[0]
        # coarse: token_id hash (coarse_bucket_mod)
        coarse_buckets = (input_ids.cpu().numpy() % 1024).tolist()  # 0..1023

        # mid: token shape + sliding window hash
        mid_buckets = []
        for i, s in enumerate(token_texts):
            num, cap, sp = self.token_shape_flags(s)
            # combine into small hash
            h = (num << 0) | (cap << 1) | (sp << 2)
            # mix with local context (low-cost): length mod 16
            h = (h * 31 + len(s)) % 2048
            mid_buckets.append(h)

        # fine: position bin + small ngram fingerprint
        fine_buckets = []
        for i, s in enumerate(token_texts):
            # position bin -- coarse grid over sequence
            pos_bin = i // 64  # adjust resolution externally for long contexts
            fingerprint = (hash(s) & 0xffff) % 4096
            fine_buckets.append((pos_bin * 4096 + fingerprint) % 65536)

        return torch.tensor(coarse_buckets, dtype=torch.long), torch.tensor(mid_buckets, dtype=torch.long), torch.tensor(fine_buckets, dtype=torch.long)

# Cell 4: REWA hierarchical attention
class RewaHierarchicalAttention(nn.Module):
    """
    Hierarchical REWA attention with 3-level routing (coarse, mid, fine).
    This is an attention drop-in patch for GPT2Model blocks.
    """

    def __init__(self, config: GPT2Config,
                 bucket_sizes: Tuple[int,int,int] = (256, 64, 16),
                 n_hashes: int = 2,
                 use_rotary: bool = False):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        # hierarchical bucket sizes per level
        self.bucket_sizes = bucket_sizes  # (coarse, mid, fine)
        self.n_hashes = n_hashes
        self.use_rotary = use_rotary

        # random probe matrices per head and per level (cached)
        # We'll lazily initialize probes to shape [levels, heads, head_dim, probe_dim]
        self.probes = None
        
        # Storage for witness buckets if injected
        self.witness_buckets = None

    def set_witness_buckets(self, buckets):
        self.witness_buckets = buckets

    def _ensure_probes(self, device, levels=3, probe_dim=64):
        if self.probes is None:
            # choose probe_dim per level relative to bucket sizes
            heads = self.n_head
            self.probes = nn.Parameter(torch.randn(levels, heads, self.head_dim, probe_dim, device=device), requires_grad=False)
            # keep as buffers to avoid optimizer issues
            for i in range(levels):
                self.register_buffer(f"_probe_{i}", self.probes[i])

    def hash_for_level(self, x_head: torch.Tensor, probe: torch.Tensor, n_buckets: int):
        """
        x_head: [B, N, D]
        probe: [B, D, probe_dim] (batched per head)
        Returns: bucket_idx [B, N] in 0..n_buckets-1
        Approach: project -> sign/argmax trick -> compress to bucket via hashing
        """
        # project: [B, N, probe_dim]
        # probe is [B_heads, D, probe_dim], x_head is [B_heads, N, D]
        proj = torch.einsum("bnd,bdp->bnp", x_head, probe)  # cheap
        # combine projections to single integer bucket index
        # using argmax across probe_dim (gives region)
        region = torch.argmax(proj, dim=-1)  # [B, N], values 0..probe_dim-1
        # compress to buckets by hashing region with a simple operation
        # add mod to keep well distributed
        bucket_idx = (region * 9973) % n_buckets
        return bucket_idx

    def chunked_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bucket_idx: torch.Tensor, chunk_size: int):
        """
        q,k,v: [B, N, D] merged (batch*head, N, D)
        bucket_idx: [B, N] bucket ids
        chunk_size: approx number tokens per chunk
        returns: out [B, N, D]
        Implementation:
          - argsort by bucket_idx -> sorted sequences
          - split into chunks of size chunk_size, compute local attention within chunk
          - optionally attend to previous chunk to handle boundaries
        """
        B, N, D = q.shape
        device = q.device

        # argsort and gather
        sorted_idx = torch.argsort(bucket_idx, dim=1)  # [B, N]
        unsort_idx = torch.argsort(sorted_idx, dim=1)

        q_sorted = batched_gather(q, sorted_idx)  # [B, N, D]
        k_sorted = batched_gather(k, sorted_idx)
        v_sorted = batched_gather(v, sorted_idx)

        # We'll process in chunks
        out_sorted = torch.zeros_like(q_sorted)
        # number chunks
        n_chunks = math.ceil(N / chunk_size)
        # pad to multiple of chunk_size for easier reshape
        pad_len = n_chunks * chunk_size - N
        if pad_len > 0:
            pad_q = torch.zeros(B, pad_len, D, device=device, dtype=q_sorted.dtype)
            pad_k = torch.zeros_like(pad_q)
            pad_v = torch.zeros_like(pad_q)
            q_pad = torch.cat([q_sorted, pad_q], dim=1)
            k_pad = torch.cat([k_sorted, pad_k], dim=1)
            v_pad = torch.cat([v_sorted, pad_v], dim=1)
        else:
            q_pad, k_pad, v_pad = q_sorted, k_sorted, v_sorted

        # reshape to [B, n_chunks, chunk_size, D]
        q_chunks = q_pad.view(B, n_chunks, chunk_size, D)
        k_chunks = k_pad.view(B, n_chunks, chunk_size, D)
        v_chunks = v_pad.view(B, n_chunks, chunk_size, D)

        # For each chunk, compute attention within chunk and with previous chunk (to handle boundary)
        out_chunks = []
        for i in range(n_chunks):
            q_i = q_chunks[:, i]  # [B, chunk, D]
            # attend to chunk i and optionally chunk i-1
            if i > 0:
                k_cat = torch.cat([k_chunks[:, i-1], k_chunks[:, i]], dim=1)  # [B, 2*chunk, D]
                v_cat = torch.cat([v_chunks[:, i-1], v_chunks[:, i]], dim=1)
            else:
                k_cat = k_chunks[:, i]
                v_cat = v_chunks[:, i]
            # compute scaled dot-product attention
            # using explicit matmul -> small shapes so okay
            scores = torch.matmul(q_i, k_cat.transpose(-2, -1)) / math.sqrt(D)  # [B, chunk, 2*chunk?]
            attn = F.softmax(scores, dim=-1)
            out_i = torch.matmul(attn, v_cat)  # [B, chunk, D]
            out_chunks.append(out_i)

        out_pad = torch.cat(out_chunks, dim=1)  # [B, n_chunks*chunk, D]
        # crop to N (remove pad)
        out_sorted = out_pad[:, :N, :]
        # unsort back
        out = batched_gather(out_sorted, unsort_idx)
        return out

    def forward(self, x: torch.Tensor, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False, witness_buckets=None, **kwargs):
        """
        x: [batch, seq_len, embed_dim]
        witness_buckets: tuple of 3 tensors [seq] or [batch, seq] with bucket ids
        """
        if witness_buckets is None:
            witness_buckets = self.witness_buckets
            
        if witness_buckets is None:
             # Fallback or error? For this demo, we assume they are set.
             # But to be safe if called without them (e.g. initial check), we might return dummy.
             # However, let's assume they are provided or injected.
             raise ValueError("witness_buckets must be provided or set via set_witness_buckets")

        B, N, E = x.shape
        # Project QKV
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # reshape for heads: [B, heads, N, head_dim] -> merge batch and head -> [B*heads, N, D]
        def reshape_for_heads(t):
            t = t.view(B, N, self.n_head, self.head_dim).permute(0,2,1,3).reshape(B*self.n_head, N, self.head_dim)
            return t

        qh = reshape_for_heads(q)
        kh = reshape_for_heads(k)
        vh = reshape_for_heads(v)

        # ensure probes initialized
        device = x.device
        self._ensure_probes(device=device, levels=3, probe_dim=64)
        # probes from buffers
        probes = [getattr(self, f"_probe_{i}") for i in range(3)]  # each [heads, D, probe_dim]

        # witness_buckets are given per token [seq] or [batch, seq]
        # Convert to per head batch: expand across heads then merge batch*head
        # If witness_buckets are shape [N], make them [B,N]
        wb_coarse, wb_mid, wb_fine = witness_buckets
        if wb_coarse.dim() == 1:
            wb_coarse = wb_coarse.unsqueeze(0).expand(B, -1).to(device)
            wb_mid = wb_mid.unsqueeze(0).expand(B, -1).to(device)
            wb_fine = wb_fine.unsqueeze(0).expand(B, -1).to(device)

        # Expand for heads: [B, N] -> [B*heads, N]
        def expand_heads(w):
            w = w.unsqueeze(1).expand(B, self.n_head, N).reshape(B*self.n_head, N)
            return w

        wb_coarse_h = expand_heads(wb_coarse)
        wb_mid_h = expand_heads(wb_mid)
        wb_fine_h = expand_heads(wb_fine)

        # For each level compute bucket idx via probes then perform chunked_attention
        # Level -> bucket_count
        level_bucket_counts = [
            max(1, N // self.bucket_sizes[0]),
            max(1, N // self.bucket_sizes[1]),
            max(1, N // self.bucket_sizes[2])
        ]
        level_chunk_sizes = [
            self.bucket_sizes[0],
            self.bucket_sizes[1],
            self.bucket_sizes[2]
        ]

        outs = torch.zeros_like(qh)
        # compute per-level and accumulate
        provided_buckets = [wb_coarse_h, wb_mid_h, wb_fine_h]
        
        for lvl in range(3):
            probe_lvl = probes[lvl]  # [heads, D, probe_dim] (buffer)
            # replicate probe for batch*head merging: we need per merged head probe of shape [D, probe_dim]
            # probes stored per head; expand to B*head by repeating
            probe_per_merged = probe_lvl.repeat(B, 1, 1)  # [B*heads, D, probe_dim]
            
            # compute bucket indices via projection of qh OR use provided buckets
            if provided_buckets[lvl] is not None:
                bucket_idx = provided_buckets[lvl]
            else:
                # qh: [B*head, N, D], probe_per_merged: [B*head, D, p]
                bucket_idx = self.hash_for_level(qh, probe_per_merged, level_bucket_counts[lvl])
            
            # chunked attention
            out_lvl = self.chunked_attention(qh, kh, vh, bucket_idx, chunk_size=level_chunk_sizes[lvl])
            # accumulate (equal weights for now)
            outs += out_lvl

        # average across levels
        outs = outs / 3.0

        # reshape back to [B, N, E]
        outs = outs.view(B, self.n_head, N, self.head_dim).permute(0,2,1,3).reshape(B, N, E)
        out = self.out_proj(outs)
        
        # GPT2 attention returns (output, present)
        return (out, None)

# Cell 5: patch GPT-2
def patch_gpt2_with_rewa(config: GPT2Config, model: GPT2Model, bucket_sizes=(256,64,16)):
    """
    Replace attention modules inside model.h (Block) with RewaHierarchicalAttention.
    Returns patched model.
    """
    for i, block in enumerate(model.h):
        # Some GPT2 blocks include attention inside a module attr 'attn' or 'attn.c_attn' etc.
        # For transformers GPT2Model: block.attn is a module with c_attn, c_proj
        # We'll replace block.attn with our module while preserving out_proj weights shape
        rewa_attn = RewaHierarchicalAttention(config, bucket_sizes=bucket_sizes)
        # copy linear weights shapes (optional) - we leave as independent init for demo
        model.h[i].attn = rewa_attn
    return model

# Cell 6: sinusoidal positional embeddings
def make_sinusoidal_positional_embeddings(max_len: int, dim: int, device):
    pe = torch.zeros(max_len, dim, device=device)
    position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # [max_len, dim]

# Cell 7: demo harness
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def build_long_input(seed_positions: List[int], needle: str, total_len: int, vocab_size: int):
    """
    Build token ids with needle inserted at specified positions.
    For speed, we generate random tokens elsewhere.
    """
    ids = torch.randint(0, vocab_size, (total_len,), dtype=torch.long)
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    for pos in seed_positions:
        if pos + len(needle_ids) < total_len:
            ids[pos:pos+len(needle_ids)] = torch.tensor(needle_ids, dtype=torch.long)
    return ids, needle_ids

def run_rewa_retrieval(model: GPT2Model, input_ids: torch.LongTensor, needle_ids: List[int], top_k=5):
    """
    1) compute hidden states (last hidden layer)
    2) compute witness buckets from token strs and run REWA routing to compute per-token aggregated representation
    3) compute similarity between final query token (last token) and all earlier tokens in representation space
    4) return top-k nearest token indices
    """
    model.eval()
    with torch.no_grad():
        # decode token texts
        token_texts = tokenizer.batch_decode(input_ids.tolist(), clean_up_tokenization_spaces=False)
        # lightweight embeddings for witness extractor: use small token embedding from model.wte if exists
        try:
            wte = model.wte.weight  # if loaded
            emb = wte[input_ids].to(device)  # [N, E]
        except Exception:
            emb = None

        extractor = WitnessExtractor(tokenizer)
        wb_coarse, wb_mid, wb_fine = extractor.extract(input_ids.cpu(), token_texts, embeddings=None)
        # move to device
        wb_coarse = wb_coarse.to(device)
        wb_mid = wb_mid.to(device)
        wb_fine = wb_fine.to(device)

        # Inject witness buckets into all layers
        for block in model.h:
             if hasattr(block.attn, 'set_witness_buckets'):
                 block.attn.set_witness_buckets((wb_coarse, wb_mid, wb_fine))

        # compute hidden states
        outputs = model(input_ids.unsqueeze(0).to(device), output_hidden_states=True)
        last_hidden = outputs.last_hidden_state.squeeze(0)  # [N, E]

        # We'll compute pairwise similarity between final token hidden vector and each earlier token, 
        # but only after routing with REWA attention — here we reuse the model's patched attention by doing a forward pass 
        # that returns per-token attended outputs (we will call the model blocks individually)
        # For simplicity, we simulate by computing dot-product similarity on last_hidden (this is a proxy).
        q_vec = last_hidden[-1]  # [E]
        sims = torch.matmul(last_hidden, q_vec)
        topk = torch.topk(sims[:-1], k=top_k)  # ignore the last token itself
        return topk.indices.cpu().tolist(), sims.cpu().tolist()

# Cell 9: basic benchmark wrapper (time + memory)
def memory_stats():
    cuda_mem = torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0
    # For MPS we don't have a direct equivalent of memory_allocated easily accessible in same way, 
    # but we can track RSS
    return {"cuda_alloc": cuda_mem, "rss": psutil.Process().memory_info().rss / (1024 * 1024)}

def benchmark_retrieval(model, seq_len=200000, positions=[100,50000,190000], needle="THE SECRET CODE IS: 3141592653"):
    start = time.time()
    ids, needle_ids = build_long_input(positions, needle, seq_len, tokenizer.vocab_size)
    print(f"Input built. len={seq_len}")
    mem_before = memory_stats()
    print("mem before (MB):", mem_before)
    topk_idx, sims = run_rewa_retrieval(model, ids, needle_ids, top_k=10)
    mem_after = memory_stats()
    elapsed = time.time() - start
    print("Elapsed:", elapsed, "mem after (MB):", mem_after)
    print("Top hits:", topk_idx)
    found = any(any(idx <= p < idx+len(needle_ids) for p in positions) for idx in topk_idx)
    print("Found:", found)
    return found

if __name__ == "__main__":
    # Setup
    cfg = GPT2Config.from_pretrained("gpt2")
    # for extremely large n_ctx we will not use the built-in positional embeddings; instead we'll use RoPE or sinusoidal patch
    cfg.n_ctx = 1024  # keep small for default; we'll manage longer contexts via our witness extractor and not rely on model wpe
    model = GPT2Model(cfg)
    print("Original model loaded. Patching with REWA...")
    model = patch_gpt2_with_rewa(cfg, model, bucket_sizes=(256,64,16))
    if device.type != "cpu":
        model = model.half()
    model = model.to(device)
    model.eval()

    # Quick functional test at small scale
    print("\n--- Running Small Scale Test (4096) ---")
    small_len = 4096
    needle = "THE SECRET CODE IS: 3141592653"
    positions = [100, 2000, 3000]
    ids, needle_ids = build_long_input(positions, needle, small_len, tokenizer.vocab_size)
    topk_idx, sims = run_rewa_retrieval(model, ids, needle_ids, top_k=5)
    print("Topk similar positions:", topk_idx)
    # inspect whether any of inserted positions present
    found = any(any(idx <= p < idx+len(needle_ids) for p in positions) for idx in topk_idx)
    print("Needle found among topk:", found)

    # Example run at smaller size to validate pipeline
    print("\n--- Running Benchmark (8192) ---")
    found = benchmark_retrieval(model, seq_len=8192, positions=[100,2000,5000])
    
    # Note: For 200k, uncomment below. Be aware of memory usage.
    print("\n--- Running Benchmark (200k) ---")
    found = benchmark_retrieval(model, seq_len=200000, positions=[100, 50000, 150000])
