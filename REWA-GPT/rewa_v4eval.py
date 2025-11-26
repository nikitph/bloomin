# rewa_v4eval.py
# ---------------------------------------------------------------
# REWA-GPT v4 evaluator — supports multi-token needles (sequence match)
# ---------------------------------------------------------------

import torch
import math
import random
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2Config, GPT2Model
from rewa_gpt_v4_train import WitnessExtractor, RewaHierarchicalAttention, patch_gpt2_with_rewa

# -----------------------------
# Device setup
# -----------------------------
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
# Build test sequence with needle inserted (may be multi-token)
# ============================================================
def build_test_sequence(tokenizer, seq_len=1024, needle="SECRET123"):
    # random tokens
    ids = torch.randint(0, tokenizer.vocab_size, (seq_len,), dtype=torch.long)
    needle_ids = tokenizer.encode(needle, add_special_tokens=False)
    L = len(needle_ids)
    # choose insertion position ensuring room for the whole needle and not at position 0
    pos = random.randint(1, seq_len - L - 1)
    ids[pos:pos+L] = torch.tensor(needle_ids, dtype=torch.long)
    return ids, pos, needle_ids

# ============================================================
# Compute sequence log-probabilities under the model
# (logits[t] predicts token at position t+1 in our training setup)
# ============================================================
def rewa_sequence_retrieval(model, tokenizer, input_ids, needle_ids, top_k=10):
    model.eval()
    N = input_ids.shape[0]

    # 1. Witness extraction and injection
    token_texts = tokenizer.batch_decode(input_ids.tolist(), clean_up_tokenization_spaces=False)
    extractor = WitnessExtractor(tokenizer)
    wb_coarse, wb_mid, wb_fine = extractor.extract(input_ids.cpu(), token_texts)
    buckets = (wb_coarse.to(device), wb_mid.to(device), wb_fine.to(device))

    for block in model.h:
        # attach witness buckets if block has set_witness_buckets
        if hasattr(block.attn, "set_witness_buckets"):
            block.attn.set_witness_buckets(buckets)

    # 2. Forward pass (single batch)
    with torch.no_grad():
        out = model(input_ids.unsqueeze(0).to(device), output_hidden_states=False)
        hidden = out.last_hidden_state.squeeze(0)  # [N, E]

    # 3. Compute logits and log-probs
    # logits[t] corresponds to prediction for token at position t+1 (matching training)
    logits = hidden @ model.wte.weight.T  # [N, V]
    log_probs = F.log_softmax(logits, dim=-1)  # [N, V]

    # 4. Score every possible start position s for the needle (multi-token)
    needle_len = len(needle_ids)
    candidates = []
    # Valid starts: s in [1, N - needle_len] inclusive (since logits[s-1] predicts token s)
    s_min = 1
    s_max = N - needle_len  # inclusive
    if s_max < s_min:
        return [], [], []  # sequence too short

    # We'll vectorize moderately: gather the required log_probs indices for each offset
    # Build list of (positions, token) pairs per start s
    # For start s: need positions P = [s-1 + j for j in 0..L-1], tokens T = needle_ids[j]
    # We'll compute score_s = sum_j log_probs[P[j], T[j]]
    for s in range(s_min, s_max + 1):
        positions = [s - 1 + j for j in range(needle_len)]
        # sum log-probabilities
        score = 0.0
        valid = True
        for pos, tok in zip(positions, needle_ids):
            if pos < 0 or pos >= N:
                valid = False
                break
            score += float(log_probs[pos, tok].cpu().item())
        if valid:
            candidates.append((s, score))
    if len(candidates) == 0:
        return [], [], []

    # 5. Select Top-K by score (higher = more likely)
    candidates.sort(key=lambda x: x[1], reverse=True)
    topk = candidates[:top_k]
    pred_positions = [s for s, sc in topk]
    scores = [sc for s, sc in topk]

    # Also return full per-position score array if caller wants it
    # Build full_score array aligned to start positions s in [0..N-1] with None for invalid starts
    full_scores = [None] * N
    for s, sc in candidates:
        full_scores[s] = sc

    return pred_positions, scores, full_scores

# ============================================================
# Main entry
# ============================================================
if __name__ == "__main__":
    tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")

    cfg = GPT2Config.from_pretrained("distilgpt2")
    cfg.n_ctx = 2048
    cfg.n_positions = 2048

    # Load model and checkpoint
    model = GPT2Model(cfg)
    model = patch_gpt2_with_rewa(cfg, model)
    # load your trained checkpoint (if path differs, change)
    state = torch.load("checkpoints/rewa_gpt_v4.pt", map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    if device.type != "cpu":
        model = model.half()

    print("REWA-GPT v4 loaded on", device)

    # Build test
    needle_text = "THE SECRET CODE IS: 3141592653"  # example (multi-token)
    seq_len = 1024
    input_ids, true_pos, needle_ids = build_test_sequence(tokenizer, seq_len=seq_len, needle=needle_text)

    print(f"Inserted needle (len={len(needle_ids)}) at position {true_pos}")
    print("Needle token ids:", needle_ids[:10], "..." if len(needle_ids)>10 else "")

    # Run retrieval
    topk_idx, topk_scores, all_scores = rewa_sequence_retrieval(model, tokenizer, input_ids, needle_ids, top_k=20)

    print("Top-k predicted start positions:", topk_idx)
    print("Top-k scores:", topk_scores)
    found = any(true_pos == s for s in topk_idx)
    print("Needle found (exact start match):", found)

    # after you've computed topk_idx and all_scores in rewa_v4eval.py
    true_score = all_scores[true_pos]
    print("True start pos score:", true_score)
    print("Top1 score:", topk_scores[0], "Top1 pos:", topk_idx[0])

    # Bonus: print best candidate context snippet
    if len(topk_idx) > 0:
        s = topk_idx[0]
        excerpt = tokenizer.decode(input_ids[s:s+len(needle_ids)].tolist(), clean_up_tokenization_spaces=False)
        print(f"Top-1 excerpt at {s}:", repr(excerpt))