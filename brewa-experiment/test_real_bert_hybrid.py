"""
Test Hybrid REWA on Real BERT Embeddings
========================================

Validate the Hybrid REWA encoder on real-world sentence embeddings from BERT.
This tests if the model generalizes to real semantic structures, not just synthetic clusters.
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertTokenizer
from hybrid_rewa_encoder import HybridREWAEncoder

def get_bert_embeddings(sentences):
    """Get [CLS] embeddings from BERT."""
    print("Loading BERT...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    
    embeddings = []
    print("Encoding sentences...")
    for sent in sentences:
        inputs = tokenizer(sent, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use [CLS] token embedding
        cls_embedding = outputs.last_hidden_state[0, 0, :]  # [768]
        embeddings.append(cls_embedding)
    
    return torch.stack(embeddings)

def test_real_sentences():
    print("="*70)
    print("Testing Hybrid REWA on Real BERT Embeddings")
    print("="*70)
    
    # 1. Define sentences with clear semantic groups
    sentences = [
        # Group 1: Cats (0, 1, 2)
        "The cat sat on the mat",
        "A feline rested on the carpet",
        "Cats enjoy sitting on soft surfaces",
        
        # Group 2: Python Programming (3, 4, 5)
        "Programming in Python is enjoyable",
        "Writing code with Python brings satisfaction",
        "Python development can be fun",
        
        # Group 3: Finance (6, 7, 8)
        "The stock market fluctuates daily",
        "Financial markets experience volatility",
        "Stock prices change frequently"
    ]
    
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2])
    
    # 2. Get real embeddings
    embeddings = get_bert_embeddings(sentences)
    print(f"Got {len(embeddings)} embeddings of shape {embeddings.shape}")
    
    # 3. Load trained hybrid encoder
    print("\nLoading trained Hybrid REWA model...")
    encoder = HybridREWAEncoder(d_model=768, m_dim=256, random_ratio=0.5)
    try:
        encoder.load_state_dict(torch.load('hybrid_rewa_best.pth'))
    except FileNotFoundError:
        print("Warning: 'hybrid_rewa_best.pth' not found, using 'hybrid_rewa_encoder.pth' or uninitialized if neither exists.")
        # Fallback for testing if file names differ
        try:
             encoder.load_state_dict(torch.load('hybrid_rewa_encoder.pth'))
        except:
            print("Using uninitialized model (for smoke test only if weights missing)")

    encoder.eval()
    
    # 4. Compute similarities
    print("\nComputing similarities...")
    with torch.no_grad():
        # Original BERT similarity
        orig_norm = F.normalize(embeddings, dim=-1)
        orig_sim = torch.mm(orig_norm, orig_norm.T)
        
        # REWA encoded similarity
        encoded = encoder(embeddings.unsqueeze(0), add_noise=False).squeeze(0)
        # Check normalization
        print(f"Encoded norm: {encoded.norm(dim=-1).mean():.4f}")
        
        rewa_sim = torch.mm(encoded, encoded.T)
    
    # 5. Analyze results
    print("\n" + "="*70)
    print("Similarity Analysis")
    print("="*70)
    
    groups = ["Cats", "Python", "Finance"]
    
    # Intra-group similarity (should be high)
    print("\nIntra-group Similarity (Target: High)")
    intra_sims = []
    intra_sims_orig = []
    
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            if labels[i] == labels[j]:
                print(f"  {groups[labels[i]]}: '{sentences[i][:20]}...' vs '{sentences[j][:20]}...'")
                print(f"    BERT: {orig_sim[i,j]:.3f} -> REWA: {rewa_sim[i,j]:.3f}")
                intra_sims.append(rewa_sim[i,j].item())
                intra_sims_orig.append(orig_sim[i,j].item())
    
    # Inter-group similarity (should be low)
    print("\nInter-group Similarity (Target: Low)")
    inter_sims = []
    inter_sims_orig = []
    
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            if labels[i] != labels[j]:
                # Only print a few to avoid clutter
                if i % 3 == 0 and j % 3 == 0: 
                    print(f"  {groups[labels[i]]} vs {groups[labels[j]]}")
                    print(f"    BERT: {orig_sim[i,j]:.3f} -> REWA: {rewa_sim[i,j]:.3f}")
                inter_sims.append(rewa_sim[i,j].item())
                inter_sims_orig.append(orig_sim[i,j].item())
                
    avg_intra = np.mean(intra_sims)
    avg_inter = np.mean(inter_sims)
    
    avg_intra_orig = np.mean(intra_sims_orig)
    avg_inter_orig = np.mean(inter_sims_orig)
    
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    print(f"{'Metric':<20} {'BERT (Original)':<20} {'Hybrid REWA':<20} {'Preserved':<10}")
    print("-"*70)
    print(f"{'Avg Intra-group':<20} {avg_intra_orig:<20.3f} {avg_intra:<20.3f} {avg_intra/avg_intra_orig:<10.1%}")
    print(f"{'Avg Inter-group':<20} {avg_inter_orig:<20.3f} {avg_inter:<20.3f} {avg_inter/avg_inter_orig:<10.1%}")
    print(f"{'Separation (Diff)':<20} {avg_intra_orig - avg_inter_orig:<20.3f} {avg_intra - avg_inter:<20.3f}")
    
    # Check ranking preservation
    print("\nRanking Preservation Check:")
    # For each query, is the top-1 neighbor (excluding self) in the same group?
    correct_top1 = 0
    for i in range(len(sentences)):
        # Get REWA neighbors
        scores = rewa_sim[i].clone()
        scores[i] = -1.0 # Exclude self
        top_idx = scores.argmax().item()
        
        is_correct = (labels[i] == labels[top_idx])
        correct_top1 += 1 if is_correct else 0
        
        status = "✅" if is_correct else "❌"
        print(f"  Query {i} ({groups[labels[i]]}): Top match {top_idx} ({groups[labels[top_idx]]}) {status}")

    print(f"\nTop-1 Accuracy: {correct_top1}/{len(sentences)} ({correct_top1/len(sentences):.1%})")
    
    if correct_top1 == len(sentences):
        print("\n✅ SUCCESS: Perfect ranking preservation on real sentences!")
    else:
        print("\n⚠️ WARNING: Ranking errors detected.")

if __name__ == "__main__":
    test_real_sentences()
