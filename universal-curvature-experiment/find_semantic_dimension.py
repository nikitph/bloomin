"""
Find Semantic Intrinsic Dimension using REWA
=============================================

Goal: Find the minimum dimension where REWA achieves 85%+ recall.
This is the TRUE semantic intrinsic dimension (vs PCA's variance-based dimension).

Test Model: DistilBERT
- PCA: 264D (95% variance) → 76.6% recall
- REWA: Find minimum D for 85% recall
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, '/Users/truckx/PycharmProjects/bloomin/adversarial-rewa-release')
from src.model import AdversarialHybridREWAEncoder
from src.utils import load_and_embed_data, split_categories, evaluate_recall

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def quick_train_rewa(train_emb, train_labels, val_emb, val_labels, m_dim, epochs=20):
    """Quick training to find if dimension works."""
    model = AdversarialHybridREWAEncoder(d_model=768, m_dim=m_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    
    best_val_recall = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        
        # Simple contrastive training (faster than full adversarial)
        for _ in range(20):
            idx = torch.randperm(len(train_emb))[:64]
            batch_emb = train_emb[idx]
            batch_labels = train_labels[idx]
            
            encoded = model(batch_emb.unsqueeze(0), add_noise=False).squeeze(0)
            encoded = F.normalize(encoded, dim=-1)
            
            # Contrastive loss
            sim_matrix = torch.matmul(encoded, encoded.T) / 0.07
            labels_expanded = batch_labels.unsqueeze(1)
            mask = (labels_expanded == labels_expanded.T).float()
            mask = mask.fill_diagonal_(0)
            
            num_positives = mask.sum(dim=1)
            exp_sim = torch.exp(sim_matrix)
            sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)
            log_prob = sim_matrix - torch.log(sum_exp_sim)
            loss = -(mask * log_prob).sum(dim=1) / num_positives.clamp(min=1)
            loss = loss.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Evaluate
        if epoch % 5 == 0:
            val_recall = evaluate_recall(model, val_emb, val_labels)
            if val_recall > best_val_recall:
                best_val_recall = val_recall
            print(f"  Epoch {epoch}: Val Recall = {val_recall:.1%}")
    
    return best_val_recall


def find_semantic_intrinsic_dimension():
    """Binary search to find minimum dimension for 85% recall."""
    print("="*70)
    print("FINDING SEMANTIC INTRINSIC DIMENSION")
    print("="*70)
    print("\nModel: DistilBERT")
    print("Target: 85% recall")
    print("Method: REWA with binary search\n")
    
    # Load data
    print("Loading 20 Newsgroups dataset...")
    embeddings, labels, target_names = load_and_embed_data()
    
    seen_emb, seen_labels, unseen_emb, unseen_labels = split_categories(
        embeddings, labels, target_names, n_unseen=5
    )
    
    # Use smaller subset for speed
    n_train = min(5000, len(seen_emb))
    indices = torch.randperm(len(seen_emb))[:n_train]
    train_emb = seen_emb[indices].to(DEVICE)
    train_labels = seen_labels[indices].to(DEVICE)
    
    # Validation set
    val_emb = unseen_emb[:500].to(DEVICE)
    val_labels = unseen_labels[:500].to(DEVICE)
    
    print(f"\nTrain: {len(train_emb)}, Val: {len(val_emb)}")
    
    # Test dimensions: [64, 96, 128, 192, 256, 384]
    dimensions = [64, 96, 128, 192, 256, 384]
    results = {}
    
    print("\n" + "="*70)
    print("Testing Different Dimensions")
    print("="*70)
    
    for m_dim in dimensions:
        print(f"\nTesting {m_dim}D (compression: {768/m_dim:.1f}x)...")
        recall = quick_train_rewa(train_emb, train_labels, val_emb, val_labels, m_dim, epochs=20)
        results[m_dim] = recall
        
        print(f"  → Best Recall: {recall:.1%}")
        
        if recall >= 0.85:
            print(f"  ✅ Target reached!")
    
    # Find semantic intrinsic dimension
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    semantic_dim = None
    for dim in sorted(results.keys()):
        recall = results[dim]
        status = "✅" if recall >= 0.85 else "❌"
        print(f"{dim:3d}D: {recall:.1%} {status}")
        
        if recall >= 0.85 and semantic_dim is None:
            semantic_dim = dim
    
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"\nPCA Intrinsic Dimension (95% variance):")
    print(f"  264D → 76.6% recall")
    print(f"  Conclusion: Variance-based, doesn't preserve semantics")
    
    if semantic_dim:
        print(f"\nREWA Semantic Intrinsic Dimension (85% recall):")
        print(f"  {semantic_dim}D → {results[semantic_dim]:.1%} recall")
        print(f"  Compression: {768/semantic_dim:.1f}x")
        print(f"  Conclusion: Task-aware, preserves semantic similarity")
        
        if semantic_dim < 264:
            print(f"\n✨ REWA finds LOWER dimension ({semantic_dim}D vs 264D PCA)")
            print(f"   PCA overestimates by {264-semantic_dim}D ({(264-semantic_dim)/264*100:.1f}%)")
        else:
            print(f"\n⚠️  REWA needs HIGHER dimension ({semantic_dim}D vs 264D PCA)")
            print(f"   PCA underestimates semantic complexity")
    else:
        print(f"\n⚠️  None of the tested dimensions reached 85% recall")
        print(f"   Semantic dimension > 384D")
    
    # Plot
    plot_results(results)
    
    return semantic_dim, results


def plot_results(results):
    """Plot dimension vs recall."""
    dims = sorted(results.keys())
    recalls = [results[d] for d in dims]
    
    plt.figure(figsize=(10, 6))
    plt.plot(dims, recalls, 'o-', linewidth=2, markersize=10, color='#2ecc71')
    plt.axhline(0.85, color='red', linestyle='--', linewidth=2, label='85% target')
    plt.axhline(0.766, color='orange', linestyle=':', linewidth=2, label='PCA (264D): 76.6%')
    
    plt.xlabel('Dimension', fontsize=12, fontweight='bold')
    plt.ylabel('Recall@10', fontsize=12, fontweight='bold')
    plt.title('Semantic Intrinsic Dimension (REWA)', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/semantic_intrinsic_dimension.png', dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved plot to results/semantic_intrinsic_dimension.png")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    semantic_dim, results = find_semantic_intrinsic_dimension()
