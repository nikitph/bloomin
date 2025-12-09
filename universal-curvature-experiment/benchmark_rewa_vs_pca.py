"""
Benchmark REWA vs PCA Compression
==================================

Compare Adversarial REWA against PCA for BERT and GPT-2 compression.
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

sys.path.insert(0, '/Users/truckx/PycharmProjects/bloomin/adversarial-rewa-release')
from src.model import AdversarialHybridREWAEncoder

from model_loaders import load_model_and_get_embeddings
from corpus_loaders import load_sentences

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def evaluate_retrieval_numpy(embeddings_full, embeddings_compressed, k=10, n_queries=1000):
    """Evaluate retrieval with numpy arrays."""
    n_samples = len(embeddings_full)
    n_queries = min(n_queries, n_samples)
    
    # Normalize
    emb_full_norm = embeddings_full / np.linalg.norm(embeddings_full, axis=1, keepdims=True)
    emb_comp_norm = embeddings_compressed / np.linalg.norm(embeddings_compressed, axis=1, keepdims=True)
    
    query_indices = np.random.choice(n_samples, n_queries, replace=False)
    
    recalls = []
    for query_idx in query_indices:
        # Ground truth
        query_full = emb_full_norm[query_idx:query_idx+1]
        sims_full = cosine_similarity(query_full, emb_full_norm)[0]
        top_k_full = np.argsort(sims_full)[-k-1:-1][::-1]
        
        # Compressed
        query_comp = emb_comp_norm[query_idx:query_idx+1]
        sims_comp = cosine_similarity(query_comp, emb_comp_norm)[0]
        top_k_comp = np.argsort(sims_comp)[-k-1:-1][::-1]
        
        recall = len(set(top_k_full) & set(top_k_comp)) / k
        recalls.append(recall)
    
    return np.mean(recalls)


def benchmark_model(model_name, embeddings_np, m_dim=256):
    """
    Benchmark both PCA and REWA for a model.
    
    Returns:
        dict with PCA and REWA results
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking {model_name.upper()}")
    print(f"{'='*70}")
    
    # PCA compression
    print(f"\n1. PCA Compression to {m_dim}D...")
    pca = PCA(n_components=m_dim)
    X_norm = embeddings_np / np.linalg.norm(embeddings_np, axis=1, keepdims=True)
    pca_compressed = pca.fit_transform(X_norm)
    pca_variance = pca.explained_variance_ratio_.sum()
    pca_recall = evaluate_retrieval_numpy(embeddings_np, pca_compressed, k=10, n_queries=500)
    
    print(f"  Variance explained: {pca_variance:.1%}")
    print(f"  Recall@10: {pca_recall:.1%}")
    
    # REWA compression
    print(f"\n2. REWA Compression to {m_dim}D...")
    checkpoint_path = f'checkpoints/rewa_{model_name}_{m_dim}d_best.pth'
    
    if os.path.exists(checkpoint_path):
        model = AdversarialHybridREWAEncoder(d_model=768, m_dim=m_dim).to(DEVICE)
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        model.eval()
        
        with torch.no_grad():
            embeddings_torch = torch.from_numpy(embeddings_np).float().to(DEVICE)
            rewa_compressed = model(embeddings_torch, add_noise=False).cpu().numpy()
        
        rewa_recall = evaluate_retrieval_numpy(embeddings_np, rewa_compressed, k=10, n_queries=500)
        print(f"  Recall@10: {rewa_recall:.1%}")
    else:
        print(f"  ⚠️  Checkpoint not found: {checkpoint_path}")
        print(f"  Run train_rewa_compression.py first!")
        rewa_recall = None
    
    return {
        'pca_variance': pca_variance,
        'pca_recall': pca_recall,
        'rewa_recall': rewa_recall,
        'improvement': (rewa_recall - pca_recall) if rewa_recall else None
    }


def plot_comparison(results, output_path='./results'):
    """Plot PCA vs REWA comparison."""
    models = list(results.keys())
    pca_recalls = [r['pca_recall'] for r in results.values()]
    rewa_recalls = [r['rewa_recall'] for r in results.values() if r['rewa_recall'] is not None]
    
    if len(rewa_recalls) != len(models):
        print("\n⚠️  Missing REWA results, skipping plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Comparison bar chart
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, pca_recalls, width, label='PCA', color='#e74c3c', alpha=0.7)
    bars2 = ax1.bar(x + width/2, rewa_recalls, width, label='REWA', color='#2ecc71', alpha=0.7)
    
    ax1.set_ylabel('Recall@10', fontsize=12, fontweight='bold')
    ax1.set_title('PCA vs REWA Compression (256D)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.upper() for m in models])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1.0)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Improvement chart
    improvements = [r['improvement'] * 100 for r in results.values()]
    colors = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvements]
    bars = ax2.barh(models, improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    ax2.axvline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Improvement (percentage points)', fontsize=12, fontweight='bold')
    ax2.set_title('REWA Improvement over PCA', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, improvements)):
        ax2.text(imp + (1 if imp > 0 else -1), i, f'{imp:+.1f}pp',
                ha='left' if imp > 0 else 'right', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'rewa_vs_pca_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved plot to {output_file}")


def main():
    print("="*70)
    print("REWA VS PCA BENCHMARK")
    print("="*70)
    
    # Load embeddings
    print("\nLoading embeddings...")
    texts = load_sentences(n_samples=5000)
    
    # BERT
    print("\nLoading BERT embeddings...")
    bert_config = {'source': 'transformers', 'model': 'bert-base-uncased', 'dim': 768}
    bert_embeddings = load_model_and_get_embeddings(bert_config, texts)
    
    # GPT-2
    print("\nLoading GPT-2 embeddings...")
    gpt2_config = {'source': 'transformers', 'model': 'gpt2', 'dim': 768}
    gpt2_embeddings = load_model_and_get_embeddings(gpt2_config, texts)
    
    # Benchmark
    results = {}
    results['bert'] = benchmark_model('bert', bert_embeddings, m_dim=256)
    results['gpt2'] = benchmark_model('gpt2', gpt2_embeddings, m_dim=256)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    for model, res in results.items():
        print(f"\n{model.upper()} (768D → 256D, 3x compression):")
        print(f"  PCA Recall@10:  {res['pca_recall']:.1%}")
        if res['rewa_recall']:
            print(f"  REWA Recall@10: {res['rewa_recall']:.1%}")
            print(f"  Improvement:    {res['improvement']*100:+.1f} percentage points")
        else:
            print(f"  REWA: Not trained yet")
    
    # Plot
    plot_comparison(results)
    
    print("\n" + "="*70)


if __name__ == "__main__":
    np.random.seed(42)
    main()
