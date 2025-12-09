"""
GPT-2 Compression Validation Experiment
========================================

Test if compressing GPT-2 embeddings to 12D (95% variance) preserves retrieval quality.

Hypothesis: If 12D captures 95% variance, we should see 90%+ recall on retrieval tasks.
Potential: 64x compression with minimal quality loss = MASSIVE speedup!
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Reuse model loader from sweep
import sys
sys.path.append('.')
from model_loaders import load_model_and_get_embeddings
from corpus_loaders import load_sentences


def pca_compress(embeddings, n_components):
    """Compress embeddings using PCA."""
    # Normalize first (consistent with spherical geometry)
    X = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    pca = PCA(n_components=n_components)
    compressed = pca.fit_transform(X)
    
    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"  Compressed to {n_components}D, variance explained: {variance_explained:.1%}")
    
    return compressed, pca


def evaluate_retrieval(embeddings_full, embeddings_compressed, k=10, n_queries=1000):
    """
    Evaluate retrieval quality by comparing full vs compressed embeddings.
    
    Args:
        embeddings_full: Original embeddings (N, D_full)
        embeddings_compressed: Compressed embeddings (N, D_compressed)
        k: Number of neighbors to retrieve
        n_queries: Number of queries to test
        
    Returns:
        Dictionary with recall metrics
    """
    n_samples = len(embeddings_full)
    n_queries = min(n_queries, n_samples)
    
    # Normalize both
    emb_full_norm = embeddings_full / np.linalg.norm(embeddings_full, axis=1, keepdims=True)
    emb_comp_norm = embeddings_compressed / np.linalg.norm(embeddings_compressed, axis=1, keepdims=True)
    
    # Use random queries
    query_indices = np.random.choice(n_samples, n_queries, replace=False)
    
    recalls = []
    for query_idx in tqdm(query_indices, desc="Evaluating retrieval"):
        # Ground truth: top-k from full embeddings
        query_full = emb_full_norm[query_idx:query_idx+1]
        sims_full = cosine_similarity(query_full, emb_full_norm)[0]
        top_k_full = np.argsort(sims_full)[-k-1:-1][::-1]  # Exclude self
        
        # Compressed: top-k from compressed embeddings
        query_comp = emb_comp_norm[query_idx:query_idx+1]
        sims_comp = cosine_similarity(query_comp, emb_comp_norm)[0]
        top_k_comp = np.argsort(sims_comp)[-k-1:-1][::-1]  # Exclude self
        
        # Calculate recall
        recall = len(set(top_k_full) & set(top_k_comp)) / k
        recalls.append(recall)
    
    results = {
        'recall_mean': np.mean(recalls),
        'recall_std': np.std(recalls),
        'recall_min': np.min(recalls),
        'recall_max': np.max(recalls),
        'n_queries': n_queries,
        'k': k
    }
    
    return results


def plot_compression_vs_recall(embeddings, dimensions_to_test, output_path='./results'):
    """
    Test multiple compression levels and plot recall vs dimension.
    """
    recalls = []
    variances = []
    
    print("\nTesting multiple compression levels...")
    for n_components in dimensions_to_test:
        print(f"\nTesting {n_components}D compression:")
        compressed, pca = pca_compress(embeddings, n_components)
        variance = pca.explained_variance_ratio_.sum()
        
        results = evaluate_retrieval(embeddings, compressed, k=10, n_queries=500)
        
        recalls.append(results['recall_mean'])
        variances.append(variance)
        
        print(f"  Recall@10: {results['recall_mean']:.1%} ± {results['recall_std']:.1%}")
        print(f"  Variance: {variance:.1%}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Recall vs Dimension
    ax1.plot(dimensions_to_test, recalls, 'o-', linewidth=2, markersize=8)
    ax1.axhline(0.9, color='red', linestyle='--', label='90% recall target')
    ax1.set_xlabel('Compressed Dimension', fontsize=12)
    ax1.set_ylabel('Recall@10', fontsize=12)
    ax1.set_title('Retrieval Quality vs Compression', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend()
    
    # Variance vs Dimension
    ax2.plot(dimensions_to_test, variances, 'o-', linewidth=2, markersize=8, color='green')
    ax2.axhline(0.95, color='red', linestyle='--', label='95% variance')
    ax2.set_xlabel('Compressed Dimension', fontsize=12)
    ax2.set_ylabel('Variance Explained', fontsize=12)
    ax2.set_title('Variance Explained vs Dimension', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'gpt2_compression_validation.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved plot to {output_file}")
    
    return recalls, variances


def main():
    print("="*70)
    print("GPT-2 COMPRESSION VALIDATION EXPERIMENT")
    print("="*70)
    print("\nHypothesis: 12D compression (95% variance) preserves 90%+ retrieval quality")
    print("Potential: 64x speedup with minimal quality loss!\n")
    
    # Load GPT-2 embeddings
    print("Step 1: Loading GPT-2 embeddings...")
    model_config = {
        'source': 'transformers',
        'model': 'gpt2',
        'dim': 768
    }
    
    # Load Wikipedia sentences
    texts = load_sentences(n_samples=10000)
    print(f"  ✓ Loaded {len(texts)} sentences")
    
    # Get embeddings
    embeddings = load_model_and_get_embeddings(model_config, texts)
    print(f"  ✓ Embeddings shape: {embeddings.shape}")
    
    # Test at 12D (the intrinsic dimension we found)
    print("\n" + "="*70)
    print("Step 2: Testing 12D compression (intrinsic dimension)")
    print("="*70)
    
    compressed_12d, pca_12d = pca_compress(embeddings, n_components=12)
    results_12d = evaluate_retrieval(embeddings, compressed_12d, k=10, n_queries=1000)
    
    print(f"\n{'='*70}")
    print("RESULTS AT 12D (95% VARIANCE)")
    print(f"{'='*70}")
    print(f"Recall@10: {results_12d['recall_mean']:.1%} ± {results_12d['recall_std']:.1%}")
    print(f"Min recall: {results_12d['recall_min']:.1%}")
    print(f"Max recall: {results_12d['recall_max']:.1%}")
    print(f"Compression: 768D → 12D = {768/12:.1f}x")
    
    if results_12d['recall_mean'] >= 0.90:
        print("\n✅ HYPOTHESIS CONFIRMED! 90%+ recall achieved!")
        print("   → 64x compression with minimal quality loss is VIABLE!")
    else:
        print(f"\n⚠️  Recall below 90% target ({results_12d['recall_mean']:.1%})")
        print("   → May need higher dimension for production use")
    
    # Test multiple dimensions
    print("\n" + "="*70)
    print("Step 3: Testing multiple compression levels")
    print("="*70)
    
    dimensions_to_test = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
    recalls, variances = plot_compression_vs_recall(embeddings, dimensions_to_test)
    
    # Find optimal dimension for 90% recall
    for i, (dim, recall) in enumerate(zip(dimensions_to_test, recalls)):
        if recall >= 0.90:
            print(f"\n✅ Optimal dimension for 90% recall: {dim}D")
            print(f"   Compression: {768/dim:.1f}x")
            print(f"   Variance explained: {variances[i]:.1%}")
            break
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
