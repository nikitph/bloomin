"""
Encoder Model Compression Validation
=====================================

Test if RoBERTa and DistilBERT compression at their intrinsic dimensions
preserves retrieval quality better than GPT-2.

Hypothesis: Encoder models have high VSA (Variance-Semantic Alignment)
- RoBERTa at 236D (95% variance) → 85%+ recall
- DistilBERT at 264D (95% variance) → 85%+ recall

If true: Confirms encoder models are fundamentally different from GPT-2
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# Reuse from previous experiment
import sys
sys.path.append('.')
from model_loaders import load_model_and_get_embeddings
from corpus_loaders import load_sentences


def pca_compress(embeddings, n_components):
    """Compress embeddings using PCA."""
    X = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    pca = PCA(n_components=n_components)
    compressed = pca.fit_transform(X)
    
    variance_explained = pca.explained_variance_ratio_.sum()
    
    return compressed, pca, variance_explained


def evaluate_retrieval(embeddings_full, embeddings_compressed, k=10, n_queries=1000):
    """Evaluate retrieval quality."""
    n_samples = len(embeddings_full)
    n_queries = min(n_queries, n_samples)
    
    # Normalize
    emb_full_norm = embeddings_full / np.linalg.norm(embeddings_full, axis=1, keepdims=True)
    emb_comp_norm = embeddings_compressed / np.linalg.norm(embeddings_compressed, axis=1, keepdims=True)
    
    query_indices = np.random.choice(n_samples, n_queries, replace=False)
    
    recalls = []
    for query_idx in tqdm(query_indices, desc="Evaluating retrieval"):
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
    
    return {
        'recall_mean': np.mean(recalls),
        'recall_std': np.std(recalls),
        'recall_min': np.min(recalls),
        'recall_max': np.max(recalls),
    }


def test_model_compression(model_name, model_config, intrinsic_dim, texts):
    """Test compression for a single model."""
    print(f"\n{'='*70}")
    print(f"Testing {model_name}")
    print(f"{'='*70}")
    
    # Get embeddings
    print(f"Extracting embeddings...")
    embeddings = load_model_and_get_embeddings(model_config, texts)
    print(f"  ✓ Embeddings shape: {embeddings.shape}")
    
    # Compress to intrinsic dimension
    print(f"\nCompressing to {intrinsic_dim}D (intrinsic dimension)...")
    compressed, pca, variance = pca_compress(embeddings, intrinsic_dim)
    print(f"  ✓ Variance explained: {variance:.1%}")
    
    # Evaluate retrieval
    print(f"\nEvaluating retrieval quality...")
    results = evaluate_retrieval(embeddings, compressed, k=10, n_queries=1000)
    
    print(f"\n{'='*70}")
    print(f"RESULTS FOR {model_name}")
    print(f"{'='*70}")
    print(f"Compression: 768D → {intrinsic_dim}D ({768/intrinsic_dim:.1f}x)")
    print(f"Variance explained: {variance:.1%}")
    print(f"Recall@10: {results['recall_mean']:.1%} ± {results['recall_std']:.1%}")
    print(f"Min recall: {results['recall_min']:.1%}")
    print(f"Max recall: {results['recall_max']:.1%}")
    
    # Check hypothesis
    if results['recall_mean'] >= 0.85:
        print(f"\n✅ HYPOTHESIS CONFIRMED! Recall ≥ 85%")
        print(f"   → High VSA: Variance aligns with semantic structure")
    else:
        print(f"\n⚠️  Recall below 85% target ({results['recall_mean']:.1%})")
        print(f"   → Lower VSA than expected")
    
    return results, variance


def plot_comparison(results_dict, output_path='./results'):
    """Plot comparison of all models."""
    models = list(results_dict.keys())
    recalls = [r['recall'] for r in results_dict.values()]
    variances = [r['variance'] for r in results_dict.values()]
    dims = [r['dim'] for r in results_dict.values()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Recall comparison
    colors = ['#2ecc71' if r >= 0.85 else '#e74c3c' for r in recalls]
    bars1 = ax1.bar(models, recalls, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax1.axhline(0.85, color='red', linestyle='--', linewidth=2, label='85% target')
    ax1.set_ylabel('Recall@10', fontsize=12, fontweight='bold')
    ax1.set_title('Retrieval Quality After Compression', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, recall in zip(bars1, recalls):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{recall:.1%}',
                ha='center', va='bottom', fontweight='bold')
    
    # Variance vs Recall scatter
    ax2.scatter(variances, recalls, s=200, alpha=0.7, edgecolor='black', linewidth=2)
    for i, model in enumerate(models):
        ax2.annotate(f'{model}\n({dims[i]}D)', 
                    (variances[i], recalls[i]),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    ax2.axhline(0.85, color='red', linestyle='--', alpha=0.5, label='85% recall')
    ax2.axvline(0.95, color='blue', linestyle='--', alpha=0.5, label='95% variance')
    ax2.set_xlabel('Variance Explained', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Recall@10', fontsize=12, fontweight='bold')
    ax2.set_title('Variance-Semantic Alignment (VSA)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'encoder_compression_validation.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved plot to {output_file}")


def main():
    print("="*70)
    print("ENCODER MODEL COMPRESSION VALIDATION")
    print("="*70)
    print("\nHypothesis: Encoder models have high VSA (Variance-Semantic Alignment)")
    print("Testing: RoBERTa (236D) and DistilBERT (264D)")
    print("Target: 85%+ recall at intrinsic dimension\n")
    
    # Load shared corpus
    print("Loading Wikipedia sentences...")
    texts = load_sentences(n_samples=10000)
    print(f"  ✓ Loaded {len(texts)} sentences\n")
    
    results_dict = {}
    
    # Test RoBERTa
    roberta_config = {
        'source': 'transformers',
        'model': 'roberta-base',
        'dim': 768
    }
    roberta_results, roberta_var = test_model_compression(
        'RoBERTa', roberta_config, 236, texts
    )
    results_dict['RoBERTa'] = {
        'recall': roberta_results['recall_mean'],
        'variance': roberta_var,
        'dim': 236
    }
    
    # Test DistilBERT
    distilbert_config = {
        'source': 'transformers',
        'model': 'distilbert-base-uncased',
        'dim': 768
    }
    distilbert_results, distilbert_var = test_model_compression(
        'DistilBERT', distilbert_config, 264, texts
    )
    results_dict['DistilBERT'] = {
        'recall': distilbert_results['recall_mean'],
        'variance': distilbert_var,
        'dim': 264
    }
    
    # Add GPT-2 for comparison (from previous experiment)
    results_dict['GPT-2'] = {
        'recall': 0.252,  # From previous experiment
        'variance': 0.953,
        'dim': 12
    }
    
    # Plot comparison
    plot_comparison(results_dict)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: VSA (Variance-Semantic Alignment) Analysis")
    print("="*70)
    
    for model, data in results_dict.items():
        vsa_score = data['recall'] / data['variance'] if data['variance'] > 0 else 0
        print(f"\n{model}:")
        print(f"  Dimension: {data['dim']}D")
        print(f"  Variance: {data['variance']:.1%}")
        print(f"  Recall: {data['recall']:.1%}")
        print(f"  VSA Score: {vsa_score:.2f} (recall/variance)")
        
        if data['recall'] >= 0.85:
            print(f"  ✅ High VSA - Good compression candidate")
        else:
            print(f"  ❌ Low VSA - Poor compression candidate")
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
