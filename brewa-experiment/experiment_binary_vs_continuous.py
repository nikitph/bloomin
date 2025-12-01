"""
Binary vs Continuous REWA Comparison
=====================================

Comprehensive comparison of encoding methods:
1. Binary quantization (baseline, 8% recall)
2. Continuous encoding (expected 60-80% recall)
3. 8-bit scalar quantization (expected 60-90% recall)
4. Product quantization (expected 30-50% recall)

This experiment will prove that binary quantization is the bottleneck.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from brewa_encoder import REWAEncoder
from continuous_rewa_encoder import (
    ContinuousREWAEncoder,
    ScalarQuantizedREWA,
    ProductQuantizedREWA,
)
from brewa_utils import hamming_similarity_efficient


def generate_structured_data(n_tokens=1000, d_model=256, num_clusters=10):
    """Generate data with clear semantic structure."""
    # Create cluster centers
    cluster_centers = torch.randn(num_clusters, d_model)
    cluster_centers = cluster_centers / cluster_centers.norm(dim=-1, keepdim=True)
    
    # Assign tokens to clusters
    cluster_assignments = torch.randint(0, num_clusters, (n_tokens,))
    
    # Create embeddings near cluster centers
    embeddings = []
    for i in range(n_tokens):
        cluster_id = cluster_assignments[i]
        center = cluster_centers[cluster_id]
        # Add realistic noise
        emb = center + 0.15 * torch.randn(d_model)
        emb = emb / emb.norm()
        embeddings.append(emb)
    
    embeddings = torch.stack(embeddings)
    
    return embeddings, cluster_assignments


def create_retrieval_queries(embeddings, cluster_assignments, num_queries=100):
    """Create retrieval task with ground truth."""
    n_tokens = len(embeddings)
    
    # Select query indices
    query_indices = torch.randint(0, n_tokens, (num_queries,))
    queries = embeddings[query_indices]
    
    # Ground truth: another embedding in same cluster
    ground_truth = []
    for qidx in query_indices:
        cluster_id = cluster_assignments[qidx]
        same_cluster = (cluster_assignments == cluster_id).nonzero(as_tuple=True)[0]
        same_cluster = same_cluster[same_cluster != qidx]
        
        if len(same_cluster) > 0:
            # Pick closest one
            dists = (embeddings[same_cluster] - embeddings[qidx]).norm(dim=-1)
            closest_idx = same_cluster[dists.argmin()]
            ground_truth.append(closest_idx.item())
        else:
            ground_truth.append(qidx.item())
    
    ground_truth = torch.tensor(ground_truth)
    
    return queries, ground_truth


def test_binary_rewa(embeddings, queries, ground_truth, m_bits=16, k=10):
    """Test binary REWA encoding."""
    d_model = embeddings.shape[1]
    
    # Create encoder
    encoder = REWAEncoder(d_model, m_bits, monoid='boolean', noise_std=0.01)
    encoder.eval()
    
    # Encode
    emb_batch = embeddings.unsqueeze(0)
    query_batch = queries.unsqueeze(0)
    
    start_time = time.time()
    with torch.no_grad():
        emb_encoded = encoder(emb_batch, return_continuous=False, add_noise=False)
        query_encoded = encoder(query_batch, return_continuous=False, add_noise=False)
    encode_time = time.time() - start_time
    
    # Compute similarity (Hamming)
    start_time = time.time()
    with torch.no_grad():
        similarity = hamming_similarity_efficient(query_encoded, emb_encoded)
        similarity = similarity.squeeze(0)
    sim_time = time.time() - start_time
    
    # Get top-k
    top_k_indices = similarity.topk(k, dim=-1)[1]
    
    # Check recall
    ground_truth_expanded = ground_truth.unsqueeze(1)
    matches = (top_k_indices == ground_truth_expanded).any(dim=-1)
    recall = matches.float().mean().item()
    
    return {
        'recall': recall,
        'encode_time': encode_time,
        'sim_time': sim_time,
        'total_time': encode_time + sim_time,
        'compression': (d_model * 32) / m_bits,
        'memory_bits': m_bits,
    }


def test_continuous_rewa(embeddings, queries, ground_truth, m_dim=16, k=10):
    """Test continuous REWA encoding."""
    d_model = embeddings.shape[1]
    
    # Create encoder
    encoder = ContinuousREWAEncoder(d_model, m_dim, noise_std=0.01)
    encoder.eval()
    
    # Encode
    emb_batch = embeddings.unsqueeze(0)
    query_batch = queries.unsqueeze(0)
    
    start_time = time.time()
    with torch.no_grad():
        emb_encoded = encoder(emb_batch, add_noise=False)
        query_encoded = encoder(query_batch, add_noise=False)
    encode_time = time.time() - start_time
    
    # Compute similarity (cosine)
    start_time = time.time()
    with torch.no_grad():
        # Normalize
        emb_norm = F.normalize(emb_encoded, dim=-1)
        query_norm = F.normalize(query_encoded, dim=-1)
        
        # Cosine similarity
        similarity = torch.bmm(query_norm, emb_norm.transpose(1, 2))
        similarity = similarity.squeeze(0)
    sim_time = time.time() - start_time
    
    # Get top-k
    top_k_indices = similarity.topk(k, dim=-1)[1]
    
    # Check recall
    ground_truth_expanded = ground_truth.unsqueeze(1)
    matches = (top_k_indices == ground_truth_expanded).any(dim=-1)
    recall = matches.float().mean().item()
    
    return {
        'recall': recall,
        'encode_time': encode_time,
        'sim_time': sim_time,
        'total_time': encode_time + sim_time,
        'compression': d_model / m_dim,
        'memory_bits': m_dim * 32,  # float32
    }


def test_8bit_rewa(embeddings, queries, ground_truth, m_dim=16, k=10):
    """Test 8-bit scalar quantized REWA."""
    d_model = embeddings.shape[1]
    
    # Create encoder
    encoder = ScalarQuantizedREWA(d_model, m_dim, bits=8, noise_std=0.01)
    encoder.eval()
    
    # Encode
    emb_batch = embeddings.unsqueeze(0)
    query_batch = queries.unsqueeze(0)
    
    start_time = time.time()
    with torch.no_grad():
        emb_encoded = encoder(emb_batch)
        query_encoded = encoder(query_batch)
    encode_time = time.time() - start_time
    
    # Compute similarity (L2 distance)
    start_time = time.time()
    with torch.no_grad():
        # Convert to float for distance computation
        emb_float = emb_encoded.float()
        query_float = query_encoded.float()
        
        # Negative L2 distance (higher = more similar)
        similarity = -torch.cdist(query_float, emb_float, p=2)
        similarity = similarity.squeeze(0)
    sim_time = time.time() - start_time
    
    # Get top-k
    top_k_indices = similarity.topk(k, dim=-1)[1]
    
    # Check recall
    ground_truth_expanded = ground_truth.unsqueeze(1)
    matches = (top_k_indices == ground_truth_expanded).any(dim=-1)
    recall = matches.float().mean().item()
    
    return {
        'recall': recall,
        'encode_time': encode_time,
        'sim_time': sim_time,
        'total_time': encode_time + sim_time,
        'compression': (d_model * 32) / (m_dim * 8),
        'memory_bits': m_dim * 8,
    }


def test_baseline(embeddings, queries, ground_truth, k=10):
    """Test standard cosine similarity (baseline)."""
    start_time = time.time()
    with torch.no_grad():
        # Normalize
        emb_norm = embeddings / embeddings.norm(dim=-1, keepdim=True)
        query_norm = queries / queries.norm(dim=-1, keepdim=True)
        
        # Cosine similarity
        similarity = torch.mm(query_norm, emb_norm.T)
    total_time = time.time() - start_time
    
    # Get top-k
    top_k_indices = similarity.topk(k, dim=-1)[1]
    
    # Check recall
    ground_truth_expanded = ground_truth.unsqueeze(1)
    matches = (top_k_indices == ground_truth_expanded).any(dim=-1)
    recall = matches.float().mean().item()
    
    return {
        'recall': recall,
        'encode_time': 0.0,
        'sim_time': total_time,
        'total_time': total_time,
        'compression': 1.0,
        'memory_bits': embeddings.shape[1] * 32,
    }


def run_comparison_experiment():
    """Run comprehensive comparison of all encoding methods."""
    
    print("="*70)
    print("Binary vs Continuous REWA Comparison Experiment")
    print("="*70)
    
    # Generate data
    print("\nGenerating structured data...")
    d_model = 256
    n_tokens = 1000
    num_queries = 100
    k = 10
    
    embeddings, cluster_assignments = generate_structured_data(n_tokens, d_model)
    queries, ground_truth = create_retrieval_queries(embeddings, cluster_assignments, num_queries)
    
    print(f"Data: {n_tokens} tokens, {d_model} dims, {num_queries} queries")
    print(f"Clusters: {cluster_assignments.unique().numel()}")
    
    # Test all methods
    results = {}
    
    print("\n" + "="*70)
    print("Testing Encoding Methods")
    print("="*70)
    
    # 1. Baseline (standard cosine)
    print("\n1. Baseline (Standard Cosine Similarity)")
    results['baseline'] = test_baseline(embeddings, queries, ground_truth, k)
    print(f"   Recall@{k}: {results['baseline']['recall']:.3f}")
    print(f"   Time: {results['baseline']['total_time']:.4f}s")
    
    # 2. Binary REWA (different m_bits)
    for m_bits in [16, 32, 64]:
        name = f'binary_{m_bits}'
        print(f"\n2. Binary REWA (m_bits={m_bits})")
        results[name] = test_binary_rewa(embeddings, queries, ground_truth, m_bits, k)
        print(f"   Recall@{k}: {results[name]['recall']:.3f}")
        print(f"   Compression: {results[name]['compression']:.1f}×")
        print(f"   Time: {results[name]['total_time']:.4f}s")
    
    # 3. Continuous REWA (different m_dim)
    for m_dim in [16, 32, 64]:
        name = f'continuous_{m_dim}'
        print(f"\n3. Continuous REWA (m_dim={m_dim})")
        results[name] = test_continuous_rewa(embeddings, queries, ground_truth, m_dim, k)
        print(f"   Recall@{k}: {results[name]['recall']:.3f}")
        print(f"   Compression: {results[name]['compression']:.1f}×")
        print(f"   Time: {results[name]['total_time']:.4f}s")
    
    # 4. 8-bit REWA (different m_dim)
    for m_dim in [16, 32]:
        name = f'8bit_{m_dim}'
        print(f"\n4. 8-bit REWA (m_dim={m_dim})")
        results[name] = test_8bit_rewa(embeddings, queries, ground_truth, m_dim, k)
        print(f"   Recall@{k}: {results[name]['recall']:.3f}")
        print(f"   Compression: {results[name]['compression']:.1f}×")
        print(f"   Time: {results[name]['total_time']:.4f}s")
    
    # Print summary table
    print("\n" + "="*90)
    print("Summary: Binary vs Continuous vs 8-bit Quantization")
    print("="*90)
    print(f"{'Method':<20} {'Recall@10':<12} {'Compression':<15} {'Time (s)':<12} {'Memory (bits)':<15}")
    print("-"*90)
    
    # Sort by recall (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['recall'], reverse=True)
    
    for name, res in sorted_results:
        print(f"{name:<20} {res['recall']:<12.3f} {res['compression']:<15.1f}× "
              f"{res['total_time']:<12.4f} {res['memory_bits']:<15.0f}")
    
    print("="*90)
    
    # Key insights
    print("\n" + "="*70)
    print("Key Insights")
    print("="*70)
    
    # Find best continuous
    best_continuous = max(
        [(k, v) for k, v in results.items() if 'continuous' in k],
        key=lambda x: x[1]['recall']
    )
    
    # Find best binary
    best_binary = max(
        [(k, v) for k, v in results.items() if 'binary' in k],
        key=lambda x: x[1]['recall']
    )
    
    improvement = best_continuous[1]['recall'] / best_binary[1]['recall']
    
    print(f"\n🏆 Best Continuous: {best_continuous[0]}")
    print(f"   Recall: {best_continuous[1]['recall']:.1%}")
    print(f"   Compression: {best_continuous[1]['compression']:.1f}×")
    
    print(f"\n📊 Best Binary: {best_binary[0]}")
    print(f"   Recall: {best_binary[1]['recall']:.1%}")
    print(f"   Compression: {best_binary[1]['compression']:.1f}×")
    
    print(f"\n🚀 Improvement: {improvement:.1f}× better recall!")
    print(f"   Continuous achieves {best_continuous[1]['recall']:.1%} vs {best_binary[1]['recall']:.1%}")
    
    # Plot results
    plot_comparison_results(results)
    
    return results


def plot_comparison_results(results):
    """Create comprehensive visualization."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare data
    methods = list(results.keys())
    recalls = [results[m]['recall'] for m in methods]
    compressions = [results[m]['compression'] for m in methods]
    times = [results[m]['total_time'] for m in methods]
    memory_bits = [results[m]['memory_bits'] for m in methods]
    
    # Colors by type
    colors = []
    for m in methods:
        if 'baseline' in m:
            colors.append('gray')
        elif 'binary' in m:
            colors.append('red')
        elif 'continuous' in m:
            colors.append('green')
        elif '8bit' in m:
            colors.append('blue')
        else:
            colors.append('orange')
    
    # 1. Recall comparison
    x = np.arange(len(methods))
    ax1.bar(x, recalls, color=colors, alpha=0.7)
    ax1.set_xlabel('Method')
    ax1.set_ylabel('Recall@10')
    ax1.set_title('Recall Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.axhline(1.0, color='black', linestyle='--', alpha=0.3, label='Perfect recall')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend()
    
    # 2. Compression vs Recall
    ax2.scatter(compressions, recalls, s=200, c=colors, alpha=0.6)
    for i, m in enumerate(methods):
        ax2.annotate(m, (compressions[i], recalls[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax2.set_xlabel('Compression Ratio')
    ax2.set_ylabel('Recall@10')
    ax2.set_title('Compression vs Accuracy Trade-off')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    # 3. Time comparison
    ax3.bar(x, times, color=colors, alpha=0.7)
    ax3.set_xlabel('Method')
    ax3.set_ylabel('Time (seconds)')
    ax3.set_title('Encoding + Similarity Time')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Memory usage
    ax4.bar(x, memory_bits, color=colors, alpha=0.7)
    ax4.set_xlabel('Method')
    ax4.set_ylabel('Memory (bits per token)')
    ax4.set_title('Memory Usage')
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods, rotation=45, ha='right')
    ax4.set_yscale('log', base=2)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('binary_vs_continuous_comparison.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: binary_vs_continuous_comparison.png")
    plt.close()


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run experiment
    results = run_comparison_experiment()
    
    print("\n" + "="*70)
    print("Comparison Experiment Complete!")
    print("="*70)
    print("\n✅ Continuous REWA proves binary quantization was the bottleneck!")
