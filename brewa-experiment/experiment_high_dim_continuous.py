"""
High-Dimension Continuous REWA Test
====================================

Test continuous REWA with higher dimensions to achieve target 60-80% recall.

Hypothesis: Low recall (11%) is due to:
1. Too aggressive compression (256 → 16 = 16× is too much)
2. Need higher m_dim (64, 128, 256) for better preservation

Expected: With m_dim=128-256, should achieve 60-80% recall
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time

from continuous_rewa_encoder import ContinuousREWAEncoder


def generate_high_quality_data(n_tokens=1000, d_model=768, num_clusters=20):
    """Generate high-quality clustered data (like BERT embeddings)."""
    # Create well-separated cluster centers
    cluster_centers = torch.randn(num_clusters, d_model)
    cluster_centers = cluster_centers / cluster_centers.norm(dim=-1, keepdim=True)
    
    # Make clusters more separated
    cluster_centers = cluster_centers * 2.0
    
    # Assign tokens to clusters
    cluster_assignments = torch.randint(0, num_clusters, (n_tokens,))
    
    # Create embeddings with small intra-cluster variance
    embeddings = []
    for i in range(n_tokens):
        cluster_id = cluster_assignments[i]
        center = cluster_centers[cluster_id]
        # Small noise for tight clusters
        emb = center + 0.05 * torch.randn(d_model)  # Reduced from 0.15
        emb = emb / emb.norm()
        embeddings.append(emb)
    
    embeddings = torch.stack(embeddings)
    
    return embeddings, cluster_assignments


def create_queries(embeddings, cluster_assignments, num_queries=100):
    """Create retrieval queries."""
    n_tokens = len(embeddings)
    
    query_indices = torch.randint(0, n_tokens, (num_queries,))
    queries = embeddings[query_indices]
    
    ground_truth = []
    for qidx in query_indices:
        cluster_id = cluster_assignments[qidx]
        same_cluster = (cluster_assignments == cluster_id).nonzero(as_tuple=True)[0]
        same_cluster = same_cluster[same_cluster != qidx]
        
        if len(same_cluster) > 0:
            dists = (embeddings[same_cluster] - embeddings[qidx]).norm(dim=-1)
            closest_idx = same_cluster[dists.argmin()]
            ground_truth.append(closest_idx.item())
        else:
            ground_truth.append(qidx.item())
    
    ground_truth = torch.tensor(ground_truth)
    
    return queries, ground_truth


def test_continuous_rewa_high_dim(embeddings, queries, ground_truth, m_dim, k=10):
    """Test continuous REWA with higher dimensions."""
    d_model = embeddings.shape[1]
    
    # Create encoder
    encoder = ContinuousREWAEncoder(d_model, m_dim, noise_std=0.001)  # Lower noise
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
        emb_norm = F.normalize(emb_encoded, dim=-1)
        query_norm = F.normalize(query_encoded, dim=-1)
        
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
    }


def run_high_dim_experiment():
    """Test continuous REWA with various high dimensions."""
    
    print("="*70)
    print("High-Dimension Continuous REWA Experiment")
    print("="*70)
    
    # Generate high-quality data
    print("\nGenerating high-quality clustered data...")
    d_model = 768  # BERT dimension
    n_tokens = 1000
    num_queries = 100
    k = 10
    
    embeddings, cluster_assignments = generate_high_quality_data(n_tokens, d_model)
    queries, ground_truth = create_queries(embeddings, cluster_assignments, num_queries)
    
    print(f"Data: {n_tokens} tokens, {d_model} dims, {num_queries} queries")
    print(f"Clusters: {cluster_assignments.unique().numel()}")
    
    # Test baseline
    print("\n" + "="*70)
    print("Baseline (Standard Cosine Similarity)")
    print("="*70)
    
    start_time = time.time()
    with torch.no_grad():
        emb_norm = embeddings / embeddings.norm(dim=-1, keepdim=True)
        query_norm = queries / queries.norm(dim=-1, keepdim=True)
        
        similarity = torch.mm(query_norm, emb_norm.T)
    baseline_time = time.time() - start_time
    
    top_k_baseline = similarity.topk(k, dim=-1)[1]
    ground_truth_expanded = ground_truth.unsqueeze(1)
    matches_baseline = (top_k_baseline == ground_truth_expanded).any(dim=-1)
    recall_baseline = matches_baseline.float().mean().item()
    
    print(f"Recall@{k}: {recall_baseline:.3f}")
    print(f"Time: {baseline_time:.4f}s")
    
    # Test continuous REWA with different dimensions
    print("\n" + "="*70)
    print("Continuous REWA (Various Dimensions)")
    print("="*70)
    
    m_dims = [32, 64, 128, 256, 512]  # Power of 2 for Hadamard
    results = {}
    
    for m_dim in m_dims:
        print(f"\nTesting m_dim={m_dim}")
        results[m_dim] = test_continuous_rewa_high_dim(
            embeddings, queries, ground_truth, m_dim, k
        )
        print(f"  Recall@{k}: {results[m_dim]['recall']:.3f}")
        print(f"  Compression: {results[m_dim]['compression']:.1f}×")
        print(f"  Time: {results[m_dim]['total_time']:.4f}s")
    
    # Summary table
    print("\n" + "="*80)
    print("Summary: Continuous REWA Performance vs Dimension")
    print("="*80)
    print(f"{'m_dim':<10} {'Recall@10':<12} {'vs Baseline':<15} {'Compression':<15} {'Time (s)':<12}")
    print("-"*80)
    
    for m_dim in m_dims:
        res = results[m_dim]
        ratio = res['recall'] / recall_baseline if recall_baseline > 0 else 0
        print(f"{m_dim:<10} {res['recall']:<12.3f} {ratio:<15.1%} "
              f"{res['compression']:<15.1f}× {res['total_time']:<12.4f}")
    
    print("="*80)
    
    # Find best
    best_m_dim = max(results.items(), key=lambda x: x[1]['recall'])[0]
    best_recall = results[best_m_dim]['recall']
    
    print(f"\n🏆 Best Performance:")
    print(f"   m_dim = {best_m_dim}")
    print(f"   Recall@10 = {best_recall:.1%}")
    print(f"   Compression = {results[best_m_dim]['compression']:.1f}×")
    print(f"   vs Baseline = {best_recall/recall_baseline:.1%}")
    
    # Plot results
    plot_high_dim_results(results, recall_baseline, m_dims)
    
    return results, recall_baseline


def plot_high_dim_results(results, recall_baseline, m_dims):
    """Plot results."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    recalls = [results[m]['recall'] for m in m_dims]
    compressions = [results[m]['compression'] for m in m_dims]
    times = [results[m]['total_time'] for m in m_dims]
    ratios = [r / recall_baseline for r in recalls]
    
    # 1. Recall vs m_dim
    ax1.plot(m_dims, recalls, marker='o', linewidth=2, markersize=10, label='Continuous REWA')
    ax1.axhline(recall_baseline, color='red', linestyle='--', label='Baseline', linewidth=2)
    ax1.set_xlabel('m_dim')
    ax1.set_ylabel('Recall@10')
    ax1.set_title('Recall vs Dimension')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    
    # 2. Recall ratio vs m_dim
    ax2.plot(m_dims, ratios, marker='s', linewidth=2, markersize=10, color='green')
    ax2.axhline(1.0, color='red', linestyle='--', label='Equal to baseline', linewidth=2)
    ax2.set_xlabel('m_dim')
    ax2.set_ylabel('Recall Ratio (REWA / Baseline)')
    ax2.set_title('Relative Performance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    # 3. Compression vs Recall
    ax3.scatter(compressions, recalls, s=200, c=recalls, cmap='RdYlGn', vmin=0, vmax=1)
    for i, m in enumerate(m_dims):
        ax3.annotate(f'{m}', (compressions[i], recalls[i]),
                    xytext=(5, 5), textcoords='offset points')
    ax3.axhline(recall_baseline, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Compression Ratio')
    ax3.set_ylabel('Recall@10')
    ax3.set_title('Compression vs Accuracy Trade-off')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    
    # 4. Time vs m_dim
    ax4.plot(m_dims, times, marker='^', linewidth=2, markersize=10, color='purple')
    ax4.set_xlabel('m_dim')
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Encoding + Similarity Time')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log', base=2)
    
    plt.tight_layout()
    plt.savefig('high_dim_continuous_rewa.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: high_dim_continuous_rewa.png")
    plt.close()


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    results, baseline = run_high_dim_experiment()
    
    print("\n" + "="*70)
    print("High-Dimension Experiment Complete!")
    print("="*70)
