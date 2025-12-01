"""
Parameter Sweep Experiment
===========================

Systematically test different combinations of:
- m_bits: [16, 32, 64, 128, 256]
- noise_std: [0.001, 0.01, 0.05, 0.1, 0.2]

To find optimal parameters for BREWA encoding.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from brewa_encoder import REWAEncoder
from brewa_utils import hamming_similarity_efficient


def generate_test_data(n_tokens=1000, d_model=256, num_queries=100):
    """Generate test data with clear structure."""
    # Create clustered embeddings for better structure
    num_clusters = 10
    cluster_centers = torch.randn(num_clusters, d_model)
    cluster_centers = cluster_centers / cluster_centers.norm(dim=-1, keepdim=True)
    
    # Assign tokens to clusters
    cluster_assignments = torch.randint(0, num_clusters, (n_tokens,))
    
    embeddings = []
    for i in range(n_tokens):
        cluster_id = cluster_assignments[i]
        center = cluster_centers[cluster_id]
        # Add small noise
        emb = center + 0.1 * torch.randn(d_model)
        emb = emb / emb.norm()
        embeddings.append(emb)
    
    embeddings = torch.stack(embeddings)
    
    # Queries: select random tokens
    query_indices = torch.randint(0, n_tokens, (num_queries,))
    queries = embeddings[query_indices]
    
    # Ground truth: find another token in same cluster
    ground_truth = []
    for qidx in query_indices:
        cluster_id = cluster_assignments[qidx]
        same_cluster = (cluster_assignments == cluster_id).nonzero(as_tuple=True)[0]
        same_cluster = same_cluster[same_cluster != qidx]
        if len(same_cluster) > 0:
            ground_truth.append(same_cluster[0].item())
        else:
            ground_truth.append(qidx.item())
    
    ground_truth = torch.tensor(ground_truth)
    
    return embeddings, queries, ground_truth


def test_parameters(m_bits, noise_std, d_model=256, n_tokens=1000, k=10):
    """Test a specific parameter combination."""
    # Generate data
    embeddings, queries, ground_truth = generate_test_data(n_tokens, d_model)
    
    # Create encoder
    encoder = REWAEncoder(d_model, m_bits, monoid='boolean', noise_std=noise_std)
    encoder.eval()
    
    # Encode
    emb_batch = embeddings.unsqueeze(0)
    query_batch = queries.unsqueeze(0)
    
    start_time = time.time()
    with torch.no_grad():
        emb_encoded = encoder(emb_batch, return_continuous=False, add_noise=False)
        query_encoded = encoder(query_batch, return_continuous=False, add_noise=False)
    encode_time = time.time() - start_time
    
    # Compute similarity
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
    }


def run_parameter_sweep():
    """Run full parameter sweep."""
    print("="*60)
    print("BREWA Parameter Sweep")
    print("="*60)
    
    # Parameter ranges
    m_bits_values = [16, 32, 64, 128, 256]
    noise_std_values = [0.001, 0.01, 0.05, 0.1, 0.2]
    
    d_model = 256
    n_tokens = 1000
    k = 10
    
    print(f"\nTesting {len(m_bits_values)} × {len(noise_std_values)} = {len(m_bits_values) * len(noise_std_values)} combinations")
    print(f"d_model={d_model}, n_tokens={n_tokens}, k={k}")
    print()
    
    # Store results
    results = {}
    
    # Run sweep
    total_tests = len(m_bits_values) * len(noise_std_values)
    pbar = tqdm(total=total_tests, desc="Parameter sweep")
    
    for m_bits in m_bits_values:
        for noise_std in noise_std_values:
            key = (m_bits, noise_std)
            
            # Run test 3 times and average
            recalls = []
            times = []
            
            for trial in range(3):
                result = test_parameters(m_bits, noise_std, d_model, n_tokens, k)
                recalls.append(result['recall'])
                times.append(result['total_time'])
            
            results[key] = {
                'recall_mean': np.mean(recalls),
                'recall_std': np.std(recalls),
                'time_mean': np.mean(times),
                'time_std': np.std(times),
            }
            
            pbar.update(1)
    
    pbar.close()
    
    # Print results table
    print("\n" + "="*100)
    print("Parameter Sweep Results")
    print("="*100)
    print(f"{'m_bits':<10} {'noise_std':<12} {'Recall@10':<15} {'Time (s)':<15} {'Compression':<15}")
    print("-"*100)
    
    # Sort by recall (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]['recall_mean'], reverse=True)
    
    for (m_bits, noise_std), res in sorted_results:
        compression = (d_model * 32) / m_bits
        print(f"{m_bits:<10} {noise_std:<12.3f} {res['recall_mean']:<15.3f} "
              f"{res['time_mean']:<15.4f} {compression:<15.1f}×")
    
    print("="*100)
    
    # Find best parameters
    best_key = max(results.items(), key=lambda x: x[1]['recall_mean'])[0]
    best_m_bits, best_noise_std = best_key
    best_recall = results[best_key]['recall_mean']
    
    print(f"\n🏆 Best Parameters:")
    print(f"   m_bits = {best_m_bits}")
    print(f"   noise_std = {best_noise_std}")
    print(f"   Recall@10 = {best_recall:.3f}")
    print(f"   Compression = {(d_model * 32) / best_m_bits:.1f}×")
    
    # Plot results
    plot_parameter_sweep(results, m_bits_values, noise_std_values)
    
    return results, best_key


def plot_parameter_sweep(results, m_bits_values, noise_std_values):
    """Create visualization of parameter sweep results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Heatmap of recall vs parameters
    recall_grid = np.zeros((len(noise_std_values), len(m_bits_values)))
    
    for i, noise_std in enumerate(noise_std_values):
        for j, m_bits in enumerate(m_bits_values):
            recall_grid[i, j] = results[(m_bits, noise_std)]['recall_mean']
    
    im1 = ax1.imshow(recall_grid, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(range(len(m_bits_values)))
    ax1.set_xticklabels(m_bits_values)
    ax1.set_yticks(range(len(noise_std_values)))
    ax1.set_yticklabels([f"{x:.3f}" for x in noise_std_values])
    ax1.set_xlabel('m_bits')
    ax1.set_ylabel('noise_std')
    ax1.set_title('Recall@10 Heatmap')
    plt.colorbar(im1, ax=ax1)
    
    # Add text annotations
    for i in range(len(noise_std_values)):
        for j in range(len(m_bits_values)):
            text = ax1.text(j, i, f'{recall_grid[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    # 2. Recall vs m_bits (for different noise levels)
    for noise_std in noise_std_values:
        recalls = [results[(m, noise_std)]['recall_mean'] for m in m_bits_values]
        ax2.plot(m_bits_values, recalls, marker='o', label=f'noise={noise_std:.3f}')
    
    ax2.set_xlabel('m_bits')
    ax2.set_ylabel('Recall@10')
    ax2.set_title('Recall vs m_bits')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    # 3. Recall vs noise_std (for different m_bits)
    for m_bits in m_bits_values:
        recalls = [results[(m_bits, n)]['recall_mean'] for n in noise_std_values]
        ax3.plot(noise_std_values, recalls, marker='o', label=f'm_bits={m_bits}')
    
    ax3.set_xlabel('noise_std')
    ax3.set_ylabel('Recall@10')
    ax3.set_title('Recall vs noise_std')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # 4. Compression vs Recall (Pareto frontier)
    d_model = 256
    compressions = []
    recalls = []
    labels = []
    
    for (m_bits, noise_std), res in results.items():
        compression = (d_model * 32) / m_bits
        compressions.append(compression)
        recalls.append(res['recall_mean'])
        labels.append(f"{m_bits}b, {noise_std:.3f}n")
    
    scatter = ax4.scatter(compressions, recalls, c=recalls, cmap='RdYlGn', 
                         s=100, alpha=0.6, vmin=0, vmax=1)
    ax4.set_xlabel('Compression Ratio')
    ax4.set_ylabel('Recall@10')
    ax4.set_title('Compression vs Accuracy Trade-off')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log', base=2)
    plt.colorbar(scatter, ax=ax4)
    
    # Annotate best point
    best_idx = np.argmax(recalls)
    ax4.annotate(labels[best_idx], 
                (compressions[best_idx], recalls[best_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig('parameter_sweep_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: parameter_sweep_results.png")
    plt.close()


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run parameter sweep
    results, best_params = run_parameter_sweep()
    
    print("\n" + "="*60)
    print("Parameter Sweep Complete!")
    print("="*60)
    
    # Save results
    import json
    
    # Convert results to JSON-serializable format
    results_json = {}
    for (m_bits, noise_std), res in results.items():
        key = f"m{m_bits}_n{noise_std:.3f}"
        results_json[key] = {
            'm_bits': m_bits,
            'noise_std': noise_std,
            **res
        }
    
    with open('parameter_sweep_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("\nResults saved to: parameter_sweep_results.json")
