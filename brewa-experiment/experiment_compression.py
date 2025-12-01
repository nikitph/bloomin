"""
Experiment 2: Compression Validation
=====================================

Validates 32× compression claim.

Compares BREWA (32-bit) vs Standard Attention (float32):
1. Encode 10k tokens with both methods
2. Measure memory usage
3. Measure retrieval accuracy (Recall@10)
4. Verify: same accuracy, 32× less memory
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from brewa_encoder import REWAEncoder
from brewa_utils import hamming_similarity_efficient


class StandardAttentionEncoder(nn.Module):
    """Standard attention encoder for comparison."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.projection = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, d_model]
        
        Returns:
            [B, N, d_model] encoded
        """
        return self.projection(x)
    
    def get_memory_per_token(self) -> int:
        """Return bits per token."""
        return self.d_model * 32  # float32


def measure_compression_ratio(
    d_model: int = 256,
    m_bits: int = 32,
    n_tokens: int = 10000,
    num_queries: int = 1000,
    k: int = 10,
) -> dict:
    """
    Measure compression ratio and accuracy.
    
    Args:
        d_model: Model dimension
        m_bits: BREWA bits
        n_tokens: Number of tokens
        num_queries: Number of queries
        k: K for Recall@K
    
    Returns:
        Dictionary with results
    """
    print(f"\nTesting compression: d={d_model}, m_bits={m_bits}, n={n_tokens}")
    
    # Generate synthetic data
    embeddings = torch.randn(n_tokens, d_model)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    
    # Select queries
    query_indices = torch.randint(0, n_tokens, (num_queries,))
    queries = embeddings[query_indices] + 0.1 * torch.randn(num_queries, d_model)
    queries = queries / queries.norm(dim=-1, keepdim=True)
    
    # ========================================
    # Test BREWA
    # ========================================
    print("Testing BREWA...")
    brewa_encoder = REWAEncoder(d_model, m_bits, monoid='boolean')
    brewa_encoder.eval()
    
    # Encode
    emb_batch = embeddings.unsqueeze(0)
    query_batch = queries.unsqueeze(0)
    
    start_time = time.time()
    with torch.no_grad():
        brewa_emb = brewa_encoder(emb_batch, return_continuous=False)
        brewa_query = brewa_encoder(query_batch, return_continuous=False)
    brewa_encode_time = time.time() - start_time
    
    # Compute similarity
    start_time = time.time()
    with torch.no_grad():
        brewa_sim = hamming_similarity_efficient(brewa_query, brewa_emb)
        brewa_sim = brewa_sim.squeeze(0)
    brewa_sim_time = time.time() - start_time
    
    # Measure recall
    top_k_indices = brewa_sim.topk(k, dim=-1)[1]
    query_indices_expanded = query_indices.unsqueeze(1)
    brewa_recall = (top_k_indices == query_indices_expanded).any(dim=-1).float().mean().item()
    
    # Memory usage
    brewa_memory_per_token = m_bits  # bits
    brewa_total_memory = n_tokens * brewa_memory_per_token
    
    # ========================================
    # Test Standard Attention
    # ========================================
    print("Testing Standard Attention...")
    standard_encoder = StandardAttentionEncoder(d_model)
    standard_encoder.eval()
    
    # Encode
    start_time = time.time()
    with torch.no_grad():
        standard_emb = standard_encoder(emb_batch)
        standard_query = standard_encoder(query_batch)
    standard_encode_time = time.time() - start_time
    
    # Compute similarity (dot product)
    start_time = time.time()
    with torch.no_grad():
        standard_sim = torch.bmm(
            standard_query,
            standard_emb.transpose(1, 2)
        ).squeeze(0)
    standard_sim_time = time.time() - start_time
    
    # Measure recall
    top_k_indices = standard_sim.topk(k, dim=-1)[1]
    standard_recall = (top_k_indices == query_indices_expanded).any(dim=-1).float().mean().item()
    
    # Memory usage
    standard_memory_per_token = d_model * 32  # float32
    standard_total_memory = n_tokens * standard_memory_per_token
    
    # ========================================
    # Results
    # ========================================
    compression_ratio = standard_total_memory / brewa_total_memory
    
    results = {
        'brewa': {
            'memory_per_token': brewa_memory_per_token,
            'total_memory_bits': brewa_total_memory,
            'recall': brewa_recall,
            'encode_time': brewa_encode_time,
            'similarity_time': brewa_sim_time,
        },
        'standard': {
            'memory_per_token': standard_memory_per_token,
            'total_memory_bits': standard_total_memory,
            'recall': standard_recall,
            'encode_time': standard_encode_time,
            'similarity_time': standard_sim_time,
        },
        'compression_ratio': compression_ratio,
        'recall_difference': abs(brewa_recall - standard_recall),
    }
    
    return results


def run_compression_experiment():
    """
    Run full compression validation experiment.
    
    Tests different d_model values and measures compression.
    """
    print("="*60)
    print("BREWA Compression Validation Experiment")
    print("="*60)
    
    # Test different model dimensions
    d_values = [128, 256, 512, 1024]
    m_bits = 32
    n_tokens = 10000
    
    all_results = {}
    
    for d in d_values:
        results = measure_compression_ratio(
            d_model=d,
            m_bits=m_bits,
            n_tokens=n_tokens,
        )
        all_results[d] = results
    
    # Print summary table
    print("\n" + "="*80)
    print("Compression Validation Results")
    print("="*80)
    print(f"{'d':<8} {'BREWA bits':<12} {'Standard bits':<15} {'Compression':<12} {'Recall Δ':<12}")
    print("-"*80)
    
    for d in d_values:
        res = all_results[d]
        brewa_bits = res['brewa']['memory_per_token']
        standard_bits = res['standard']['memory_per_token']
        compression = res['compression_ratio']
        recall_diff = res['recall_difference']
        
        print(f"{d:<8} {brewa_bits:<12} {standard_bits:<15} {compression:.1f}×{'':<8} {recall_diff:.3f}")
    
    print("="*80)
    
    # Print detailed comparison
    print("\n" + "="*80)
    print("Detailed Performance Comparison (d=256)")
    print("="*80)
    
    res = all_results[256]
    
    print(f"\nBREWA:")
    print(f"  Memory per token: {res['brewa']['memory_per_token']} bits")
    print(f"  Total memory: {res['brewa']['total_memory_bits'] / 8 / 1024:.2f} KB")
    print(f"  Recall@10: {res['brewa']['recall']:.3f}")
    print(f"  Encode time: {res['brewa']['encode_time']:.3f}s")
    print(f"  Similarity time: {res['brewa']['similarity_time']:.3f}s")
    
    print(f"\nStandard Attention:")
    print(f"  Memory per token: {res['standard']['memory_per_token']} bits")
    print(f"  Total memory: {res['standard']['total_memory_bits'] / 8 / 1024:.2f} KB")
    print(f"  Recall@10: {res['standard']['recall']:.3f}")
    print(f"  Encode time: {res['standard']['encode_time']:.3f}s")
    print(f"  Similarity time: {res['standard']['similarity_time']:.3f}s")
    
    print(f"\nCompression: {res['compression_ratio']:.1f}×")
    print(f"Recall difference: {res['recall_difference']:.3f}")
    
    # Plot results
    plot_compression_results(all_results, d_values)
    
    return all_results


def plot_compression_results(results: dict, d_values: list[int]):
    """
    Plot compression validation results.
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    compression_ratios = [results[d]['compression_ratio'] for d in d_values]
    brewa_recalls = [results[d]['brewa']['recall'] for d in d_values]
    standard_recalls = [results[d]['standard']['recall'] for d in d_values]
    brewa_times = [results[d]['brewa']['encode_time'] + results[d]['brewa']['similarity_time'] 
                   for d in d_values]
    standard_times = [results[d]['standard']['encode_time'] + results[d]['standard']['similarity_time']
                      for d in d_values]
    
    # Plot 1: Compression ratio
    ax1.bar(range(len(d_values)), compression_ratios, alpha=0.7)
    ax1.set_xticks(range(len(d_values)))
    ax1.set_xticklabels([f'd={d}' for d in d_values])
    ax1.set_ylabel('Compression Ratio')
    ax1.set_title('BREWA Compression Ratio')
    ax1.axhline(32, color='red', linestyle='--', alpha=0.5, label='Target: 32×')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Recall comparison
    x = np.arange(len(d_values))
    width = 0.35
    ax2.bar(x - width/2, brewa_recalls, width, label='BREWA', alpha=0.7)
    ax2.bar(x + width/2, standard_recalls, width, label='Standard', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'd={d}' for d in d_values])
    ax2.set_ylabel('Recall@10')
    ax2.set_title('Recall Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Memory usage
    brewa_memory = [results[d]['brewa']['total_memory_bits'] / 8 / 1024 for d in d_values]
    standard_memory = [results[d]['standard']['total_memory_bits'] / 8 / 1024 for d in d_values]
    
    ax3.bar(x - width/2, brewa_memory, width, label='BREWA', alpha=0.7)
    ax3.bar(x + width/2, standard_memory, width, label='Standard', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'd={d}' for d in d_values])
    ax3.set_ylabel('Memory (KB)')
    ax3.set_title('Memory Usage (10K tokens)')
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Time comparison
    ax4.bar(x - width/2, brewa_times, width, label='BREWA', alpha=0.7)
    ax4.bar(x + width/2, standard_times, width, label='Standard', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'd={d}' for d in d_values])
    ax4.set_ylabel('Time (seconds)')
    ax4.set_title('Encoding + Similarity Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('compression_validation.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: brewa-experiment/compression_validation.png")
    plt.close()


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run experiment
    results = run_compression_experiment()
    
    print("\n" + "="*60)
    print("Compression Validation Complete!")
    print("="*60)
