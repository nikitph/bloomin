"""
Experiment 1: Capacity Validation
==================================

Validates Theorem 6.1: n_max ≈ exp(√d)

For varying d ∈ {64, 128, 256, 512}:
1. Create sequences of increasing length n
2. Measure recall@k for witness matching
3. Find n_max where recall drops below 90%
4. Verify: n_max ≈ exp(√d)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

from brewa_encoder import REWAEncoder
from brewa_utils import hamming_similarity_efficient, max_context_length


def generate_synthetic_data(
    n_tokens: int,
    d_model: int,
    num_queries: int = 100,
    gap: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate synthetic data for capacity testing.
    
    Creates:
    - Random token embeddings
    - Query tokens (subset of tokens)
    - Ground truth matches
    
    Args:
        n_tokens: Number of tokens in sequence
        d_model: Embedding dimension
        num_queries: Number of query tokens
        gap: Separation between clusters
    
    Returns:
        embeddings: [n_tokens, d_model]
        queries: [num_queries, d_model]
        ground_truth: [num_queries] indices of matches
    """
    # Generate random embeddings
    embeddings = torch.randn(n_tokens, d_model)
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    
    # Select random indices as ground truth
    ground_truth = torch.randint(0, n_tokens, (num_queries,))
    
    # Queries are the embeddings at ground truth indices + small noise
    queries = embeddings[ground_truth] + 0.1 * torch.randn(num_queries, d_model)
    queries = queries / queries.norm(dim=-1, keepdim=True)
    
    return embeddings, queries, ground_truth


def measure_recall_at_k(
    encoder: REWAEncoder,
    embeddings: torch.Tensor,
    queries: torch.Tensor,
    ground_truth: torch.Tensor,
    k: int = 10,
) -> float:
    """
    Measure Recall@K for witness matching.
    
    Args:
        encoder: REWA encoder
        embeddings: [n_tokens, d_model] token embeddings
        queries: [num_queries, d_model] query embeddings
        ground_truth: [num_queries] true match indices
        k: Number of top results to consider
    
    Returns:
        Recall@K (fraction of queries with correct match in top-k)
    """
    n_tokens = embeddings.shape[0]
    num_queries = queries.shape[0]
    
    # Encode embeddings and queries
    with torch.no_grad():
        # Add batch dimension
        emb_batch = embeddings.unsqueeze(0)  # [1, n_tokens, d_model]
        query_batch = queries.unsqueeze(0)  # [1, num_queries, d_model]
        
        # Encode
        emb_encoded = encoder(emb_batch, return_continuous=False)  # [1, n_tokens, m_bits]
        query_encoded = encoder(query_batch, return_continuous=False)  # [1, num_queries, m_bits]
        
        # Compute similarity
        similarity = hamming_similarity_efficient(
            query_encoded, emb_encoded
        )  # [1, num_queries, n_tokens]
        
        similarity = similarity.squeeze(0)  # [num_queries, n_tokens]
        
        # Get top-k indices
        top_k_indices = similarity.topk(k, dim=-1)[1]  # [num_queries, k]
        
        # Check if ground truth is in top-k
        ground_truth_expanded = ground_truth.unsqueeze(1)  # [num_queries, 1]
        matches = (top_k_indices == ground_truth_expanded).any(dim=-1)  # [num_queries]
        
        recall = matches.float().mean().item()
    
    return recall


def find_capacity_limit(
    d_model: int,
    m_bits: int = 32,
    min_n: int = 100,
    max_n: int = 100000,
    recall_threshold: float = 0.9,
    num_queries: int = 100,
    k: int = 10,
) -> tuple[int, list[int], list[float]]:
    """
    Find n_max where recall drops below threshold.
    
    Args:
        d_model: Model dimension
        m_bits: Bits in encoding
        min_n: Minimum sequence length to test
        max_n: Maximum sequence length to test
        recall_threshold: Recall threshold (default 0.9)
        num_queries: Number of queries per test
        k: K for Recall@K
    
    Returns:
        n_max: Maximum context length
        n_values: List of tested n values
        recall_values: List of recall values
    """
    print(f"\nTesting d={d_model}, m_bits={m_bits}")
    print(f"Theoretical n_max: {max_context_length(d_model)}")
    
    # Create encoder
    encoder = REWAEncoder(d_model, m_bits, monoid='boolean')
    encoder.eval()
    
    # Binary search for n_max
    n_values = []
    recall_values = []
    
    # Logarithmic sampling
    n_test_values = np.logspace(
        np.log10(min_n),
        np.log10(max_n),
        num=20
    ).astype(int)
    
    for n in tqdm(n_test_values, desc=f"d={d_model}"):
        # Generate data
        embeddings, queries, ground_truth = generate_synthetic_data(
            n_tokens=n,
            d_model=d_model,
            num_queries=num_queries,
        )
        
        # Measure recall
        recall = measure_recall_at_k(
            encoder, embeddings, queries, ground_truth, k=k
        )
        
        n_values.append(n)
        recall_values.append(recall)
        
        print(f"  n={n:6d}, Recall@{k}={recall:.3f}")
    
    # Find n_max (first n where recall < threshold)
    n_max = max_n
    for n, recall in zip(n_values, recall_values):
        if recall < recall_threshold:
            n_max = n
            break
    
    print(f"Found n_max: {n_max} (recall drops below {recall_threshold})")
    
    return n_max, n_values, recall_values


def run_capacity_experiment():
    """
    Run full capacity validation experiment.
    
    Tests d ∈ {64, 128, 256, 512} and validates n_max ≈ exp(√d).
    """
    print("="*60)
    print("BREWA Capacity Validation Experiment")
    print("="*60)
    print("\nTheorem 6.1: n_max ≈ exp(√d)")
    print()
    
    # Test different dimensions
    d_values = [64, 128, 256, 512]
    m_bits = 32
    
    results = {}
    
    for d in d_values:
        # Adjust max_n based on theoretical limit
        theoretical_max = max_context_length(d)
        max_n = min(theoretical_max * 2, 1_000_000)  # Cap at 1M for speed
        
        n_max, n_vals, recall_vals = find_capacity_limit(
            d_model=d,
            m_bits=m_bits,
            max_n=max_n,
        )
        
        results[d] = {
            'n_max': n_max,
            'n_values': n_vals,
            'recall_values': recall_vals,
            'theoretical_n_max': theoretical_max,
        }
    
    # Print summary table
    print("\n" + "="*70)
    print("Capacity Validation Results")
    print("="*70)
    print(f"{'d':<10} {'Theoretical':<15} {'Measured':<15} {'Error':<15}")
    print("-"*70)
    
    for d in d_values:
        theoretical = results[d]['theoretical_n_max']
        measured = results[d]['n_max']
        error = abs(measured - theoretical) / theoretical * 100
        
        print(f"{d:<10} {theoretical:<15} {measured:<15} {error:.1f}%")
    
    print("="*70)
    
    # Plot results
    plot_capacity_results(results, d_values)
    
    return results


def plot_capacity_results(results: dict, d_values: list[int]):
    """
    Plot capacity validation results.
    
    Creates two plots:
    1. Recall vs sequence length for each d
    2. Measured vs theoretical n_max
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Recall curves
    for d in d_values:
        n_vals = results[d]['n_values']
        recall_vals = results[d]['recall_values']
        
        ax1.plot(n_vals, recall_vals, marker='o', label=f'd={d}')
        
        # Mark theoretical n_max
        theoretical = results[d]['theoretical_n_max']
        ax1.axvline(theoretical, linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Sequence Length (n)')
    ax1.set_ylabel('Recall@10')
    ax1.set_xscale('log')
    ax1.set_title('Recall vs Sequence Length')
    ax1.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90% threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Measured vs Theoretical
    theoretical_vals = [results[d]['theoretical_n_max'] for d in d_values]
    measured_vals = [results[d]['n_max'] for d in d_values]
    
    ax2.scatter(theoretical_vals, measured_vals, s=100, alpha=0.7)
    
    # Diagonal line (perfect match)
    max_val = max(max(theoretical_vals), max(measured_vals))
    ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect match')
    
    # Add labels
    for d, theo, meas in zip(d_values, theoretical_vals, measured_vals):
        ax2.annotate(f'd={d}', (theo, meas), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    
    ax2.set_xlabel('Theoretical n_max')
    ax2.set_ylabel('Measured n_max')
    ax2.set_title('Measured vs Theoretical Capacity')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('capacity_validation.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: brewa-experiment/capacity_validation.png")
    plt.close()


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run experiment
    results = run_capacity_experiment()
    
    print("\n" + "="*60)
    print("Capacity Validation Complete!")
    print("="*60)
