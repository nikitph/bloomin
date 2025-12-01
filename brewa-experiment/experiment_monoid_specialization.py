"""
Experiment 3: Monoid Specialization
====================================

Validates that different monoid heads specialize in different tasks:
- Boolean: Exact pattern matching
- Tropical: Shortest path reasoning
- Real: Semantic similarity

Measures which head activates most for each task type.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from multi_monoid_attention import (
    BooleanREWAHead,
    TropicalREWAHead,
    RealREWAHead,
    ProductMonoidHead,
)


def create_exact_matching_task(
    n_tokens: int = 1000,
    d_model: int = 128,
    pattern_length: int = 5,
    num_queries: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create exact pattern matching task.
    
    Task: Find exact token sequences in a longer sequence.
    Boolean head should excel at this.
    
    Returns:
        Q, K, V, ground_truth
    """
    # Create random embeddings
    embeddings = torch.randn(n_tokens, d_model)
    
    # Create patterns (exact copies)
    pattern_starts = torch.randint(0, n_tokens - pattern_length, (num_queries,))
    
    queries = []
    for start in pattern_starts:
        # Query is exact copy of pattern
        pattern = embeddings[start:start+pattern_length].mean(dim=0)
        queries.append(pattern)
    
    queries = torch.stack(queries)  # [num_queries, d_model]
    
    # Add batch dimension
    Q = queries.unsqueeze(0)  # [1, num_queries, d_model]
    K = embeddings.unsqueeze(0)  # [1, n_tokens, d_model]
    V = embeddings.unsqueeze(0)  # [1, n_tokens, d_model]
    
    ground_truth = pattern_starts  # [num_queries]
    
    return Q, K, V, ground_truth


def create_shortest_path_task(
    n_nodes: int = 100,
    d_model: int = 128,
    num_queries: int = 50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create shortest path task.
    
    Task: Find shortest paths in a graph.
    Tropical head should excel at this.
    
    Returns:
        Q, K, V, ground_truth
    """
    # Create random graph (adjacency matrix)
    # Use distances instead of binary adjacency
    adj_matrix = torch.rand(n_nodes, n_nodes) * 10
    adj_matrix = (adj_matrix + adj_matrix.T) / 2  # Symmetric
    
    # Embed nodes based on their distances
    # Nodes with similar distance profiles should be similar
    embeddings = torch.randn(n_nodes, d_model)
    
    # Make embeddings reflect graph structure
    for i in range(n_nodes):
        # Add weighted sum of neighbors
        neighbors = adj_matrix[i] < 3.0  # Close neighbors
        if neighbors.sum() > 0:
            embeddings[i] += 0.3 * embeddings[neighbors].mean(dim=0)
    
    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
    
    # Queries: find nodes with shortest total distance
    query_nodes = torch.randint(0, n_nodes, (num_queries,))
    queries = embeddings[query_nodes]
    
    # Ground truth: nodes closest to query nodes
    ground_truth = []
    for qnode in query_nodes:
        distances = adj_matrix[qnode]
        closest = distances.argsort()[1]  # Exclude self
        ground_truth.append(closest)
    
    ground_truth = torch.tensor(ground_truth)
    
    # Add batch dimension
    Q = queries.unsqueeze(0)
    K = embeddings.unsqueeze(0)
    V = embeddings.unsqueeze(0)
    
    return Q, K, V, ground_truth


def create_semantic_similarity_task(
    n_tokens: int = 1000,
    d_model: int = 128,
    num_clusters: int = 10,
    num_queries: int = 100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create semantic similarity task.
    
    Task: Find semantically similar tokens (same cluster).
    Real head should excel at this.
    
    Returns:
        Q, K, V, ground_truth
    """
    # Create clustered embeddings
    cluster_centers = torch.randn(num_clusters, d_model)
    cluster_centers = cluster_centers / cluster_centers.norm(dim=-1, keepdim=True)
    
    # Assign tokens to clusters
    cluster_assignments = torch.randint(0, num_clusters, (n_tokens,))
    
    # Create embeddings near cluster centers
    embeddings = []
    for i in range(n_tokens):
        cluster_id = cluster_assignments[i]
        center = cluster_centers[cluster_id]
        # Add noise
        emb = center + 0.2 * torch.randn(d_model)
        emb = emb / emb.norm()
        embeddings.append(emb)
    
    embeddings = torch.stack(embeddings)
    
    # Queries: random tokens
    query_indices = torch.randint(0, n_tokens, (num_queries,))
    queries = embeddings[query_indices]
    
    # Ground truth: find another token in same cluster
    ground_truth = []
    for qidx in query_indices:
        cluster_id = cluster_assignments[qidx]
        # Find other tokens in same cluster
        same_cluster = (cluster_assignments == cluster_id).nonzero(as_tuple=True)[0]
        # Exclude query itself
        same_cluster = same_cluster[same_cluster != qidx]
        if len(same_cluster) > 0:
            ground_truth.append(same_cluster[0].item())
        else:
            ground_truth.append(qidx.item())  # Fallback
    
    ground_truth = torch.tensor(ground_truth)
    
    # Add batch dimension
    Q = queries.unsqueeze(0)
    K = embeddings.unsqueeze(0)
    V = embeddings.unsqueeze(0)
    
    return Q, K, V, ground_truth


def measure_head_performance(
    head: nn.Module,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ground_truth: torch.Tensor,
    k: int = 5,
) -> float:
    """
    Measure head performance on a task.
    
    Returns:
        Recall@K
    """
    with torch.no_grad():
        output, attn_weights = head(Q, K, V)
        
        # Get top-k attended tokens
        top_k_indices = attn_weights.squeeze(0).topk(k, dim=-1)[1]  # [num_queries, k]
        
        # Check if ground truth in top-k
        ground_truth_expanded = ground_truth.unsqueeze(1)  # [num_queries, 1]
        matches = (top_k_indices == ground_truth_expanded).any(dim=-1)
        
        recall = matches.float().mean().item()
    
    return recall


def run_specialization_experiment():
    """
    Run full monoid specialization experiment.
    """
    print("="*60)
    print("BREWA Monoid Specialization Experiment")
    print("="*60)
    
    d_model = 128
    m_bits = 32
    
    # Create heads
    print("\nCreating attention heads...")
    boolean_head = BooleanREWAHead(d_model, d_head=64, m_bits=m_bits)
    tropical_head = TropicalREWAHead(d_model, d_head=64, m_bits=m_bits)
    real_head = RealREWAHead(d_model, d_head=64, m_bits=m_bits)
    
    boolean_head.eval()
    tropical_head.eval()
    real_head.eval()
    
    heads = {
        'Boolean': boolean_head,
        'Tropical': tropical_head,
        'Real': real_head,
    }
    
    # Test on different tasks
    tasks = {
        'Exact Matching': create_exact_matching_task,
        'Shortest Path': create_shortest_path_task,
        'Semantic Similarity': create_semantic_similarity_task,
    }
    
    results = {}
    
    for task_name, task_fn in tasks.items():
        print(f"\nTesting task: {task_name}")
        
        # Create task data
        Q, K, V, ground_truth = task_fn()
        
        task_results = {}
        
        for head_name, head in heads.items():
            recall = measure_head_performance(head, Q, K, V, ground_truth, k=5)
            task_results[head_name] = recall
            print(f"  {head_name}: Recall@5 = {recall:.3f}")
        
        results[task_name] = task_results
    
    # Print summary table
    print("\n" + "="*70)
    print("Monoid Specialization Results")
    print("="*70)
    print(f"{'Task':<25} {'Boolean':<15} {'Tropical':<15} {'Real':<15}")
    print("-"*70)
    
    for task_name in tasks.keys():
        task_res = results[task_name]
        print(f"{task_name:<25} {task_res['Boolean']:<15.3f} "
              f"{task_res['Tropical']:<15.3f} {task_res['Real']:<15.3f}")
    
    print("="*70)
    
    # Identify best head for each task
    print("\nBest Head for Each Task:")
    for task_name, task_res in results.items():
        best_head = max(task_res, key=task_res.get)
        best_score = task_res[best_head]
        print(f"  {task_name}: {best_head} ({best_score:.3f})")
    
    # Plot results
    plot_specialization_results(results, tasks.keys())
    
    return results


def plot_specialization_results(results: dict, task_names: list[str]):
    """
    Plot monoid specialization results.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    head_names = ['Boolean', 'Tropical', 'Real']
    x = np.arange(len(task_names))
    width = 0.25
    
    # Plot bars for each head
    for i, head_name in enumerate(head_names):
        scores = [results[task][head_name] for task in task_names]
        ax.bar(x + i*width, scores, width, label=head_name, alpha=0.8)
    
    ax.set_xlabel('Task Type')
    ax.set_ylabel('Recall@5')
    ax.set_title('Monoid Head Specialization')
    ax.set_xticks(x + width)
    ax.set_xticklabels(task_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('specialization_validation.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: brewa-experiment/specialization_validation.png")
    plt.close()


if __name__ == "__main__":
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run experiment
    results = run_specialization_experiment()
    
    print("\n" + "="*60)
    print("Monoid Specialization Experiment Complete!")
    print("="*60)
