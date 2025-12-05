"""Renormalization Group operator for witness coarse-graining."""

import numpy as np
from sklearn.cluster import AgglomerativeClustering


def compute_gram_matrix(distributions):
    """
    Compute witness-witness similarity matrix.
    
    G_ij = CosineSimilarity(Distribution_i, Distribution_j)
    
    Args:
        distributions: (N, L) array where each column is a witness distribution
        
    Returns:
        (L, L) Gram matrix
    """
    # Normalize columns to unit vectors
    norms = np.linalg.norm(distributions, axis=0, keepdims=True) + 1e-10
    normalized = distributions / norms
    
    # Compute cosine similarity
    gram = np.dot(normalized.T, normalized)
    
    return gram


def renormalization_step(current_distributions, target_size):
    """
    Coarse-grain witnesses from size L to size target_size.
    This implements the 'Blocking' operator.
    
    The RG operator merges witnesses based on their semantic affinity
    (how similarly they activate across the data).
    
    Args:
        current_distributions: (N, L) array of witness probabilities
        target_size: Target number of witnesses after coarse-graining
        
    Returns:
        (N, target_size) array of renormalized witness distributions
    """
    n_samples, n_witnesses = current_distributions.shape
    
    print(f"  Renormalizing: {n_witnesses} -> {target_size} witnesses")
    
    # 1. Compute Witness-Witness Information Metric (Fisher Distance)
    # How similar are two witnesses? (Do they activate for the same data?)
    G_witness = compute_gram_matrix(current_distributions)
    
    # 2. Cluster Witnesses (Agglomerative Clustering)
    # Merge witnesses that are geometrically close
    # Distance = 1 - Similarity
    distance_matrix = 1.0 - G_witness
    np.fill_diagonal(distance_matrix, 0)  # Ensure diagonal is 0
    
    clustering = AgglomerativeClustering(
        n_clusters=target_size,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distance_matrix)
    
    # 3. Create Macroscopic Packets
    new_distributions = np.zeros((n_samples, target_size))
    
    for cluster_id in range(target_size):
        # Indices of old witnesses merging into this new packet
        indices = np.where(labels == cluster_id)[0]
        
        # Renormalization: Sum probabilities (conservation of mass)
        # P_new = Sum(P_old)
        new_distributions[:, cluster_id] = np.sum(
            current_distributions[:, indices], 
            axis=1
        )
    
    # Normalize to keep probability axioms valid
    row_sums = np.sum(new_distributions, axis=1, keepdims=True) + 1e-10
    new_distributions = new_distributions / row_sums
    
    print(f"  Mean cluster size: {n_witnesses / target_size:.2f}")
    
    return new_distributions


if __name__ == "__main__":
    from data_generation import generate_hierarchical_data
    from witness_init import extract_micro_witnesses
    
    # Test RG operator
    data = generate_hierarchical_data()
    dist, _, _ = extract_micro_witnesses(data)
    
    print(f"\nTesting RG operator...")
    print(f"Initial size: {dist.shape[1]}")
    
    # Apply one RG step
    new_dist = renormalization_step(dist, target_size=128)
    print(f"After RG: {new_dist.shape[1]}")
    print(f"Probability conservation: {np.allclose(np.sum(new_dist, axis=1), 1.0)}")
