"""Witness initialization and affinity computation."""

import numpy as np
from sklearn.cluster import KMeans
from config import CONFIG


def compute_affinities(data, prototypes, temperature=1.0):
    """
    Compute soft affinities between data points and witness prototypes.
    
    P_x(w) ~ exp(-distance(x, w) / temperature)
    
    Args:
        data: (N, D) array of data points
        prototypes: (L, D) array of witness prototypes
        temperature: Temperature for softmax
        
    Returns:
        (N, L) array of affinities (probabilities)
    """
    # Compute pairwise distances
    # distances[i, j] = ||data[i] - prototypes[j]||^2
    distances = np.sum(data**2, axis=1, keepdims=True) + \
                np.sum(prototypes**2, axis=1) - \
                2 * np.dot(data, prototypes.T)
    
    # Convert to affinities via softmax
    affinities = np.exp(-distances / temperature)
    
    # Normalize to get probability distributions
    affinities = affinities / (np.sum(affinities, axis=1, keepdims=True) + 1e-10)
    
    return affinities


def top_k_witnesses(distributions, k):
    """
    Convert soft distributions to discrete witness sets (top-k).
    
    Args:
        distributions: (N, L) array of witness probabilities
        k: Number of top witnesses to keep
        
    Returns:
        (N, L) binary array indicating top-k witnesses
    """
    witness_sets = np.zeros_like(distributions)
    
    # For each data point, mark top-k witnesses
    top_k_indices = np.argpartition(distributions, -k, axis=1)[:, -k:]
    
    for i in range(len(distributions)):
        witness_sets[i, top_k_indices[i]] = 1
    
    return witness_sets


def extract_micro_witnesses(data):
    """
    Initialize microscopic witnesses using k-means clustering.
    
    Args:
        data: (N, D) array of data points
        
    Returns:
        witness_distributions: (N, L_MICRO) soft affinities
        witness_sets: (N, L_MICRO) binary top-k sets
        prototypes: (L_MICRO, D) witness centroids
    """
    print(f"\nInitializing {CONFIG['L_MICRO']} microscopic witnesses...")
    
    # Use k-means to find witness prototypes
    kmeans = KMeans(
        n_clusters=CONFIG["L_MICRO"],
        random_state=42,
        n_init=10,
        max_iter=300
    )
    kmeans.fit(data)
    prototypes = kmeans.cluster_centers_
    
    # Compute soft affinities
    witness_distributions = compute_affinities(data, prototypes)
    
    # Convert to discrete sets for Boolean REWA checks
    witness_sets = top_k_witnesses(witness_distributions, CONFIG["TOP_K_WITNESSES"])
    
    print(f"Witness initialization complete")
    print(f"  Mean affinity: {np.mean(witness_distributions):.6f}")
    print(f"  Max affinity: {np.max(witness_distributions):.6f}")
    print(f"  Active witnesses per point: {np.sum(witness_sets, axis=1).mean():.2f}")
    
    return witness_distributions, witness_sets, prototypes


if __name__ == "__main__":
    from data_generation import generate_hierarchical_data
    
    data = generate_hierarchical_data()
    dist, sets, proto = extract_micro_witnesses(data)
    print(f"\nDistributions shape: {dist.shape}")
    print(f"Sets shape: {sets.shape}")
    print(f"Prototypes shape: {proto.shape}")
