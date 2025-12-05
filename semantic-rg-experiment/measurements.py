"""Measurement functions for RG flow observables."""

import numpy as np


def measure_couplings(distributions, data):
    """
    Measure the geometric couplings at a given RG scale.
    
    Returns observables:
    - L: Witness multiplicity (number of witnesses)
    - rho: Signal strength (average pairwise overlap)
    - K: Curvature proxy (inverse participation ratio)
    - Chi: Thermodynamic variable from equation of state
    
    Args:
        distributions: (N, L) witness probability distributions
        data: (N, D) original data points
        
    Returns:
        dict: Dictionary of measured observables
    """
    n_samples, L = distributions.shape
    
    # 1. Witness Multiplicity L(s)
    witness_count = L
    
    # 2. Signal Strength rho(s) (Average pairwise overlap)
    # Proxy: Mean value of off-diagonal elements of Gram matrix
    pairwise_overlaps = np.dot(distributions, distributions.T)
    
    # Extract off-diagonal elements
    mask = ~np.eye(n_samples, dtype=bool)
    rho = np.mean(pairwise_overlaps[mask])
    
    # 3. Curvature K(s) (Ollivier-Ricci Curvature Proxy)
    # Estimate effective dimensionality via Participation Ratio
    # High curvature (Hyperbolic) -> Low effective dim (Hierarchical concentration)
    eigenvalues = np.linalg.svd(pairwise_overlaps, compute_uv=False)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter numerical noise
    
    # Participation Ratio
    PR = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)
    K_proxy = 1.0 / (PR + 1e-10)  # Inverse participation ratio
    
    # 4. Thermodynamic Variable Chi
    # Use Equation of State from Paper 4
    # Chi = (m * Delta^2) / log(N)
    Delta = rho  # Approximation for gap
    m = L  # Assuming 1 bit per witness for simplicity
    Chi = (m * Delta**2) / (np.log(n_samples) + 1e-10)
    
    return {
        "L": witness_count,
        "rho": rho,
        "K": K_proxy,
        "Chi": Chi,
        "PR": PR  # Also return participation ratio for analysis
    }


def evaluate_retrieval_accuracy(distributions, data, k=10):
    """
    Evaluate retrieval accuracy using witness representations.
    
    For each query point, find k nearest neighbors in witness space
    and compare to ground truth nearest neighbors in original space.
    
    Args:
        distributions: (N, L) witness probability distributions
        data: (N, D) original data points
        k: Number of neighbors to retrieve
        
    Returns:
        float: Recall@k score
    """
    n_samples = len(data)
    
    # Compute ground truth neighbors in original space
    # distances[i, j] = ||data[i] - data[j]||^2
    data_distances = np.sum(data**2, axis=1, keepdims=True) + \
                     np.sum(data**2, axis=1) - \
                     2 * np.dot(data, data.T)
    
    # For each point, find k+1 nearest (including itself)
    true_neighbors = np.argsort(data_distances, axis=1)[:, :k+1]
    
    # Compute distances in witness space
    # Use negative dot product as distance (higher overlap = closer)
    witness_similarities = np.dot(distributions, distributions.T)
    witness_distances = -witness_similarities
    
    # Find k+1 nearest in witness space
    pred_neighbors = np.argsort(witness_distances, axis=1)[:, :k+1]
    
    # Compute recall
    recalls = []
    for i in range(n_samples):
        # Exclude the point itself
        true_set = set(true_neighbors[i, 1:])
        pred_set = set(pred_neighbors[i, 1:])
        
        # Recall = |intersection| / k
        recall = len(true_set & pred_set) / k
        recalls.append(recall)
    
    return np.mean(recalls)


if __name__ == "__main__":
    from data_generation import generate_hierarchical_data
    from witness_init import extract_micro_witnesses
    
    # Test measurements
    data = generate_hierarchical_data()
    dist, _, _ = extract_micro_witnesses(data)
    
    print("\nMeasuring couplings...")
    obs = measure_couplings(dist, data)
    for key, value in obs.items():
        print(f"  {key}: {value:.6f}")
    
    print("\nEvaluating retrieval accuracy...")
    acc = evaluate_retrieval_accuracy(dist, data, k=10)
    print(f"  Recall@10: {acc:.4f}")
