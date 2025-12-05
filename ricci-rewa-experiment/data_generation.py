"""
Data Generation: Hierarchical Gaussian Logic
Creates synthetic manifold with clustered structure
"""

import torch
import numpy as np
from config import CONFIG


def generate_data():
    """
    Generate N points clustered in DIM_INPUT space.
    This represents the "True Semantic Geometry"
    
    Returns:
        data: torch.Tensor of shape [N_SAMPLES, DIM_INPUT]
        labels: torch.Tensor of shape [N_SAMPLES] (cluster assignments)
    """
    np.random.seed(CONFIG["SEED"])
    torch.manual_seed(CONFIG["SEED"])
    
    n_samples = CONFIG["N_SAMPLES"]
    dim_input = CONFIG["DIM_INPUT"]
    n_clusters = CONFIG["N_CLUSTERS"]
    
    samples_per_cluster = n_samples // n_clusters
    
    # Generate cluster centers
    centers = torch.randn(n_clusters, dim_input) * 5.0  # Spread centers out
    
    data = []
    labels = []
    
    for cluster_id, center in enumerate(centers):
        # Sample points around this center
        points = center + torch.randn(samples_per_cluster, dim_input) * 1.0
        data.append(points)
        labels.extend([cluster_id] * samples_per_cluster)
    
    data = torch.cat(data, dim=0)
    labels = torch.tensor(labels)
    
    # Shuffle
    perm = torch.randperm(n_samples)
    data = data[perm]
    labels = labels[perm]
    
    return data, labels


if __name__ == "__main__":
    # Test data generation
    data, labels = generate_data()
    print(f"Generated data shape: {data.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of unique clusters: {len(torch.unique(labels))}")
    print(f"Data mean: {data.mean():.3f}, std: {data.std():.3f}")
