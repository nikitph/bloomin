import torch
import numpy as np
import scipy.sparse as sp
from typing import Tuple, Optional

def generate_synthetic_data(n_samples: int, ambient_dim: int, n_clusters: int = 5, cluster_std: float = 1.0) -> torch.Tensor:
    """
    Generates synthetic data spread across clusters to simulate semantic basins.
    """
    means = np.random.randn(n_clusters, ambient_dim) * 5
    X = []
    labels = []
    
    samples_per_cluster = n_samples // n_clusters
    
    for i in range(n_clusters):
        # Generate samples for this cluster
        cluster_data = np.random.randn(samples_per_cluster, ambient_dim) * cluster_std + means[i]
        X.append(cluster_data)
        labels.extend([i] * samples_per_cluster)
        
    X = np.vstack(X)
    # Shuffle
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X = X[indices]
    
    return torch.tensor(X, dtype=torch.float32)

def build_knn_graph(data: torch.Tensor, k: int = 10, mode: str = 'connectivity') -> torch.sparse.Tensor:
    """
    Builds a k-nearest neighbor graph and returns its normalized Laplacian.
    
    Args:
        data: (N, D) tensor of data points.
        k: Number of neighbors.
        mode: 'connectivity' or 'distance' (weighted).
        
    Returns:
        L: (N, N) sparse torch tensor representing the normalized Laplacian.
    """
    N = data.shape[0]
    
    # Using FAISS for fast KNN if available, else naive distance
    # For simplicity/portability in this snippet, using sklearn or torch cdist for small scales
    # But usually FAISS is preferred. I'll use a torch based implementation for simplicity in this file
    # assuming N isn't massive. For massive N, use FAISS.
    
    # Compute pairwise distances
    # NOTE: For very large N, this is O(N^2) memory. 
    # In a real heavy production system we'd use FAISS here.
    # We'll assume N ~ 1000-5000 for this prototype or use chunking.
    
    # Let's try to be a bit efficient
    # Calculate distances
    dist_matrix = torch.cdist(data, data) # (N, N)
    
    # Get top k neighbors
    # topk returns largest, so we negate distance or create a huge default
    values, indices = torch.topk(dist_matrix, k + 1, largest=False) 
    # neighbor 0 is self, so take 1..k+1
    
    row_indices = torch.arange(N).view(-1, 1).expand(-1, k+1)
    
    src = row_indices[:, 1:].flatten()
    dst = indices[:, 1:].flatten()
    
    # Build adjacency matrix
    if mode == 'connectivity':
        weights = torch.ones(src.shape[0])
    else:
        # Heat kernel weighting
        sigma = 1.0 # could be heuristic
        dists = values[:, 1:].flatten()
        weights = torch.exp(-dists**2 / (2 * sigma**2))
        
    # Create sparse adjacency matrix
    indices_adj = torch.stack([src, dst])
    
    # Make it symmetric (undirected graph)
    # The KNN graph is not naturally symmetric. A -> B doesn't mean B -> A.
    # Usually we symmetrize: max(A, A.T) or min(A, A.T) or mean.
    # Here we'll just add edges both ways for simplicity in sparse construction
    
    src_sym = torch.cat([src, dst])
    dst_sym = torch.cat([dst, src])
    weights_sym = torch.cat([weights, weights])
    
    indices_sym = torch.stack([src_sym, dst_sym])
    
    # Coalesce to sum duplicate weights if any (simplistic symmetrization logic)
    adj = torch.sparse_coo_tensor(indices_sym, weights_sym, (N, N)).coalesce()
    
    # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    # Degree matrix
    # Sparse sum is not fully supported in all torch versions, convert to dense diagonal for logic
    adj_dense = adj.to_dense() # Warning: Only suitable for small-medium N
    adj_dense = (adj_dense > 0).float() # Force connectivity only for robust Laplacian
    
    degree = adj_dense.sum(dim=1)
    d_inv_sqrt = torch.pow(degree, -0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0
    
    # D^{-1/2} @ A @ D^{-1/2}
    # Broadcasting for efficiency: multiply rows then cols
    normalized_adj = adj_dense * d_inv_sqrt.view(-1, 1) * d_inv_sqrt.view(1, -1)
    
    identity = torch.eye(N)
    L = identity - normalized_adj
    
    return L.to_sparse()

def sparse_dense_mul(s, d):
    """
    Helper for sparse x dense matrix multiplication.
    s: (N, N) sparse
    d: (N, 1) or (N) dense
    """
    if d.dim() == 1:
        d = d.view(-1, 1)
    return torch.sparse.mm(s, d).squeeze()
