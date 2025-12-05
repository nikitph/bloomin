"""
Geometry Measurement: The "Ricci" Metrics
Compute Gram matrix (metric tensor proxy) and curvature entropy
"""

import torch
import numpy as np
from scipy.stats import entropy


def compute_geometry_snapshot(encoder, data):
    """
    Compute geometric properties of the embedding manifold
    
    Args:
        encoder: MLPEncoder instance
        data: torch.Tensor [N, DIM_INPUT]
    
    Returns:
        dict with keys:
            - G: Gram matrix [N, N] (metric tensor proxy)
            - entropy: curvature entropy (scalar)
            - embeddings: normalized embeddings [N, DIM_EMBED]
    """
    encoder.eval()
    
    with torch.no_grad():
        # 1. Forward pass
        embeddings = encoder(data)  # Already normalized in encoder
        
        # 2. Compute Gram Matrix (Approximation of Metric Tensor g)
        # G_ij represents the pairwise geometry
        G = torch.matmul(embeddings, embeddings.T)
        
        # 3. Compute "Curvature" Proxy (Spectral distribution)
        # High entropy eigenvalues = Flat/Noisy
        # Low entropy eigenvalues = Curved/Structured
        eigenvalues = torch.linalg.svdvals(G)
        
        # Normalize eigenvalues to form a probability distribution
        eigenvalues = eigenvalues.cpu().numpy()
        eigenvalues = eigenvalues / eigenvalues.sum()
        
        # Compute entropy
        curvature_entropy = entropy(eigenvalues)
    
    return {
        "G": G.cpu(),
        "entropy": curvature_entropy,
        "embeddings": embeddings.cpu()
    }


def compute_metric_deviation(G1, G2):
    """
    Compute Frobenius norm between two metric tensors
    
    Args:
        G1, G2: torch.Tensor [N, N]
    
    Returns:
        deviation: scalar (Frobenius norm of difference)
    """
    diff = G1 - G2
    deviation = torch.norm(diff, p='fro').item()
    return deviation


if __name__ == "__main__":
    from model import MLPEncoder
    from data_generation import generate_data
    
    # Test geometry computation
    data, _ = generate_data()
    encoder = MLPEncoder()
    
    snapshot = compute_geometry_snapshot(encoder, data)
    
    print(f"Gram matrix shape: {snapshot['G'].shape}")
    print(f"Curvature entropy: {snapshot['entropy']:.3f}")
    print(f"Embeddings shape: {snapshot['embeddings'].shape}")
    
    # Test metric deviation
    snapshot2 = compute_geometry_snapshot(encoder, data)
    deviation = compute_metric_deviation(snapshot['G'], snapshot2['G'])
    print(f"Metric deviation (should be ~0): {deviation:.6f}")
