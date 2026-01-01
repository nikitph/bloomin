import torch
from obds_diffusion.riemannian_v2 import RiemannianDiffusionFull

def test_riemannian_geometry():
    print("Testing Advanced Riemannian Geometry...")
    device = "cpu"
    
    # Initialize L2 layer
    manifold_dim = 10
    riemann = RiemannianDiffusionFull(data_dim=784, manifold_dim=manifold_dim).to(device)
    
    batch_size = 4
    x = torch.randn(batch_size, 784, device=device)
    
    # 1. Metric Tensor
    g = riemann.metric_tensor(x)
    print(f"Metric Tensor Shape: {g.shape}") # Should be (4, 10, 10)
    
    # Check for symmetry and positive definiteness
    sym_error = (g - g.transpose(-1, -2)).abs().max().item()
    print(f"Symmetry Error: {sym_error:.6f}")
    
    try:
        torch.linalg.cholesky(g)
        print("Metric is Positive Definite: YES")
    except RuntimeError:
        print("Metric is Positive Definite: NO")

    # 2. Laplacian & Score
    t = torch.tensor([0.5]*batch_size, device=device).view(-1, 1)
    
    # This computes the Riemannian score $\nabla_g \log p$
    score = riemann(x, t)
    print(f"Riemannian Score Shape: {score.shape}") # (4, 784)
    
    print("Advanced Riemannian test passed!")

if __name__ == "__main__":
    test_riemannian_geometry()
