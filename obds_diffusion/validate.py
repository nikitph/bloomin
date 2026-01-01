import torch
import torch.nn as nn
import numpy as np
from .model import OBDSDiffusion
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

def validate_polynomial_sufficiency(model, device='cpu'):
    print("\n--- Phase 1: Polynomial Sufficiency ---")
    # We want to see how well the polynomial approximates the true score
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    
    x_0, _ = next(iter(val_loader))
    x_0 = x_0.to(device)
    t = torch.randint(0, model.n_timesteps, (x_0.shape[0],), device=device)
    alpha_cumprod = model.alphas_cumprod[t].view(-1, 1)
    noise = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_cumprod) * x_0 + torch.sqrt(1 - alpha_cumprod) * noise
    
    with torch.no_grad():
        score_poly = model.poly_score(x_t, t.float() / model.n_timesteps)
        score_true = -(x_t - torch.sqrt(alpha_cumprod) * x_0) / (1 - alpha_cumprod)
        
    error = nn.MSELoss()(score_poly, score_true).item()
    print(f"Polynomial MSE: {error:.4f}")
    return error

def validate_sampling_speedup(model, device='cpu'):
    print("\n--- Phase 2: Sampling Speedup ---")
    start_time = time.time()
    samples_fast = model.sample(batch_size=16, num_steps=10)
    fast_time = time.time() - start_time
    print(f"10-step sampling time: {fast_time:.4f}s")
    
    start_time = time.time()
    samples_slow = model.sample(batch_size=16, num_steps=100)
    slow_time = time.time() - start_time
    print(f"100-step sampling time: {slow_time:.4f}s")
    
    print(f"Speedup: {slow_time/fast_time:.1f}x")
    return samples_fast

def validate_diversity(model, device='cpu'):
    print("\n--- Phase 3: Reaction-Diffusion Diversity ---")
    # For now, we measure variance and visual inspection
    samples = model.sample(batch_size=64, num_steps=50)
    variance = torch.var(samples).item()
    print(f"Sample Variance: {variance:.4f}")
    # Higher variance often indicates less mode collapse in early training
    return variance

if __name__ == "__main__":
    import time
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = OBDSDiffusion(data_dim=784, max_degree=5).to(device)
    # Try to load if exists, else it's a cold validation of the structure
    checkpoint_path = 'checkpoints/obds_diffusion_mnist.pt'
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path))
            print("Loaded trained model.")
        except:
             print("Error loading model, validating random initialization.")
    
    validate_polynomial_sufficiency(model, device)
    validate_sampling_speedup(model, device)
    validate_diversity(model, device)
