import torch
import time
import numpy as np
from .model import OBDSDiffusion
from .symbolic_composer import SymbolicComposer
import matplotlib.pyplot as plt
import os

def test_symbolic_composition():
    device = "cpu" # Sympy is cpu-heavy
    print("Testing Symbolic Composition...")
    
    # Initialize model
    model = OBDSDiffusion(data_dim=784, max_degree=3).to(device)
    # Load trained weights if available
    checkpoint_path = 'checkpoints/obds_diffusion_mnist.pt'
    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print("Loaded trained model.")
        except Exception as e:
            print(f"Could not load model: {e}")
    
    # Initialize Composer
    composer = SymbolicComposer(model)
    
    # Time composition
    start_comp = time.time()
    num_steps_test = 100
    composer.compose_steps(num_steps=num_steps_test, truncate=True)
    comp_time = time.time() - start_comp
    print(f"Composition time (N={num_steps_test}): {comp_time:.4f}s")
    
    # Benchmark sampling
    batch_size = 64 # Use larger batch for amortized check
    noise = torch.randn(batch_size, 784, device=device)
    
    # Iterative
    start = time.time()
    samples_iter = model.sample(batch_size=batch_size, num_steps=num_steps_test)
    iter_time = time.time() - start
    print(f"Iterative ({num_steps_test}-step) time: {iter_time:.4f}s")
    
    # Symbolic 1-step
    start = time.time()
    samples_sym = composer.sample(noise)
    sym_time = time.time() - start
    print(f"Symbolic (1-step) time: {sym_time:.4f}s")
    
    print(f"Speedup vs 10-step: {iter_time / sym_time:.2f}x")
    
    # Accuracy check (they should be identical if the logic matches)
    diff = torch.abs(samples_iter - samples_sym).mean()
    print(f"Mean Difference: {diff.item():.6f}")

if __name__ == "__main__":
    test_symbolic_composition()
