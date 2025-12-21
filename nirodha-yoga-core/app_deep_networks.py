import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from nirodha.core import CognitiveState, YogaRegulator

def run_ultra_deep_test():
    print("Starting Experiment 1: Ultra-Deep Network Stability (10,000 Layers)")
    dim = 512
    depth = 10000
    beta = 0.5
    
    # 1. Models
    # Native dynamics projection (residual block logic)
    projection = torch.randn(dim, dim) * 0.05
    
    # Nirodha System
    C0 = torch.randn(dim)
    state_nirodha = CognitiveState(C0.clone())
    regulator = YogaRegulator(beta=beta)
    
    # Baseline System (Standard Residuals: h = h + x)
    h_baseline = C0.clone()
    
    nirodha_norms = []
    baseline_norms = []
    
    print(f"Simulating forward pass through {depth} layers...")
    
    for i in range(depth):
        # Forward step dynamics
        delta = torch.tanh(state_nirodha.C @ projection)
        delta_b = torch.tanh(h_baseline @ projection)
        
        # Nirodha Regulation
        state_nirodha = regulator(state_nirodha, delta)
        
        # Baseline Update
        h_baseline = h_baseline + delta_b
        
        nirodha_norms.append(torch.norm(state_nirodha.C).item())
        baseline_norms.append(torch.norm(h_baseline).item())
        
        if i % 1000 == 0:
            print(f"  Layer {i:5}: Nirodha Norm={nirodha_norms[-1]:.2f}, Baseline Norm={baseline_norms[-1]:.2f}")
            if baseline_norms[-1] > 1e10:
                print("  [WARN] Baseline exploded.")
                break

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(nirodha_norms, label='Nirodha-Regulated (Lyapunov Stable)', color='blue')
    plt.plot(baseline_norms, label='Standard Residuals (Baseline)', color='red', linestyle='--')
    plt.yscale('log')
    plt.title('Ultra-Deep Stability: 10,000 Layers')
    plt.xlabel('Layer Depth')
    plt.ylabel('L2 Norm of Latent State')
    plt.legend()
    plt.grid(True)
    plt.savefig('app_deep_networks.png')
    print("\nResults saved to app_deep_networks.png")
    
    if nirodha_norms[-1] < 100 and baseline_norms[-1] > 1000:
        print("✅ SUCCESS: Nirodha maintained stability at 10,000 layers while baseline diverged.")
    else:
        print("❌ Stability differentiation not cleared.")

if __name__ == "__main__":
    run_ultra_deep_test()
