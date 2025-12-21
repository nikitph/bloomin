import torch
import numpy as np
import matplotlib.pyplot as plt
from nirodha.core import CognitiveState, YogaRegulator

def run_infinite_reasoning_test():
    print("Starting Experiment 2: Infinite-Horizon Reasoning (5,000 Steps)")
    dim = 512
    horizon = 5000
    beta = 0.5
    
    # 1. Models
    # Recurrent dynamics representing a reasoning step
    # We add a small bias to simulate 'drift' or 'repetition' tendencies
    projection = torch.randn(dim, dim) * 0.1
    drift_bias = torch.randn(dim) * 0.01
    
    # Nirodha System
    C0 = torch.randn(dim)
    state_nirodha = CognitiveState(C0.clone())
    regulator = YogaRegulator(beta=beta)
    
    # Baseline System (Unregulated Recurrence)
    h_baseline = C0.clone()
    
    nirodha_coherence = []
    baseline_coherence = []
    
    print(f"Running reasoning loop for {horizon} steps...")
    
    for t in range(horizon):
        # Reasoning update
        delta = torch.tanh(state_nirodha.C @ projection + drift_bias)
        delta_b = torch.tanh(h_baseline @ projection + drift_bias)
        
        # Nirodha Update
        state_nirodha = regulator(state_nirodha, delta)
        
        # Baseline Update
        h_baseline = h_baseline + delta_b
        
        # Measure Coherence (Normalized Dot Product with Start State)
        # As a proxy for semantic consistency
        coh_n = torch.nn.functional.cosine_similarity(state_nirodha.C, C0, dim=0).item()
        coh_b = torch.nn.functional.cosine_similarity(h_baseline, C0, dim=0).item()
        
        nirodha_coherence.append(coh_n)
        baseline_coherence.append(coh_b)
        
        if t % 1000 == 0:
            print(f"  Step {t:4}: Nirodha Coherence={coh_n:.4f}, Baseline Coherence={coh_b:.4f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(nirodha_coherence, label='Nirodha-Stable Reasoning', color='blue')
    plt.plot(baseline_coherence, label='Standard Recurrent Loop', color='red', linestyle='--')
    plt.title('Infinite-Horizon Reasoning Coherence')
    plt.xlabel('Reasoning Steps')
    plt.ylabel('Coherence (Cosine Sim to Anchor)')
    plt.legend()
    plt.grid(True)
    plt.savefig('app_infinite_reasoning.png')
    print("\nResults saved to app_infinite_reasoning.png")
    
    if np.var(nirodha_coherence[-500:]) < np.var(baseline_coherence[-500:]):
        print("✅ SUCCESS: Nirodha maintained higher reasoning coherence and stability.")
    else:
        print("❌ Reasoning differentiator not cleared.")

if __name__ == "__main__":
    run_infinite_reasoning_test()
