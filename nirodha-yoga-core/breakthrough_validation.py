import torch
import numpy as np
import matplotlib.pyplot as plt
from nirodha.core import CognitiveState, YogaRegulator

def run_breakthrough_validation():
    print("Starting 'Needle in Chaos' Breakthrough Validation...")
    dim = 512
    T = 2000 # Long horizon to show persistence
    beta = 0.5
    
    # 1. Setup Signal and Noise
    # Ultra-weak hidden signal
    hidden_signal = torch.randn(dim)
    hidden_signal = hidden_signal / torch.norm(hidden_signal) * 1e-3 # Amplitude 10^-3
    
    # High-amplitude noise updates
    noise_amplitude = 10.0
    
    # 2. Initialize Systems
    C0 = torch.zeros(dim)
    
    # Nirodha System
    state_nirodha = CognitiveState(C0.clone())
    regulator = YogaRegulator(beta=beta)
    
    # Baseline System (Unregulated Random Walk)
    state_baseline = C0.clone()
    
    # 3. Tracking Metrics
    nirodha_similarities = []
    baseline_similarities = []
    nirodha_norms = []
    baseline_norms = []

    print(f"Running for {T} steps (Signal amplitude: {torch.norm(hidden_signal).item():.4f}, Noise amplitude: {noise_amplitude})...")

    for t in range(T):
        # Generate shared noise for this step
        noise = torch.randn(dim) * noise_amplitude
        update = noise + hidden_signal
        
        # Nirodha Update
        state_nirodha = regulator(state_nirodha, update)
        
        # Baseline Update
        state_baseline = state_baseline + update
        
        # Measure Information Recovery (Cosine Similarity with hidden signal)
        # We subtract the anchor to measure the 'discovered' direction
        sim_n = torch.nn.functional.cosine_similarity(state_nirodha.C - state_nirodha.C0, hidden_signal, dim=0).item()
        sim_b = torch.nn.functional.cosine_similarity(state_baseline - C0, hidden_signal, dim=0).item()
        
        nirodha_similarities.append(sim_n)
        baseline_similarities.append(sim_b)
        
        nirodha_norms.append(torch.norm(state_nirodha.C).item())
        baseline_norms.append(torch.norm(state_baseline).item())
        
        if t % 500 == 0:
            print(f"Step {t:4}: Nirodha Sim={sim_n:.4f}, Baseline Sim={sim_b:.4f}, Nirodha Norm={nirodha_norms[-1]:.2f}")

    # 4. Results & Visualization
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(nirodha_similarities, label='Nirodha-Regulated', color='blue')
    plt.plot(baseline_similarities, label='Baseline (Unregulated)', color='red', alpha=0.5)
    plt.title('Needle in Chaos: Signal Recovery (Cosine Similarity)')
    plt.xlabel('Time Steps')
    plt.ylabel('Similarity to Hidden Signal')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(nirodha_norms, label='Nirodha Norm', color='blue')
    plt.plot(baseline_norms, label='Baseline Norm', color='red')
    plt.yscale('log')
    plt.title('System Stability (Log Norm)')
    plt.xlabel('Time Steps')
    plt.ylabel('L2 Norm')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('breakthrough_validation.png')
    print("\nVisual results saved to breakthrough_validation.png")
    
    final_sim_n = np.mean(nirodha_similarities[-100:])
    final_sim_b = np.mean(baseline_similarities[-100:])
    
    print(f"\nFINAL BREAKTHROUGH METRIC:")
    print(f"Nirodha Signal Alignment: {final_sim_n:.6f}")
    print(f"Baseline Signal Alignment: {final_sim_b:.6f}")
    
    if final_sim_n > final_sim_b * 10 or final_sim_n > 0.1:
        print("\n✅ BREAKTHROUGH VALIDATED: Nirodha successfully recovered information from chaos.")
    else:
        print("\n❌ Breakthrough not detected in this run.")

if __name__ == "__main__":
    run_breakthrough_validation()
