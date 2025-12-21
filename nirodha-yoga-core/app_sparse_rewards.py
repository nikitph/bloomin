import torch
import numpy as np
import matplotlib.pyplot as plt
from nirodha.core import CognitiveState, YogaRegulator

def run_sparse_reward_test():
    print("Starting Experiment 4: Sparse Reward Persistence")
    dim = 512
    horizon = 5000
    beta = 0.5
    reward_frequency = 1000 # Reward/Signal only every 1000 steps
    
    # 1. Hidden Sparse Signal (The "Reward" pattern)
    reward_signal = torch.randn(dim)
    reward_signal = reward_signal / torch.norm(reward_signal) * 0.1 # Weak but detectable
    
    # 2. Setup Systems
    C0 = torch.zeros(dim)
    
    # Nirodha System
    state_nirodha = CognitiveState(C0.clone())
    regulator = YogaRegulator(beta=beta)
    
    # Baseline System (Unregulated exploration/random walk)
    h_baseline = C0.clone()
    
    nirodha_discovery = []
    baseline_discovery = []
    
    print(f"Running simulation for {horizon} steps with sparse reward frequency {reward_frequency}...")
    
    for t in range(horizon):
        # Exploration noise
        noise = torch.randn(dim) * 1.0
        
        # Determine current update: Inject reward only if t is a multiple of reward_frequency
        is_reward_step = (t % reward_frequency == 0)
        update = noise + (reward_signal if is_reward_step else 0.0)
        
        # Nirodha Update
        state_nirodha = regulator(state_nirodha, update)
        
        # Baseline Update
        h_baseline = h_baseline + update
        
        # Measure discovery (alignment with reward_signal)
        # We subtract C0 to check direction of discovery
        sim_n = torch.nn.functional.cosine_similarity(state_nirodha.C - state_nirodha.C0, reward_signal, dim=0).item()
        sim_b = torch.nn.functional.cosine_similarity(h_baseline - C0, reward_signal, dim=0).item()
        
        nirodha_discovery.append(sim_n)
        baseline_discovery.append(sim_b)
        
        if is_reward_step:
            print(f"  Step {t:4} [REWARD]: Nirodha Discovery={sim_n:.6f}, Baseline Discovery={sim_b:.6f}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(nirodha_discovery, label='Nirodha (Information Persistence)', color='blue')
    plt.plot(baseline_discovery, label='Baseline (Information Loss)', color='red', linestyle='--')
    plt.title('Sparse Reward Persistence (Credit Assignment)')
    plt.xlabel('Time Steps')
    plt.ylabel('Alignment with Reward Signal')
    plt.legend()
    plt.grid(True)
    plt.savefig('app_sparse_rewards.png')
    print("\nResults saved to app_sparse_rewards.png")
    
    final_sim_n = np.mean(nirodha_discovery[-100:])
    final_sim_b = np.mean(baseline_discovery[-100:])
    
    if final_sim_n > final_sim_b + 0.05:
        print("✅ SUCCESS: Nirodha successfully preserved the sparse signal while baseline washed it out.")
    else:
        print("❌ Sparse signal differentiator not cleared.")

if __name__ == "__main__":
    run_sparse_reward_test()
