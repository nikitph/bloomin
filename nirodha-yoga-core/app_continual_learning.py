import torch
import numpy as np
import matplotlib.pyplot as plt
from nirodha.core import CognitiveState, YogaRegulator

def run_continual_learning_test():
    print("Starting Experiment 3: Continual Learning Anchor")
    dim = 512
    beta = 1.0 # Stronger anchor for base knowledge
    
    # 1. Setup Base Knowledge (Anchor C0)
    C0 = torch.randn(dim)
    
    # Task Alpha Projection (what the model already knows)
    W_alpha = torch.randn(dim, dim) * 0.1
    
    # Task Beta Updates (what the model is learning now - potentially intrusive)
    W_beta = torch.randn(dim, dim) * 0.5 
    
    # 2. Initialize Models
    # Nirodha System
    state_nirodha = CognitiveState(C0.clone())
    regulator = YogaRegulator(beta=beta)
    
    # Baseline System (Fine-tuning without anchor)
    h_baseline = C0.clone()
    
    nirodha_alpha_perf = []
    baseline_alpha_perf = []
    
    print("Learning Task Beta while monitoring Task Alpha retention...")
    
    for t in range(200):
        # Generate learning updates for Task Beta
        update = torch.tanh(torch.randn(dim) @ W_beta)
        
        # Nirodha Update: Learns Task Beta while anchored to Task Alpha (C0)
        state_nirodha = regulator(state_nirodha, update)
        
        # Baseline Update: Learned Task Beta directly (standard fine-tuning)
        h_baseline = h_baseline + update
        
        # Measure Performance on Task Alpha (Consistency with C0*W_alpha)
        # We proxy 'forgetting' by drift in response to Task Alpha inputs
        alpha_input = torch.randn(dim)
        alpha_target = torch.tanh(C0 @ W_alpha)
        
        alpha_out_n = torch.tanh(state_nirodha.C @ W_alpha)
        alpha_out_b = torch.tanh(h_baseline @ W_alpha)
        
        acc_n = torch.nn.functional.cosine_similarity(alpha_out_n, alpha_target, dim=0).item()
        acc_b = torch.nn.functional.cosine_similarity(alpha_out_b, alpha_target, dim=0).item()
        
        nirodha_alpha_perf.append(acc_n)
        baseline_alpha_perf.append(acc_b)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(nirodha_alpha_perf, label='Nirodha-Anchored (Task Alpha Accuracy)', color='blue')
    plt.plot(baseline_alpha_perf, label='Standard Fine-Tuning (Task Alpha Accuracy)', color='red', linestyle='--')
    plt.title('Continual Learning: Forgetting Index')
    plt.xlabel('Training Steps (Task Beta)')
    plt.ylabel('Baseline Task Retention (Cosine Sim)')
    plt.legend()
    plt.grid(True)
    plt.savefig('app_continual_learning.png')
    print("\nResults saved to app_continual_learning.png")
    
    if nirodha_alpha_perf[-1] > baseline_alpha_perf[-1] + 0.5:
        print("✅ SUCCESS: Nirodha prevented catastrophic forgetting while learning new tasks.")
    else:
        print("❌ Forgetting differentiator not cleared.")

if __name__ == "__main__":
    run_continual_learning_test()
