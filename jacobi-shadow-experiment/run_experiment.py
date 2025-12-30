import torch
import matplotlib.pyplot as plt
import numpy as np
from jacobi_dynamics import augmented_dynamics, lorenz_dynamics, rk4_step

def run_experiment():
    # Parameters
    dt = 0.01
    steps = 2000
    epsilon = 1e-6
    
    # Initial state (on the attractor roughly)
    x0 = torch.tensor([0.1, 0.0, 0.0])
    # Initial perturbation vector
    v0 = torch.tensor([1.0, 0.0, 0.0])
    
    # True trajectory
    x_true = x0.clone()
    # Perturbed trajectory (Shadow)
    x_perturbed = x0 + epsilon * v0
    # Jacobi Shadow state [x, v]
    state_jacobi = torch.cat([x0, v0])
    
    history_error = []
    history_jacobi = []
    history_time = []
    
    for i in range(steps):
        t = i * dt
        
        # 1. Evolve True Trajectory
        x_true = rk4_step(lorenz_dynamics, x_true, dt)
        
        # 2. Evolve Perturbed Trajectory
        x_perturbed = rk4_step(lorenz_dynamics, x_perturbed, dt)
        
        # 3. Evolve Jacobi Shadow
        state_jacobi = rk4_step(augmented_dynamics, state_jacobi, dt)
        
        # Calculate Error Normalized by epsilon to keep scales comparable initially
        error = torch.norm(x_true - x_perturbed) / epsilon
        # Calculate Jacobi Norm
        j_norm = torch.norm(state_jacobi[3:])
        
        history_error.append(error.item())
        history_jacobi.append(j_norm.item())
        history_time.append(t)
        
    # Plotting
    plt.figure(figsize=(12, 7))
    plt.style.use('dark_background')
    
    plt.plot(history_time, history_jacobi, label='Jacobi Norm (Forecast Fragility)', color='#00ffcc', linewidth=2)
    plt.plot(history_time, history_error, label='Normalized Prediction Error', color='#ff3366', linestyle='--', alpha=0.8)
    
    # Annotate breakdown point
    # Find where Jacobi crosses a significant threshold (e.g., 10^3)
    threshold = 1e2
    crossing_idx = next((i for i, x in enumerate(history_jacobi) if x > threshold), None)
    
    if crossing_idx:
        plt.axvline(x=history_time[crossing_idx], color='white', linestyle=':', alpha=0.5)
        plt.text(history_time[crossing_idx], threshold, " Jacobi crossing", color='white', verticalalignment='bottom')
        
    plt.yscale('log')
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('Magnitude (Log Scale)', fontsize=12)
    plt.title('The Killer Plot: Jacobi Norm vs. Forecast Error', fontsize=16, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", ls="-", alpha=0.1)
    
    plt.tight_layout()
    plt.savefig('jacobi-shadow-experiment/jacobi_results.png')
    print("Experiment complete. Plot saved to jacobi-shadow-experiment/jacobi_results.png")

if __name__ == "__main__":
    run_experiment()
