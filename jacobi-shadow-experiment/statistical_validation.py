import torch
import matplotlib.pyplot as plt
import numpy as np
from jacobi_dynamics import augmented_dynamics, lorenz_dynamics, rk4_step

def get_random_attractor_point():
    """Evolve from random noise to ensure we are on the attractor."""
    x = torch.randn(3) * 10
    # Evolve for 2000 steps to reach attractor
    dt = 0.01
    for _ in range(1000):
        x = rk4_step(lorenz_dynamics, x, dt)
    return x

def run_trial(x0, dt=0.01, steps=3000, epsilon=1e-6, threshold=1e2):
    v0 = torch.randn(3)
    v0 = v0 / torch.norm(v0) # Unit perturbation
    
    x_true = x0.clone()
    x_perturbed = x0 + epsilon * v0
    state_jacobi = torch.cat([x0, v0])
    
    errors = []
    jacobi_norms = []
    t_cross = None
    
    for i in range(steps):
        # Evolve
        x_true = rk4_step(lorenz_dynamics, x_true, dt)
        x_perturbed = rk4_step(lorenz_dynamics, x_perturbed, dt)
        state_jacobi = rk4_step(augmented_dynamics, state_jacobi, dt)
        
        err = torch.norm(x_true - x_perturbed) / epsilon
        j_norm = torch.norm(state_jacobi[3:])
        
        errors.append(err.item())
        jacobi_norms.append(j_norm.item())
        
        if t_cross is None and j_norm > threshold:
            t_cross = i
            
    return errors, jacobi_norms, t_cross

def statistical_experiment():
    n_trials = 20
    dt = 0.01
    all_aligned_errors = []
    window_before = 200 # steps before crossing
    window_after = 500  # steps after crossing
    
    print(f"Running {n_trials} trials...")
    
    trials_done = 0
    while trials_done < n_trials:
        x0 = get_random_attractor_point()
        errors, j_norms, t_cross = run_trial(x0)
        
        # Check if we have enough context around t_cross
        if t_cross and t_cross > window_before and (t_cross + window_after) < len(errors):
            # Extract aligned window
            aligned_err = errors[t_cross - window_before : t_cross + window_after]
            all_aligned_errors.append(aligned_err)
            trials_done += 1
            print(f"Trial {trials_done}/{n_trials} complete.")
        else:
            # If t_cross happens too early or late, retry with new IC
            continue
            
    all_aligned_errors = np.array(all_aligned_errors)
    mean_error = np.mean(all_aligned_errors, axis=0)
    std_error = np.std(all_aligned_errors, axis=0)
    
    time_aligned = np.arange(-window_before, window_after) * dt
    
    # Plotting
    plt.figure(figsize=(12, 7))
    plt.style.use('dark_background')
    
    plt.plot(time_aligned, mean_error, color='#ff3366', label='Mean Prediction Error', linewidth=2)
    plt.fill_between(time_aligned, mean_error - std_error, mean_error + std_error, 
                     color='#ff3366', alpha=0.2, label='± 1 Std Dev')
    
    plt.axvline(x=0, color='white', linestyle='--', alpha=0.8, label='Jacobi Crossing ($t_{cross}$)')
    
    plt.yscale('log')
    plt.xlabel('Time Relative to Crossing (t - $t_{cross}$)', fontsize=12)
    plt.ylabel('Normalized Prediction Error (Log Scale)', fontsize=12)
    plt.title('Statistical Inevitability: Error Evolution Aligned by Jacobi Crossing', fontsize=16, pad=20)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", ls="-", alpha=0.1)
    
    plt.tight_layout()
    plt.savefig('jacobi-shadow-experiment/statistical_inevitability.png')
    print("Statistical experiment complete. Plot saved to jacobi-shadow-experiment/statistical_inevitability.png")

if __name__ == "__main__":
    statistical_experiment()
