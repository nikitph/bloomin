"""
Visualization: Plot Self-Healing Dynamics
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_self_healing_dynamics(history_recovery, history_curvature, save_path="results/self_healing_dynamics.png"):
    """
    Plot the self-healing dynamics showing:
    1. Recovery score (metric deviation) vs time
    2. Curvature entropy vs time
    
    Args:
        history_recovery: list of recovery scores
        history_curvature: list of curvature entropies
        save_path: path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    steps = np.arange(len(history_recovery))
    
    # Plot 1: Recovery Score (should decay exponentially)
    ax1.plot(steps, history_recovery, 'b-', linewidth=2, label='Recovery Score')
    ax1.set_xlabel('Healing Steps', fontsize=12)
    ax1.set_ylabel('Metric Deviation from G₀', fontsize=12)
    ax1.set_title('Phase 3: Self-Healing Dynamics - Recovery Score', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Add exponential fit if possible
    if len(history_recovery) > 10:
        try:
            # Fit exponential decay: y = a * exp(-b*x) + c
            from scipy.optimize import curve_fit
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            popt, _ = curve_fit(exp_decay, steps, history_recovery, 
                               p0=[history_recovery[0], 0.01, history_recovery[-1]],
                               maxfev=5000)
            
            fit_curve = exp_decay(steps, *popt)
            ax1.plot(steps, fit_curve, 'r--', linewidth=1.5, alpha=0.7, 
                    label=f'Exponential Fit (λ={popt[1]:.4f})')
            ax1.legend(fontsize=11)
        except:
            pass
    
    # Plot 2: Curvature Entropy (should stabilize after spike)
    ax2.plot(steps, history_curvature, 'g-', linewidth=2, label='Curvature Entropy')
    ax2.set_xlabel('Healing Steps', fontsize=12)
    ax2.set_ylabel('Spectral Entropy', fontsize=12)
    ax2.set_title('Phase 3: Self-Healing Dynamics - Curvature Smoothing', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved visualization to {save_path}")
    plt.close()


def plot_phase_comparison(entropy_genesis, entropy_damaged, entropy_healed, 
                          save_path="results/phase_comparison.png"):
    """
    Compare curvature entropy across all three phases
    
    Args:
        entropy_genesis: curvature entropy after Phase 1
        entropy_damaged: curvature entropy after Phase 2
        entropy_healed: curvature entropy after Phase 3
        save_path: path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    phases = ['Genesis\n(Healthy)', 'Injury\n(Damaged)', 'Healing\n(Recovered)']
    entropies = [entropy_genesis, entropy_damaged, entropy_healed]
    colors = ['green', 'red', 'blue']
    
    bars = ax.bar(phases, entropies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Curvature Entropy', fontsize=12)
    ax.set_title('Geometric State Across Experiment Phases', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, entropies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Saved phase comparison to {save_path}")
    plt.close()


if __name__ == "__main__":
    # Test visualization with dummy data
    import numpy as np
    
    # Simulate recovery (exponential decay)
    steps = np.arange(50)
    recovery = 100 * np.exp(-0.05 * steps) + np.random.randn(50) * 2
    
    # Simulate curvature (spike then stabilize)
    curvature = 5.0 + 2.0 * np.exp(-0.03 * steps) + np.random.randn(50) * 0.1
    
    plot_self_healing_dynamics(recovery, curvature, save_path="results/test_dynamics.png")
    plot_phase_comparison(4.5, 6.2, 4.7, save_path="results/test_comparison.png")
    
    print("Test visualizations created successfully!")
