import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import random_peaked_distribution, set_seed
from gauge_math import GaugeMath

# Mock Agent to match user's snippet structure
class MockAgent:
    def __init__(self, dim=128):
        self.memory = MockMemory(dim)

class MockMemory:
    def __init__(self, dim):
        self.dim = dim
        self.prototypes = {}
        # Initialize Primaries
        self.prototypes['Red'] = random_peaked_distribution(dim, peak_loc=20)
        self.prototypes['Blue'] = random_peaked_distribution(dim, peak_loc=50)
        self.prototypes['Yellow'] = random_peaked_distribution(dim, peak_loc=80)
        
    def get_prototype(self, name):
        return self.prototypes[name]
        
    def fisher_distance(self, p, q):
        return GaugeMath.distance(p, q).item()

def power_mean(p, q, r, epsilon=1e-9):
    """
    Computes Generalized Power Mean M_r(p, q).
    Handles limits r->0 (Geometric), r->inf (Max), r->-inf (Min).
    """
    p = torch.clamp(p, epsilon, 1.0)
    q = torch.clamp(q, epsilon, 1.0)
    
    if abs(r) < 1e-2: # Geometric Mean Limit
        mix = torch.sqrt(p * q)
    elif r == 1:      # Arithmetic Mean
        mix = (p + q) / 2
    elif r > 10:      # Max Limit
        mix = torch.max(p, q)
    elif r < -10:     # Min Limit
        mix = torch.min(p, q)
    else:             # General Case
        # Ensure bases are positive for fractional powers
        mix = ((p**r + q**r) / 2) ** (1/r)
        
    return mix / torch.sum(mix) # Normalize

def reconstruction_test(agent, r_values):
    """
    Measures how well 'Red' can be recovered from 'Purple' 
    created with different mixing physics (r).
    """
    p_red = agent.memory.get_prototype("Red")
    p_blue = agent.memory.get_prototype("Blue")
    p_yellow = agent.memory.get_prototype("Yellow") # Needed for intersection context
    
    errors = []
    entropies = []
    
    print(f"{'r':>6} | {'Entropy':>8} | {'Recov Err':>9} | {'Phase'}")
    print("-" * 45)

    for r in r_values:
        # 1. Forward: Mix Red + Blue using physics r
        # Purple_r = Red (op) Blue
        try:
            p_purple_r = power_mean(p_red, p_blue, r)
            
            # 2. Forward: Mix Red + Yellow using physics r
            # Orange_r = Red (op) Yellow
            p_orange_r = power_mean(p_red, p_yellow, r)
            
            # 3. Backward: Recover Red via Intersection
            # Red_hat = Purple_r INTERSECT Orange_r
            # We use geometric mean for intersection (proven optimal for recovery)
            p_red_hat = power_mean(p_purple_r, p_orange_r, 0) 
            
            # 4. Measure Metrics
            # Reconstruction Error (Fisher Distance)
            error = agent.memory.fisher_distance(p_red, p_red_hat)
            
            # Entropy of the mix (Thermodynamic state)
            S = -torch.sum(p_purple_r * torch.log(p_purple_r + 1e-9)).item()
            
            phase = "Constructive" if r > 0.01 else ("Geometric" if abs(r)<0.01 else "Destructive")
            print(f"{r:6.2f} | {S:8.4f} | {error:9.4f} | {phase}")
            
            errors.append(error)
            entropies.append(S)
        except Exception as e:
            print(f"{r:6.2f} | ERROR: {e}")
            errors.append(float('nan'))
            entropies.append(float('nan'))
        
    return errors, entropies

if __name__ == "__main__":
    set_seed(42)
    agent = MockAgent(dim=128)
    
    # Execution
    # Avoid exactly 0 to test the limit handling, but include it via linspace if robust
    r_range = np.linspace(-2.0, 2.0, 41) # Sweep from Harmonic to Quadratic
    errors, entropies = reconstruction_test(agent, r_range)

    # Plotting the "Well of Meaning"
    plt.figure(figsize=(10, 6))
    plt.plot(r_range, errors, 'o-', linewidth=2, color='crimson', label='Reconstruction Error')
    plt.axvline(x=0, color='black', linestyle='--', label='Geometric Mean (r=0)')
    plt.axvline(x=1, color='gray', linestyle=':', label='Arithmetic Mean (r=1)')
    plt.title("The Thermodynamic Phase Transition of Meaning")
    plt.xlabel("Mixing Parameter r (Negative=Destructive, Positive=Constructive)")
    plt.ylabel("Recovery Error (Fisher Distance)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = "r_sweep_plot.png"
    plt.savefig(output_path)
    print(f"\nPlot saved to {output_path}")
