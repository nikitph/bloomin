import torch
import numpy as np
import matplotlib.pyplot as plt
from nirodha.core import CognitiveState, YogaRegulator
from cross_model_yoga import TransformerDynamics, DiffusionDynamics, GNNDynamics

def calculate_energy(state: CognitiveState):
    return torch.norm(state.C - state.C0)**2

def run_lyapunov_experiment():
    print("Starting Lyapunov Verification Experiment...")
    dim = 512
    betas = [0.0, 0.1, 0.5, 1.0, 2.0] # 0.0 is baseline (unregulated)
    scales = [1, 5, 20] # depth * recurrence
    architectures = {
        "Transformer": TransformerDynamics(dim),
        "Diffusion": DiffusionDynamics(dim),
        "GNN": GNNDynamics(dim)
    }

    T = 100
    results = {}

    for arch_name, dynamics in architectures.items():
        results[arch_name] = {}
        for scale in scales:
            results[arch_name][scale] = {}
            for beta in betas:
                nirodha = YogaRegulator(beta=beta)
                C0 = torch.randn(dim)
                state = CognitiveState(C0)
                
                energy_deltas = []
                
                for t in range(T):
                    E_t = calculate_energy(state)
                    
                    # Dynamics step
                    # To simulate scale, we can repeat the dynamics or amplify delta
                    raw_delta = dynamics.forward(state.C)
                    delta = raw_delta * (1 + 0.1 * scale) # Scale amplifies the update
                    
                    if beta > 0:
                        state = nirodha(state, delta)
                    else:
                        state.C = state.C + delta # Simple residual baseline
                    
                    E_next = calculate_energy(state)
                    energy_deltas.append((E_next - E_t).item())
                
                results[arch_name][scale][beta] = {
                    "mean_delta": np.mean(energy_deltas),
                    "var_delta": np.var(energy_deltas),
                    "deltas": energy_deltas
                }

    # Plotting: Expectation of Delta E vs Beta
    plt.figure(figsize=(15, 5))
    for i, arch_name in enumerate(architectures.keys()):
        plt.subplot(1, 3, i+1)
        for scale in scales:
            means = [results[arch_name][scale][b]['mean_delta'] for b in betas]
            plt.plot(betas, means, marker='o', label=f'Scale {scale}')
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f'{arch_name}: $\mathbb{{E}}[\Delta E_t]$ vs $\\beta$')
        plt.xlabel('$\\beta$')
        plt.ylabel('Mean $\Delta E$')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('lyapunov_expectation.png')

    # Plotting: Variance of Delta E vs Scale (at fixed beta=0.5)
    plt.figure(figsize=(15, 5))
    target_beta = 0.5
    for i, arch_name in enumerate(architectures.keys()):
        plt.subplot(1, 3, i+1)
        
        # Nirodha vs Baseline (beta=0)
        nirodha_vars = [results[arch_name][s][target_beta]['var_delta'] for s in scales]
        baseline_vars = [results[arch_name][s][0.0]['var_delta'] for s in scales]
        
        plt.plot(scales, nirodha_vars, marker='o', label='Nirodha (β=0.5)')
        plt.plot(scales, baseline_vars, marker='x', label='Baseline (β=0)', linestyle='--')
        
        plt.title(f'{arch_name}: Var($\Delta E_t$) vs Scale')
        plt.xlabel('Scale')
        plt.ylabel('Variance')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig('lyapunov_variance.png')
    print("\nLyapunov plots saved: lyapunov_expectation.png, lyapunov_variance.png")

if __name__ == "__main__":
    run_lyapunov_experiment()
