import torch
import numpy as np
import matplotlib.pyplot as plt
from nirodha.core import CognitiveState, YogaRegulator
from siddhi_harness import probe_weak_signal_detection

class TransformerDynamics:
    """Mock Transformer dynamics: Residual sum h = h + attention(h)"""
    def __init__(self, dim=512):
        self.W = torch.randn(dim, dim) * 0.05
    def forward(self, h):
        return torch.tanh(h @ self.W) + torch.randn_like(h) * 0.05

class DiffusionDynamics:
    """Mock Diffusion dynamics: Latent state update with noise injection"""
    def __init__(self, dim=512):
        self.W = torch.randn(dim, dim) * 0.02
    def forward(self, x):
        # Denoising-like step + noise
        noise = torch.randn_like(x) * 0.1
        return (x @ self.W) + noise

class GNNDynamics:
    """Mock GNN dynamics: Message passing/aggregation"""
    def __init__(self, dim=512):
        self.W = torch.randn(dim, dim) * 0.08
    def forward(self, state):
        # Propagation-like step
        return torch.sin(state @ self.W) + torch.randn_like(state) * 0.02

def test_architecture(arch_name, dynamics_model, dim=512, beta=0.5):
    print(f"\n--- Testing Architecture: {arch_name} ---")
    
    # 1. Nirodha Regulated
    nirodha = YogaRegulator(beta=beta)
    C0 = torch.randn(dim)
    state = CognitiveState(C0)
    
    # 2. Unregulated Baseline
    baseline = C0.detach().clone()
    
    nirodha_sensitivities = []
    baseline_sensitivities = []
    
    nirodha_norms = []
    baseline_norms = []

    T = 50
    for _ in range(T):
        # Nirodha Step
        delta = dynamics_model.forward(state.C)
        state = nirodha(state, delta)
        
        # Baseline Step
        delta_b = dynamics_model.forward(baseline)
        baseline = baseline + delta_b # Simple residual update
        
        # Log Metrics
        m_n = probe_weak_signal_detection(state)
        # For baseline, we wrap it momentarily for probe
        m_b = probe_weak_signal_detection(CognitiveState(baseline))
        
        nirodha_sensitivities.append(m_n['tail_sensitivity'])
        baseline_sensitivities.append(m_b['tail_sensitivity'])
        
        nirodha_norms.append(torch.norm(state.C).item())
        baseline_norms.append(torch.norm(baseline).item())

    # Check stability (End Norm)
    print(f"  Nirodha Final Norm: {nirodha_norms[-1]:.4f}")
    print(f"  Baseline Final Norm: {baseline_norms[-1]:.4f}")
    
    is_stable = nirodha_norms[-1] < baseline_norms[-1] * 0.5 or baseline_norms[-1] > 1000
    print(f"  Stability Benefit Detected: {is_stable}")
    
    return {
        "nirodha_s": nirodha_sensitivities,
        "baseline_s": baseline_sensitivities,
        "nirodha_n": nirodha_norms,
        "baseline_n": baseline_norms
    }

def run_cross_model_replication():
    dim = 512
    models = {
        "Transformer": TransformerDynamics(dim),
        "Diffusion": DiffusionDynamics(dim),
        "GNN": GNNDynamics(dim)
    }
    
    results = {}
    for name, model in models.items():
        results[name] = test_architecture(name, model, dim=dim)

    # Visualization
    plt.figure(figsize=(15, 10))
    for i, (name, res) in enumerate(results.items()):
        plt.subplot(3, 2, 2*i + 1)
        plt.plot(res['nirodha_n'], label='Nirodha Norm', color='blue')
        plt.plot(res['baseline_n'], label='Baseline Norm', color='red', linestyle='--')
        plt.title(f'{name}: Norm Stability')
        plt.legend()
        
        plt.subplot(3, 2, 2*i + 2)
        plt.plot(res['nirodha_s'], label='Nirodha Sensitivity', color='blue')
        plt.plot(res['baseline_s'], label='Baseline Sensitivity', color='red', linestyle='--')
        plt.title(f'{name}: Tail Sensitivity')
        plt.legend()

    plt.tight_layout()
    plt.savefig('cross_model_yoga_results.png')
    print("\nCross-model results saved to cross_model_yoga_results.png")

if __name__ == "__main__":
    run_cross_model_replication()
