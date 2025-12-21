import torch
import numpy as np
from .core import CognitiveState, YogaRegulator

class SafetyGovernor:
    """
    Closed-loop safety governor that enforces the Safe Operating Envelope (SOE).
    Dynamically adjusts beta based on empirical energy variance and sensitivity.
    """
    def __init__(self, initial_beta=0.5, v_max=10.0, sensitivity_target=0.01):
        self.regulator = YogaRegulator(beta=initial_beta)
        self.v_max = v_max
        self.sensitivity_target = sensitivity_target
        self.history = []
        self.energy_history = []

    def step(self, state: CognitiveState, delta: torch.Tensor, current_sensitivity=0.0):
        # 1. Calculate energy before update
        e_t = torch.norm(state.C - state.C0)**2
        
        # 2. Apply regulated update
        state = self.regulator(state, delta)
        
        # 3. Calculate energy after update
        e_next = torch.norm(state.C - state.C0)**2
        delta_e = (e_next - e_t).item()
        
        self.energy_history.append(delta_e)
        if len(self.energy_history) > 50:
            self.energy_history.pop(0)

        # 4. Analyze SOE Conditions
        mean_de = np.mean(self.energy_history)
        var_de = np.var(self.energy_history) if len(self.energy_history) > 5 else 0.0

        # 5. SOE Control Policy
        beta_prev = self.regulator.nirodha.beta
        
        # IF Var(ΔE_t) > V_max THEN increase β
        if var_de > self.v_max:
            self.regulator.nirodha.beta += 0.1
            print(f"  [SOE-WARN] Variance {var_de:.4f} > V_max. Increased β to {self.regulator.nirodha.beta:.2f}")
        
        # IF sensitivity < target AND Var(ΔE_t) << V_max THEN decrease β slightly
        elif current_sensitivity < self.sensitivity_target and var_de < (self.v_max * 0.1):
            self.regulator.nirodha.beta = max(0.01, self.regulator.nirodha.beta - 0.01)
            # No print here to avoid flooding, but allows "reach" growth
            
        beta_new = self.regulator.nirodha.beta
        
        return state, {"mean_de": mean_de, "var_de": var_de, "beta": beta_new}

def test_soe_governor():
    print("Testing SOE Governor Closed-Loop Logic...")
    dim = 512
    v_max = 5.0
    governor = SafetyGovernor(initial_beta=0.1, v_max=v_max)
    
    C0 = torch.randn(dim)
    state = CognitiveState(C0)
    
    # Simulate a scenario where instability starts to grow
    for t in range(100):
        # As t increases, we inject more instability
        instability_scale = 0.5 if t < 50 else 5.0
        delta = torch.randn(dim) * instability_scale
        
        state, metrics = governor.step(state, delta)
        
        if t % 20 == 0:
            print(f"Step {t:3}: Beta={metrics['beta']:.2f}, Var(ΔE)={metrics['var_de']:.4f}")

    print("\nSOE Governor Test Complete.")

if __name__ == "__main__":
    test_soe_governor()
