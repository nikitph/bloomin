from dataclasses import dataclass
import time

@dataclass
class SpectralState:
    delta_O_hat: float = 0.0       # execution stability proxy
    delta_W_hat: float = 0.0       # semantic stability proxy
    W_cross_hat: float = 0.0       # coupling work
    H_hat: float = 0.0             # hallucination index
    
    stable_count: int = 0          # consecutive stable observations
    phase: str = "FLUID"           # "FLUID", "HYBRID", "SOLID"
    
    last_updated: float = 0.0

class PhaseRegulator:
    def __init__(self, H_threshold=0.1, freeze_T=5, melt_T=2):
        # Using smaller T values for PoC speed
        self.H_threshold = H_threshold
        self.freeze_T = freeze_T
        self.melt_T = melt_T

    def update_metrics(self, state: SpectralState, delta_O, delta_W, H):
        state.delta_O_hat = delta_O
        state.delta_W_hat = delta_W
        state.H_hat = H
        state.last_updated = time.time()
        
        # update stability count
        if state.H_hat < self.H_threshold:
            state.stable_count += 1
        else:
            state.stable_count = max(0, state.stable_count - 1)

    def transition(self, state: SpectralState):
        """
        Determines if a phase change is needed.
        """
        # FLUID → SOLID (Freezing)
        if state.phase != "SOLID" and state.stable_count >= self.freeze_T:
            old_phase = state.phase
            state.phase = "SOLID"
            state.stable_count = 0
            return f"FREEZE ({old_phase} -> SOLID)"

        # SOLID → FLUID (Melting)
        if state.phase == "SOLID" and state.H_hat > 0.5: # Lower threshold to melt for PoC
            state.phase = "FLUID"
            state.stable_count = 0
            return "MELT (SOLID -> FLUID)"

        # Intermediate: FLUID → HYBRID
        if state.phase == "FLUID" and state.stable_count >= self.melt_T:
            state.phase = "HYBRID"
            return "HYBRIDIZE (FLUID -> HYBRID)"

        return "NO_CHANGE"
