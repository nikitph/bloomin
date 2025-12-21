import torch
import torch.nn as nn

class CognitiveState:
    """
    Cognitive State Wrapper.
    Maintains a decoupled anchor C0 for structural referencing.
    """
    def __init__(self, latent: torch.Tensor):
        self.C = latent                  # current state
        self.C0 = latent.detach().clone() # fixed anchor (Lyapunov center)

    def update(self, new_C: torch.Tensor):
        """Autograd-safe update: creates a new state instance or updates leaf."""
        new_state = CognitiveState(self.C0)
        new_state.C = new_C
        new_state.C0 = self.C0 # Preserve original anchor
        return new_state

class VrittiExtractor:
    """Isolates fluctuations relative to the anchor."""
    def __call__(self, C: torch.Tensor, C0: torch.Tensor):
        return C - C0

class Nirodha(nn.Module):
    """
    Nirodha Operator (Non-Destructive Suppression).
    
    Mathematical Properties:
    1. Identity at Origin: lim_{V->0} dN/dV = 1
    2. Boundedness: |N(V)| < 1/beta for all V
    3. Contractivity: |dN/dV| <= 1 for beta >= 0 (Lyapunov condition)
    """
    def __init__(self, beta=1.0):
        super().__init__()
        # Ensure beta is non-negative to maintain contractivity
        self.register_buffer('beta', torch.tensor(max(0.0, float(beta))))

    def forward(self, V):
        # f(x) = x / (1 + beta * |x|)
        # Differentiable almost everywhere, continuous everywhere.
        return V / (1.0 + self.beta * torch.abs(V) + 1e-12) # epsilon for extreme stability

class YogaRegulator(nn.Module):
    """
    Regulated Update Logic.
    Ensures state transitions are contractive toward the anchor C0.
    """
    def __init__(self, beta=1.0):
        super().__init__()
        self.vritti = VrittiExtractor()
        self.nirodha = Nirodha(beta)

    def forward(self, state: CognitiveState, model_update: torch.Tensor):
        """
        Calculates regulated C_{t+1}.
        Does NOT modify the input state in-place to preserve autograd graphs.
        """
        # 1. Proposed next state (unregulated)
        C_tilde = state.C + model_update

        # 2. Extract fluctuations (relative to anchor)
        V = self.vritti(C_tilde, state.C0)

        # 3. Apply Contractive Operator
        V_suppressed = self.nirodha(V)

        # 4. Reconstruct Regulated State
        C_new = state.C0 + V_suppressed
        
        # Return new state instance (safe for backprop)
        return state.update(C_new)
