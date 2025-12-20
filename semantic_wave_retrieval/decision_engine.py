import torch
import torch.nn.functional as F
from typing import Optional
from .engine import WaveRetrievalEngine
from .utils import sparse_dense_mul

class DecisionEngine(WaveRetrievalEngine):
    def __init__(self, data: torch.Tensor, k_neighbors: int = 15, use_cuda: bool = False):
        super().__init__(data, k_neighbors, use_cuda)

    def schrodinger_step(self, psi: torch.Tensor, V_loss: torch.Tensor, T_schrod: int = 50, dt: float = 0.01):
        """
        Step 1: Imaginary Schrodinger (Global Optimization/Tunneling)
        PDE: partial_tau psi = Delta psi - V(x)psi
        Delta = -L
        So: dpsi/dt = -L psi - V*psi
        """
        for t in range(T_schrod):
            # L psi calculation
            L_psi = sparse_dense_mul(self.L, psi)
            
            # Update: psi += dt * (-L_psi - V * psi)
            # The diffusion term (Delta psi) is -L psi
            update = -L_psi - (V_loss * psi)
            psi = psi + dt * update
            
            # Renormalize (keep total probability mass constant or just prevent explode)
            norm = torch.norm(psi) + 1e-9
            psi = psi / norm
            
        return psi

    def fisher_kpp_step(self, u: torch.Tensor, T_kpp: int = 50, dt: float = 0.1):
        """
        Step 3: Fisher-KPP (Autocatalytic Selection)
        PDE: u_t = Delta u + u(1 - u)
        """
        for t in range(T_kpp):
            # Delta u = -L u
            delta_u = -sparse_dense_mul(self.L, u)
            
            # Reaction: u(1-u)
            reaction = u * (1 - u)
            
            u = u + dt * (delta_u + reaction)
            
            # Clamp to [0, 1] range usually for logistic growth stability
            u = torch.clamp(u, 0.0, 1.0) # Soft clamp
            
        return u

    def decide(self, loss_landscape: torch.Tensor, top_k: int = 1, **kwargs):
        """
        End-to-End Pipeline
        """
        # 0. Initialize Uniform Superposition
        N = self.data.shape[0]
        psi = torch.ones(N).to(self.device) # / N # Start mostly uniform
        
        # 1. Imaginary Schrodinger
        psi = self.schrodinger_step(psi, loss_landscape, **kwargs.get('schrod_params', {}))
        
        # 2. Map to Decision Field
        u = torch.abs(psi) # Real wavefunction
        
        # 3. Fisher-KPP
        u = self.fisher_kpp_step(u, **kwargs.get('kpp_params', {}))
        
        # 4. Decision Extraction
        # Return probability distribution / top decisions
        sorted_indices = torch.argsort(u, descending=True)
        return sorted_indices[:top_k], u
