import torch
import torch.nn as nn
import numpy as np
from .polynomial_score import PolynomialScoreField
from .riemannian_diffusion import RiemannianDiffusion
from .reaction_diffusion import ReactionDiffusionScore

class OBDSDiffusion(nn.Module):
    """
    The complete Shadow-Hierarchy Diffusion Model.
    """
    def __init__(self, data_dim=784, max_degree=5):
        super().__init__()
        self.data_dim = data_dim
        self.n_timesteps = 1000
        
        self.poly_score = PolynomialScoreField(max_degree, data_dim)
        self.riemann_geom = RiemannianDiffusion(data_dim)
        self.reaction_phys = ReactionDiffusionScore(data_dim)
        
        self.register_buffer('betas', self.cosine_beta_schedule(self.n_timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, axis=0))
        
    def forward(self, x_t, t):
        """
        Combined score function ∇log p(x_t).
        """
        t_norm = t.float() / self.n_timesteps
        s_poly = self.poly_score(x_t, t_norm)
        s_react = self.reaction_phys(x_t, t_norm)
        
        return s_poly + s_react

    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    @torch.no_grad()
    def sample(self, batch_size=16, num_steps=10):
        device = next(self.parameters()).device
        x = torch.randn(batch_size, self.data_dim, device=device)
        step_indices = torch.linspace(self.n_timesteps-1, 0, num_steps).long()
        
        for t_idx in step_indices:
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            score = self.forward(x, t)
            beta = self.betas[t_idx]
            x = x + beta * score + torch.randn_like(x) * torch.sqrt(beta)
            
        return x
