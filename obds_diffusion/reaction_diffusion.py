import torch
import torch.nn as nn

class ReactionDiffusionScore(nn.Module):
    """
    Layer 5: Reaction-Diffusion terms for multi-scale texture and edge preservation.
    """
    def __init__(self, n_dims=784, F=0.037, k=0.06):
        super().__init__()
        self.F = F 
        self.k = k
        self.lambda_rd = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x_t, t):
        """
        Compute reaction-diffusion force.
        """
        half = x_t.shape[1] // 2
        u = x_t[:, :half]
        v = x_t[:, half:2*half]
        
        # Gray-Scott reaction terms
        r_u = -u * (v**2) + self.F * (1.0 - u)
        r_v = u * (v**2) - (self.F + self.k) * v
        
        reaction_score = torch.cat([r_u, r_v], dim=1)
        
        if reaction_score.shape[1] < x_t.shape[1]:
            padding = torch.zeros(x_t.shape[0], x_t.shape[1] - reaction_score.shape[1], device=x_t.device)
            reaction_score = torch.cat([reaction_score, padding], dim=1)
            
        return self.lambda_rd * reaction_score
