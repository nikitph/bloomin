import torch
import torch.nn as nn

class PolynomialScoreField(nn.Module):
    """
    Layer 3: Polynomial Score Field S(x,t) ∈ ℂ[x,t].
    Represents the score as a sum of complex polynomial terms.
    """
    def __init__(self, max_degree=5, n_dims=784):
        super().__init__()
        self.max_degree = max_degree
        self.n_dims = n_dims
        
        # Coefficients in ℂ[x,t]
        # Shape: (max_degree + 1, n_dims)
        self.coeffs = nn.Parameter(
            torch.randn(max_degree + 1, n_dims, dtype=torch.complex64) * 0.01
        )
    
    def forward(self, x_t, t):
        """
        Evaluate polynomial score.
        x_t: [batch, n_dims]
        t: [batch] or scalar, normalized to [0, 1]
        """
        device = x_t.device
        if isinstance(t, torch.Tensor):
            t = t.view(-1, 1).to(torch.complex64)
        else:
            t = torch.tensor(t, dtype=torch.complex64, device=device).view(1, 1)
            
        x_complex = x_t.to(torch.complex64)
        score = torch.zeros_like(x_complex)
        
        # Evaluate using a simple loop
        # score = sum(c_d * x^d * t^d)
        for d in range(self.max_degree + 1):
            term = self.coeffs[d] * (x_complex ** d) * (t ** d)
            score += term
            
        return score.real  # Project to real score
