import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ============================================================
# 2. EUCLIDEAN WITNESS (R^m)
# ============================================================

class WitnessEuclidean(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        # x: (B, N, D)
        return self.proj(x)   # (B, N, m)


# ============================================================
# 3. SPHERICAL WITNESS (S^n-1)
# ============================================================

class WitnessSpherical(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        v = self.proj(x)
        v = F.normalize(v, dim=-1)  # project to unit sphere
        return v  # (B, N, n)


# ============================================================
# 4. HYPERBOLIC WITNESS (Poincaré ball H^p)
# ============================================================

class WitnessHyperbolic(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        v = torch.tanh(self.proj(x))  # maps inside Poincaré ball
        return v  # (B, N, p)

    @staticmethod
    def poincare_distance(u, v):
        # u, v: (B, N, p)
        # We need to handle broadcasting for pairwise distance if inputs are (B, N, p)
        # If u and v are the same tensor (B, N, p), we want (B, N, N)
        
        # Check if we need to unsqueeze for pairwise
        if u.dim() == 3 and v.dim() == 3:
             # u: (B, N, p), v: (B, M, p) -> (B, N, M)
            diff = u.unsqueeze(2) - v.unsqueeze(1)  # (B, N, M, p)
            diff_norm_sq = (diff ** 2).sum(-1)

            u_norm_sq = (u ** 2).sum(-1).unsqueeze(-1)  # (B, N, 1)
            v_norm_sq = (v ** 2).sum(-1).unsqueeze(-2)  # (B, 1, M)
        else:
             # Fallback or other shapes, assume standard pairwise logic matches
             diff = u - v
             diff_norm_sq = (diff ** 2).sum(-1)
             u_norm_sq = (u ** 2).sum(-1)
             v_norm_sq = (v ** 2).sum(-1)

        denom = (1 - u_norm_sq) * (1 - v_norm_sq) + 1e-7
        arg = 1 + 2 * diff_norm_sq / denom
        arg = torch.clamp(arg, min=1.0 + 1e-6)

        return torch.acosh(arg)


# ============================================================
# 5. FUNCTIONAL WITNESS (RFF: Random Fourier Features)
# ============================================================

class WitnessFunctionalRFF(nn.Module):
    def __init__(self, dim_in, rff_dim=256, sigma=1.0):
        """
        RFF Witness:
            φ(x) = sqrt(2/D) * [cos(Wx), sin(Wx)]
        """
        super().__init__()
        self.rff_dim = rff_dim
        self.sigma = sigma

        # W ~ N(0, sigma^(-2) I)
        W = torch.randn(dim_in, rff_dim // 2) * (1.0 / sigma)
        self.register_buffer("W", W)

    def forward(self, x):
        # x: (B, N, D)
        proj = torch.matmul(x, self.W)   # (B, N, rff/2)
        cos = torch.cos(proj)
        sin = torch.sin(proj)
        phi = torch.cat([cos, sin], dim=-1) * math.sqrt(2.0 / self.rff_dim)
        return phi  # (B, N, rff_dim)


# ============================================================
# 6. REWA DISTANCE MODULE
# ============================================================

class REWADistance(nn.Module):
    """
    Combine Euclidean + Spherical + Hyperbolic + Functional RFF distances.
    d_total = α * d_E + β * d_S + γ * d_H + δ * d_F
    """
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, temperature=1.0):
        super().__init__()

        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.delta = nn.Parameter(torch.tensor(delta))

        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(self, w_E, w_S, w_H, w_F):
        """
        Inputs:
            w_E: Euclidean witness     (B, N, m)
            w_S: Spherical witness     (B, N, n)
            w_H: Hyperbolic witness    (B, N, p)
            w_F: Functional witness    (B, N, rff)

        Output:
            Distance matrix: (B, N, N)
        """

        # --- Euclidean distance ---
        d_E = torch.cdist(w_E, w_E, p=2)

        # --- Spherical distance ---
        # w_S is normalized
        cos_sim = torch.bmm(w_S, w_S.transpose(1, 2))
        cos_sim = torch.clamp(cos_sim, -1 + 1e-6, 1 - 1e-6)
        d_S = torch.acos(cos_sim)

        # --- Hyperbolic distance ---
        d_H = WitnessHyperbolic.poincare_distance(w_H, w_H)

        # --- Functional (RFF) Euclidean ---
        d_F = torch.cdist(w_F, w_F, p=2)

        # Combine with softplus to ensure positivity
        α = F.softplus(self.alpha)
        β = F.softplus(self.beta)
        γ = F.softplus(self.gamma)
        δ = F.softplus(self.delta)

        d_total = α * d_E + β * d_S + γ * d_H + δ * d_F

        return d_total / F.softplus(self.temperature)
