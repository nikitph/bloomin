import torch
import torch.nn as nn
import math

class AdaptiveSemiring:
    """
    Adaptive Semiring that transitions from soft approximations during training
    to hard logic during inference/evaluation.
    
    Operations:
    - ⊕ (Plus): Max (Soft: LogSumExp)
    - ⊗ (Times): Min (Soft: -LogSumExp(-x))
    """
    def __init__(self, temperature_schedule=None):
        """
        Args:
            temperature_schedule: A callable that takes step/epoch and returns T.
                                If None, defaults to fixed T=1.0 (soft).
        """
        self.temperature_schedule = temperature_schedule
        self.temp = 1.0
        self.training = True

    def set_temperature(self, t):
        self.temp = t

    def step(self, step_idx):
        if self.temperature_schedule:
            self.temp = self.temperature_schedule(step_idx)

    def semiring_max(self, values, dim=-1, keepdim=False):
        """
        Soft-max that anneals to hard-max.
        Approximates max(values) using T * logsumexp(values / T).
        """
        if self.training and self.temp > 1e-5:
            return self.temp * torch.logsumexp(values / self.temp, dim=dim, keepdim=keepdim)
        else:
            return torch.max(values, dim=dim, keepdim=keepdim).values

    def semiring_compose(self, a, b):
        """
        Tropical semiring composition: ⊗ = min.
        Soft approximation: a ⊗ b ≈ -T * log(exp(-a/T) + exp(-b/T))
        
        Args:
            a: Tensor of shape (..., D)
            b: Tensor of shape (..., D)
            
        Returns:
            Tensor of shape (..., D) representing a ⊗ b (elementwise min)
        """
        if self.training and self.temp > 1e-5:
            # Numerical stability: use stack for logsumexp
            # -T * logsumexp([-a/T, -b/T])
            # equivalent to softmin
            # Manually broadcast a and b to common shape
            a, b = torch.broadcast_tensors(a, b)
            stacked = torch.stack([-a, -b], dim=0) / self.temp
            return -self.temp * torch.logsumexp(stacked, dim=0)
        else:
            return torch.min(a, b)

    def semiring_matmul(self, A, B):
        """
        Matrix multiplication in the tropical semiring.
        (A ⊗ B)[i,j] = ⊕_k (A[i,k] ⊗ B[k,j])
                     = max_k (min(A[i,k], B[k,j]))
                     
        Args:
            A: (B, N, K)
            B: (B, K, M)
            
        Returns:
            (B, N, M)
        """
        # Naive implementation O(N^3) - sufficient for small/sparse checks or development
        # For production, we'd want a darker magic optimization or block-sparse logic
        
        b_size, n, k = A.shape
        _, _, m = B.shape
        
        # Broadcast A to (B, N, 1, K)
        A_expanded = A.unsqueeze(2)
        # Broadcast B to (B, 1, M, K) (transpose last two dims of B effectively)
        B_expanded = B.transpose(1, 2).unsqueeze(1)
        
        # Elementwise min (⊗) over the K dimension
        # Result: (B, N, M, K)
        composed = self.semiring_compose(A_expanded, B_expanded)
        
        # Max (⊕) over the K dimension
        # Result: (B, N, M)
        return self.semiring_max(composed, dim=-1)

class ExponentialSchedule:
    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.decay = (end / start) ** (1 / steps)

    def __call__(self, step):
        if step >= self.steps:
            return self.end
        return self.start * (self.decay ** step)
