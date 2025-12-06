"""
Adversarial Hybrid REWA Encoder
===============================

The core model that achieved 78.6% zero-shot recall.
Uses adversarial training to force learned features to generalize like random projections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F



class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from Domain Adversarial Neural Networks.
    Forward: identity
    Backward: negate gradient
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversal(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class FeatureAugmentation(nn.Module):
    """
    Augment features to improve generalization.
    """
    def __init__(self, dropout=0.1, noise_std=0.05, cutout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.noise_std = noise_std
        self.cutout_prob = cutout_prob
    
    def forward(self, x):
        if not self.training:
            return x
        
        # Dropout
        x = self.dropout(x)
        
        # Gaussian noise
        if torch.rand(1) < 0.5:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # Feature cutout (zero out random dimensions)
        if torch.rand(1) < self.cutout_prob:
            cutout_size = int(0.1 * x.shape[-1])
            start_idx = torch.randint(0, x.shape[-1] - cutout_size, (1,)).item()
            x[..., start_idx:start_idx+cutout_size] = 0
        
        return x

class AdversarialHybridREWAEncoder(nn.Module):
    def __init__(self, d_model=768, m_dim=256):
        super().__init__()
        
        # Standard hybrid encoder
        self.random_proj = nn.Linear(d_model, m_dim // 2, bias=False)
        nn.init.orthogonal_(self.random_proj.weight)
        self.random_proj.weight.requires_grad = False
        
        self.learned_proj = nn.Sequential(
            nn.Linear(d_model, m_dim // 2),
            nn.LayerNorm(m_dim // 2),
            FeatureAugmentation(dropout=0.2, noise_std=0.05),
        )
        
        # Gradient reversal before discriminator
        self.gradient_reversal = GradientReversal(lambda_=1.0)
        
        # DISCRIMINATOR
        self.discriminator = nn.Sequential(
            nn.Linear(m_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
    
    def forward(self, x, add_noise=None, **kwargs):
        random_part = self.random_proj(x)
        learned_part = self.learned_proj(x)
        
        # Combine
        combined = torch.cat([random_part, learned_part], dim=-1)
        return F.normalize(combined, dim=-1)
    
    def forward_with_adversarial(self, x):
        """Joint forward pass"""
        # Standard encoding
        encoded = self.forward(x, add_noise=False)
        
        # Extract learned features for discriminator
        learned_features = self.learned_proj(x)
        
        # Apply gradient reversal
        reversed_features = self.gradient_reversal(learned_features)
        
        # Discriminator prediction
        disc_pred = self.discriminator(reversed_features)
        
        return encoded, disc_pred

