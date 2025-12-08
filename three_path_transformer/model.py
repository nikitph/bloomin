import torch
import torch.nn as nn
import torch.nn.functional as F

class PowerLawSharpener(nn.Module):
    """
    Power law sharpening: p -> p^power / ||p^power||
    Used for active error correction in Slow Path and Sleep Path.
    """
    def __init__(self, power=2.0):
        super().__init__()
        self.power = power
        
    def forward(self, x, iters=1):
        """
        Apply sharpening.
        Args:
            x: [..., dim] Input tensor (usually logits or probabilities)
            iters: Number of sharpening iterations
        """
        # Ensure we are working with probabilities or near-probabilities
        # If input is raw logits, we might want to softmax first?
        # The design assumes embeddings ARE the representations.
        
        # We assume x is already somewhat positive/distribution-like from ReLU/Softmax
        # But to be safe for power law, we take abs()
        
        for _ in range(iters):
            x_sharp = torch.pow(torch.abs(x) + 1e-10, self.power)
            x_sharp = x_sharp / (x_sharp.sum(dim=-1, keepdim=True) + 1e-10)
            x = x_sharp
            
        return x

class TransformerBlock(nn.Module):
    """Standard Transformer Block"""
    def __init__(self, dim, n_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4*dim),
            nn.GELU(),
            nn.Linear(4*dim, dim)
        )
        self.ln2 = nn.LayerNorm(dim)
        
    def forward(self, x):
        # Self-Attention
        attn_out, _ = self.attention(x, x, x)
        x = self.ln1(x + attn_out)
        
        # Feed-Forward
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        
        return x

class ThreePathTransformer(nn.Module):
    """
    Three-Path Architecture:
    1. Fast Path: Standard Transformer
    2. Slow Path: Transformer + Active Sharpening
    3. Sleep Path: (Handled externally via global optimization)
    """
    def __init__(self, vocab_size, dim=128, n_heads=4, n_layers=4, sharpen_iters=10, sharpen_power=2.0):
        super().__init__()
        self.dim = dim
        self.sharpen_iters = sharpen_iters
        
        self.embedding = nn.Embedding(vocab_size, dim)
        
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads) for _ in range(n_layers)
        ])
        
        self.sharpener = PowerLawSharpener(power=sharpen_power)
        
    def encode(self, x, path='fast'):
        """
        Encode input tokens into a concept embedding.
        Args:
            x: [batch, seq_len] Token IDs
            path: 'fast' or 'slow'
        """
        # Embed
        h = self.embedding(x) # [batch, seq, dim]
        
        # Pass through layers
        for layer in self.layers:
            h = layer(h)
            
            # SLOW PATH: Apply sharpening after every layer
            if path == 'slow':
                h = self.sharpener(h, iters=self.sharpen_iters)
                
        # Pool to single embedding (mean pooling)
        h = h.mean(dim=1) # [batch, dim]
        
        # Final sharpening
        if path == 'slow':
            h = self.sharpener(h, iters=self.sharpen_iters)
            
        # Ensure output is a valid probability distribution for mixing
        # We use Softmax to enforce [0,1] and sum=1
        # BUT, if we want to preserve the "sharpened" nature, Softmax might flatten it.
        # So we use the sharpener's normalization logic (L1) instead of Softmax if we are in slow path?
        # Standard transformer usually outputs logits.
        # Let's simple Softmax for now, or assume the sharpening handles normalization.
        
        h = F.softmax(h, dim=-1)
        
        return h
        
    def mix(self, emb1, emb2, method='geometric'):
        """
        Mix two concept embeddings.
        Args:
            emb1, emb2: [batch, dim]
            method: 'geometric' or 'arithmetic'
        """
        if method == 'geometric':
            # Geometric: sqrt(p*q)
            mixed = torch.sqrt(emb1 * emb2 + 1e-10)
            mixed = mixed / (mixed.sum(dim=-1, keepdim=True) + 1e-10)
        elif method == 'arithmetic':
            mixed = (emb1 + emb2) / 2.0
        else:
            raise ValueError(f"Unknown mixing method: {method}")
            
        return mixed
