"""
Standard Transformer for Binary Tree Path Prediction

This is a baseline vanilla Transformer with standard self-attention.
We expect this to struggle on depth-20 trees unless d is very large (≈1e6).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        """
        Args:
            x: [B, N, D]
        Returns:
            [B, N, D]
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """Standard multi-head self-attention."""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, N, D]
            mask: [B, N] boolean mask (True = keep, False = ignore)
        
        Returns:
            [B, N, D]
        """
        B, N, D = x.shape
        
        # Project and reshape to [B, n_heads, N, d_k]
        q = self.q_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores: [B, n_heads, N, N]
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to [B, 1, 1, N] for broadcasting
            mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax and apply to values
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)  # [B, n_heads, N, d_k]
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-norm."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-norm architecture
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class StandardTransformer(nn.Module):
    """
    Standard Transformer for binary tree path prediction.
    
    Architecture:
    - Token embedding
    - Positional encoding
    - N Transformer blocks
    - Final pooling + prediction head
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        max_len=5000,
        num_classes=1000,
        dropout=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Embedding layers
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: [B, N] token indices
            attention_mask: [B, N] boolean mask
        
        Returns:
            logits: [B, num_classes]
        """
        # Embed tokens
        x = self.token_embed(input_ids)  # [B, N, D]
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        x = self.norm(x)
        
        # Pool: take the last valid token (end of sequence)
        if attention_mask is not None:
            # Get index of last True value in each sequence
            lengths = attention_mask.sum(dim=1) - 1  # [B]
            batch_indices = torch.arange(x.size(0), device=x.device)
            pooled = x[batch_indices, lengths]  # [B, D]
        else:
            pooled = x[:, -1, :]  # [B, D]
        
        # Project to output
        logits = self.output_proj(pooled)  # [B, num_classes]
        
        return logits
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Test the model
    print("Testing Standard Transformer...")
    
    vocab_size = 1003  # Example vocab size
    batch_size = 4
    seq_len = 100
    
    model = StandardTransformer(
        vocab_size=vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        num_classes=1000
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    logits = model(input_ids, attention_mask)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print("\n✓ Model test passed!")
