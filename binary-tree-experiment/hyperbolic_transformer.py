"""
Hierarchical Branch Transformer for Binary Tree Path Prediction

Simplified version without hyperbolic geometry - focuses on stable hierarchical attention.

Key innovation: Multi-scale branching with efficient bucketed attention.
Should solve depth-20 trees with d=512, p=256.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention that processes information at different resolutions.
    
    This is the core innovation: instead of full O(n²) attention,
    we bucket tokens and attend within buckets at multiple scales.
    """
    
    def __init__(self, d_model, n_heads, bucket_size=64, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.bucket_size = bucket_size
        
        # Standard Q, K, V projections
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
            mask: [B, N] boolean mask
        Returns:
            [B, N, D]
        """
        B, N, D = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        
        # Bucketed attention: divide sequence into chunks
        n_buckets = max(1, N // self.bucket_size)
        bucket_size = N // n_buckets if n_buckets > 0 else N
        
        outputs = []
        for i in range(0, N, bucket_size):
            end = min(i + bucket_size, N)
            
            # Attend within this bucket
            q_chunk = q[:, :, i:end, :]  # [B, H, chunk_len, d_k]
            k_chunk = k[:, :, i:end, :]
            v_chunk = v[:, :, i:end, :]
            
            # Standard attention within bucket
            scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / self.scale
            
            if mask is not None:
                mask_chunk = mask[:, i:end].unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(~mask_chunk, float('-inf'))
            
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            
            out_chunk = torch.matmul(attn, v_chunk)
            outputs.append(out_chunk)
        
        # Concatenate all chunks
        out = torch.cat(outputs, dim=2)  # [B, H, N, d_k]
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.out_proj(out)
        
        return out


class HierarchicalBranchAttention(nn.Module):
    """
    Multi-scale hierarchical attention with different bucket sizes.
    
    Processes information at coarse, medium, and fine resolutions,
    then combines them.
    """
    
    def __init__(self, d_model, n_heads, bucket_sizes=(256, 64, 16), dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.bucket_sizes = bucket_sizes
        
        # Separate attention for each scale
        self.attentions = nn.ModuleList([
            MultiScaleAttention(d_model, n_heads, bucket_size, dropout)
            for bucket_size in bucket_sizes
        ])
        
        # Combine multi-scale outputs
        self.combine = nn.Linear(d_model * len(bucket_sizes), d_model)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [B, N, D]
            mask: [B, N]
        Returns:
            [B, N, D]
        """
        # Process at each scale
        multi_scale_outputs = []
        for attn in self.attentions:
            out = attn(x, mask)
            multi_scale_outputs.append(out)
        
        # Combine scales
        combined = torch.cat(multi_scale_outputs, dim=-1)
        output = self.combine(combined)
        output = self.norm(output)
        
        return output


class HierarchicalTransformerBlock(nn.Module):
    """Transformer block with hierarchical branch attention."""
    
    def __init__(self, d_model, n_heads, d_ff, bucket_sizes=(256, 64, 16), dropout=0.1):
        super().__init__()
        
        self.attn = HierarchicalBranchAttention(d_model, n_heads, bucket_sizes, dropout)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Pre-norm with residual
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class HierarchicalBranchTransformer(nn.Module):
    """
    Hierarchical Branch Transformer for binary tree path prediction.
    
    Uses multi-scale attention instead of full O(n²) attention.
    Should solve depth-20 trees with d=512.
    """
    
    def __init__(
        self,
        vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        bucket_sizes=(256, 64, 16),
        num_classes=1000,
        dropout=0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Standard embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 10000, d_model) * 0.02)
        
        # Transformer blocks with hierarchical attention
        self.blocks = nn.ModuleList([
            HierarchicalTransformerBlock(d_model, n_heads, d_ff, bucket_sizes, dropout)
            for _ in range(n_layers)
        ])
        
        # Output
        self.norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values for stability."""
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
            input_ids: [B, N]
            attention_mask: [B, N]
        Returns:
            [B, num_classes]
        """
        B, N = input_ids.shape
        
        # Embed tokens
        x = self.token_embed(input_ids)  # [B, N, D]
        x = x + self.pos_embed[:, :N, :]
        x = self.dropout(x)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        x = self.norm(x)
        
        # Pool: take last valid token
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(B, device=x.device)
            pooled = x[batch_indices, lengths]
        else:
            pooled = x[:, -1, :]
        
        # Project to output
        logits = self.output_proj(pooled)
        
        return logits
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    print("Testing Hierarchical Branch Transformer...")
    
    vocab_size = 1003
    batch_size = 4
    seq_len = 100
    
    model = HierarchicalBranchTransformer(
        vocab_size=vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        bucket_sizes=(256, 64, 16),
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
