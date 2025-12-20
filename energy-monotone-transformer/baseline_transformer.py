
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

class StandardAttention(nn.Module):
    """
    Standard Scaled Dot-Product Attention without damping or local constraints.
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = (Q @ K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask
            
        A = F.softmax(scores, dim=-1)
        A = self.dropout(A)
        
        out = A @ V
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        out = self.out_proj(out)
        return out

class StandardTransformerBlock(nn.Module):
    """
    Standard Transformer Block:
    x = x + Attention(Norm(x))
    x = x + MLP(Norm(x))
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = StandardAttention(dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x

class StandardTransformer(nn.Module):
    """
    Standard Transformer for benchmarking.
    """
    def __init__(
        self,
        vocab_size: int,
        dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 512,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(max_seq_len, dim)
        self.emb_norm = nn.LayerNorm(dim) # Keeping this fair with our improved model
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            StandardTransformerBlock(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
        
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(self, input_ids: torch.Tensor, mask: Optional[torch.Tensor] = None, return_energies: bool = False) -> torch.Tensor:
        # Note: Standard transformer doesn't care about energy, but we track it for comparison
        from energy_monotone import energy # Reuse energy function for fair metric
        
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.emb_norm(x)
        x = self.dropout(x)
        
        energies = []
        if return_energies:
            energies.append(energy(x).item())
            
        for block in self.blocks:
            x = block(x, mask)
            if return_energies:
                energies.append(energy(x).item())
                
        x = self.norm(x)
        logits = self.head(x)
        
        if return_energies:
            return logits, energies
        return logits
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
