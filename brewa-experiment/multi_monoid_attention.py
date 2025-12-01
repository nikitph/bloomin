"""
Multi-Monoid Attention Heads
=============================

Implements specialized attention heads for different monoid types:
- BooleanREWAHead: Exact pattern matching (Hamming similarity)
- TropicalREWAHead: Shortest-path reasoning (min-plus semiring)
- RealREWAHead: Continuous similarity (dot product)
- ProductMonoidHead: Compositional fusion

Each head specializes in different reasoning types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from brewa_encoder import REWAEncoder, compute_witness_similarity
from brewa_utils import hamming_similarity_efficient, tropical_matmul


class BooleanREWAHead(nn.Module):
    """
    Boolean monoid attention head for exact pattern matching.
    
    Uses Hamming similarity on binary encodings:
        sim(q, k) = |{i : q[i] = k[i]}| / m
    
    Specializes in:
    - Exact token matching
    - Syntax patterns
    - Discrete structures
    """
    
    def __init__(self, d_model: int, d_head: int, m_bits: int = 32, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            d_head: Head dimension
            m_bits: Number of bits in encoding
            dropout: Attention dropout
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_head = d_head
        self.m_bits = m_bits
        
        # Encoders for Q, K, V
        self.q_encoder = REWAEncoder(d_model, m_bits, monoid='boolean')
        self.k_encoder = REWAEncoder(d_model, m_bits, monoid='boolean')
        self.v_proj = nn.Linear(d_model, d_head, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(m_bits)
        
    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q: [B, N, d_model] queries
            K: [B, M, d_model] keys
            V: [B, M, d_model] values
            mask: [B, N, M] attention mask
        
        Returns:
            output: [B, N, d_head]
            attn_weights: [B, N, M]
        """
        B, N, _ = Q.shape
        M = K.shape[1]
        
        # Encode Q, K to binary
        Q_bits = self.q_encoder(Q, return_continuous=False)  # [B, N, m_bits]
        K_bits = self.k_encoder(K, return_continuous=False)  # [B, M, m_bits]
        
        # Project V
        V_proj = self.v_proj(V)  # [B, M, d_head]
        
        # Compute Hamming similarity
        attn_scores = hamming_similarity_efficient(Q_bits, K_bits)  # [B, N, M]
        attn_scores = attn_scores * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N, M]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.bmm(attn_weights, V_proj)  # [B, N, d_head]
        
        return output, attn_weights


class TropicalREWAHead(nn.Module):
    """
    Tropical semiring attention head for shortest-path reasoning.
    
    Uses min-plus operations:
        (A ⊙ B)_ij = min_k (A_ik + B_kj)
    
    Specializes in:
    - Reasoning chains
    - Dependency paths
    - Graph distances
    - Compositional reasoning
    """
    
    def __init__(self, d_model: int, d_head: int, m_bits: int = 32, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_head = d_head
        self.m_bits = m_bits
        
        # Encoders (tropical uses continuous values)
        self.q_encoder = REWAEncoder(d_model, m_bits, monoid='tropical')
        self.k_encoder = REWAEncoder(d_model, m_bits, monoid='tropical')
        self.v_proj = nn.Linear(d_model, d_head, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q: [B, N, d_model]
            K: [B, M, d_model]
            V: [B, M, d_model]
            mask: [B, N, M]
        
        Returns:
            output: [B, N, d_head]
            attn_weights: [B, N, M]
        """
        B, N, _ = Q.shape
        M = K.shape[1]
        
        # Encode Q, K (continuous for tropical)
        Q_tropical = self.q_encoder(Q, return_continuous=True)  # [B, N, m_bits]
        K_tropical = self.k_encoder(K, return_continuous=True)  # [B, M, m_bits]
        
        # Project V
        V_proj = self.v_proj(V)  # [B, M, d_head]
        
        # Tropical matrix multiplication (min-plus)
        # We want similarity, so use negative distance
        # tropical_matmul expects A: [B, N, D], B: [B, D, M]
        attn_scores = -tropical_matmul(
            Q_tropical, 
            K_tropical.transpose(1, 2)  # [B, M, m_bits] -> [B, m_bits, M]
        )  # [B, N, M]
        
        # Scale by temperature
        attn_scores = attn_scores / F.softplus(self.temperature)
        
        # Apply mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax (or could use tropical softmax)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        output = torch.bmm(attn_weights, V_proj)
        
        return output, attn_weights


class RealREWAHead(nn.Module):
    """
    Real-valued REWA head for continuous similarity.
    
    Uses quantized dot products (similar to standard attention).
    
    Specializes in:
    - Semantic similarity
    - Continuous embeddings
    - Dense representations
    """
    
    def __init__(self, d_model: int, d_head: int, m_bits: int = 32, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_head = d_head
        self.m_bits = m_bits
        
        # Encoders
        self.q_encoder = REWAEncoder(d_model, m_bits, monoid='real')
        self.k_encoder = REWAEncoder(d_model, m_bits, monoid='real')
        self.v_proj = nn.Linear(d_model, d_head, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(m_bits)
        
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q: [B, N, d_model]
            K: [B, M, d_model]
            V: [B, M, d_model]
            mask: [B, N, M]
        
        Returns:
            output: [B, N, d_head]
            attn_weights: [B, N, M]
        """
        # Encode Q, K (can use continuous or quantized)
        Q_real = self.q_encoder(Q, return_continuous=True)  # [B, N, m_bits]
        K_real = self.k_encoder(K, return_continuous=True)  # [B, M, m_bits]
        
        # Project V
        V_proj = self.v_proj(V)  # [B, M, d_head]
        
        # Dot product attention (standard)
        attn_scores = torch.bmm(Q_real, K_real.transpose(1, 2))  # [B, N, M]
        attn_scores = attn_scores * self.scale
        
        # Apply mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        output = torch.bmm(attn_weights, V_proj)
        
        return output, attn_weights


class ProductMonoidHead(nn.Module):
    """
    Product monoid head for compositional fusion.
    
    Combines multiple monoid similarities using learned weights.
    
    Specializes in:
    - Multi-modal fusion
    - Compositional reasoning
    - Combining different similarity types
    """
    
    def __init__(
        self, 
        d_model: int, 
        d_head: int, 
        m_bits: int = 32,
        monoid_types: list[str] = ['boolean', 'tropical', 'real'],
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_head = d_head
        self.m_bits = m_bits
        self.monoid_types = monoid_types
        
        # Create encoders for each monoid type
        self.q_encoders = nn.ModuleDict({
            monoid: REWAEncoder(d_model, m_bits, monoid=monoid)
            for monoid in monoid_types
        })
        
        self.k_encoders = nn.ModuleDict({
            monoid: REWAEncoder(d_model, m_bits, monoid=monoid)
            for monoid in monoid_types
        })
        
        self.v_proj = nn.Linear(d_model, d_head, bias=False)
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(len(monoid_types)))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(m_bits)
        
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            Q: [B, N, d_model]
            K: [B, M, d_model]
            V: [B, M, d_model]
            mask: [B, N, M]
        
        Returns:
            output: [B, N, d_head]
            attn_weights: [B, N, M]
        """
        B, N, _ = Q.shape
        M = K.shape[1]
        
        # Compute similarity for each monoid type
        similarities = []
        
        for i, monoid in enumerate(self.monoid_types):
            # Encode
            Q_enc = self.q_encoders[monoid](
                Q, 
                return_continuous=(monoid != 'boolean')
            )
            K_enc = self.k_encoders[monoid](
                K,
                return_continuous=(monoid != 'boolean')
            )
            
            # Compute similarity
            sim = compute_witness_similarity(
                Q_enc, K_enc, 
                monoid=monoid,
                temperature=1.0
            )
            
            similarities.append(sim)
        
        # Stack similarities: [num_monoids, B, N, M]
        similarities = torch.stack(similarities, dim=0)
        
        # Normalize fusion weights
        weights = F.softmax(self.fusion_weights, dim=0)  # [num_monoids]
        weights = weights.view(-1, 1, 1, 1)  # [num_monoids, 1, 1, 1]
        
        # Weighted combination
        attn_scores = (similarities * weights).sum(dim=0)  # [B, N, M]
        attn_scores = attn_scores * self.scale
        
        # Apply mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Project V and apply attention
        V_proj = self.v_proj(V)
        output = torch.bmm(attn_weights, V_proj)
        
        return output, attn_weights
    
    def get_fusion_weights(self) -> dict[str, float]:
        """Return normalized fusion weights for each monoid."""
        weights = F.softmax(self.fusion_weights, dim=0)
        return {
            monoid: weights[i].item()
            for i, monoid in enumerate(self.monoid_types)
        }


class MultiMonoidAttention(nn.Module):
    """
    Multi-head attention with different monoid types per head.
    
    Architecture:
        - Head 0: Boolean (exact matching)
        - Head 1: Tropical (reasoning chains)
        - Head 2: Real (semantic similarity)
        - Head 3: Product (fusion)
    
    Each head specializes in different reasoning types.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 4,
        d_head: Optional[int] = None,
        m_bits: int = 32,
        dropout: float = 0.1,
        head_types: Optional[list[str]] = None,
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_head: Dimension per head (default: d_model // num_heads)
            m_bits: Bits per encoding
            dropout: Dropout rate
            head_types: List of head types (default: balanced mix)
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head or (d_model // num_heads)
        self.m_bits = m_bits
        
        # Default head types: balanced mix
        if head_types is None:
            # Distribute heads across types
            base_types = ['boolean', 'tropical', 'real', 'product']
            head_types = [base_types[i % len(base_types)] for i in range(num_heads)]
        
        assert len(head_types) == num_heads, \
            f"head_types length ({len(head_types)}) must match num_heads ({num_heads})"
        
        self.head_types = head_types
        
        # Create heads
        self.heads = nn.ModuleList()
        for head_type in head_types:
            if head_type == 'boolean':
                head = BooleanREWAHead(d_model, self.d_head, m_bits, dropout)
            elif head_type == 'tropical':
                head = TropicalREWAHead(d_model, self.d_head, m_bits, dropout)
            elif head_type == 'real':
                head = RealREWAHead(d_model, self.d_head, m_bits, dropout)
            elif head_type == 'product':
                head = ProductMonoidHead(
                    d_model, self.d_head, m_bits,
                    monoid_types=['boolean', 'tropical', 'real'],
                    dropout=dropout
                )
            else:
                raise ValueError(f"Unknown head type: {head_type}")
            
            self.heads.append(head)
        
        # Output projection
        self.out_proj = nn.Linear(num_heads * self.d_head, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Args:
            Q: [B, N, d_model]
            K: [B, M, d_model]
            V: [B, M, d_model]
            mask: [B, N, M] or [B, 1, N, M]
            return_attention_weights: If True, return attention weights
        
        Returns:
            output: [B, N, d_model]
            (optional) attention_weights: list of [B, N, M] per head
        """
        B, N, _ = Q.shape
        
        # Squeeze mask if needed
        if mask is not None and mask.dim() == 4:
            mask = mask.squeeze(1)  # [B, N, M]
        
        # Apply each head
        head_outputs = []
        attention_weights = []
        
        for head in self.heads:
            output, attn_weights = head(Q, K, V, mask)
            head_outputs.append(output)
            attention_weights.append(attn_weights)
        
        # Concatenate head outputs
        multi_head_output = torch.cat(head_outputs, dim=-1)  # [B, N, num_heads * d_head]
        
        # Output projection
        output = self.out_proj(multi_head_output)  # [B, N, d_model]
        output = self.dropout(output)
        
        if return_attention_weights:
            return output, attention_weights
        else:
            return output
    
    def get_head_statistics(self) -> dict:
        """Return statistics about head types."""
        from collections import Counter
        counts = Counter(self.head_types)
        
        return {
            'total_heads': self.num_heads,
            'head_distribution': dict(counts),
            'bits_per_head': self.m_bits,
            'total_bits': self.num_heads * self.m_bits,
        }


if __name__ == "__main__":
    print("="*60)
    print("Testing Multi-Monoid Attention Heads")
    print("="*60)
    
    # Test individual heads
    B, N, M, d_model = 2, 10, 8, 128
    Q = torch.randn(B, N, d_model)
    K = torch.randn(B, M, d_model)
    V = torch.randn(B, M, d_model)
    
    print("\n1. Boolean Head")
    bool_head = BooleanREWAHead(d_model, d_head=64, m_bits=32)
    out, attn = bool_head(Q, K, V)
    print(f"Output: {out.shape}, Attention: {attn.shape}")
    
    print("\n2. Tropical Head")
    trop_head = TropicalREWAHead(d_model, d_head=64, m_bits=32)
    out, attn = trop_head(Q, K, V)
    print(f"Output: {out.shape}, Attention: {attn.shape}")
    
    print("\n3. Real Head")
    real_head = RealREWAHead(d_model, d_head=64, m_bits=32)
    out, attn = real_head(Q, K, V)
    print(f"Output: {out.shape}, Attention: {attn.shape}")
    
    print("\n4. Product Monoid Head")
    prod_head = ProductMonoidHead(d_model, d_head=64, m_bits=32)
    out, attn = prod_head(Q, K, V)
    print(f"Output: {out.shape}, Attention: {attn.shape}")
    print(f"Fusion weights: {prod_head.get_fusion_weights()}")
    
    print("\n5. Multi-Monoid Attention")
    mma = MultiMonoidAttention(
        d_model=128,
        num_heads=4,
        m_bits=32,
        head_types=['boolean', 'tropical', 'real', 'product']
    )
    
    out = mma(Q, K, V)
    print(f"Output: {out.shape}")
    print(f"Head statistics: {mma.get_head_statistics()}")
    
    print("\n" + "="*60)
    print("Multi-Monoid Attention Tests Complete!")
    print("="*60)
