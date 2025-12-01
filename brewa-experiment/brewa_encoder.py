"""
BREWA Encoder
=============

Implements the core REWA encoder with multiple monoid types:
- Boolean: {0,1}^m with Hamming distance
- Tropical: Min-plus semiring for shortest paths
- Real: Quantized continuous values

Key innovation: Hadamard transform + diagonal noise (deterministic + sparse)
vs. Random Projections (dense random matrix)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Literal, Optional

from brewa_utils import (
    hadamard_transform_batched,
    bit_quantize,
    add_diagonal_noise,
    channel_capacity,
)


class REWAEncoder(nn.Module):
    """
    Encode d_model dimensional vectors into m-bit REWA representations.
    
    Architecture:
        1. Linear projection: R^d → R^m
        2. Hadamard transform: O(m log m) deterministic mixing
        3. Diagonal noise injection: σ² controlled
        4. Bit quantization: R^m → {0,1}^m
    
    This achieves capacity C = log₂(m) bits.
    """
    
    def __init__(
        self,
        d_model: int,
        m_bits: int = 32,
        monoid: Literal['boolean', 'tropical', 'real'] = 'real',
        noise_std: float = 0.1,
        learnable_projection: bool = True,
    ):
        """
        Args:
            d_model: Input dimension
            m_bits: Number of bits in encoding (must be power of 2)
            monoid: Type of monoid ('boolean', 'tropical', 'real')
            noise_std: Standard deviation of diagonal noise
            learnable_projection: If True, use learned projection; else identity
        """
        super().__init__()
        
        self.d_model = d_model
        self.m_bits = m_bits
        self.monoid = monoid
        self.noise_std = noise_std
        
        # Ensure m_bits is power of 2 for Hadamard
        assert m_bits > 0 and (m_bits & (m_bits - 1)) == 0, \
            f"m_bits must be power of 2, got {m_bits}"
        
        # Learnable projection to m_bits dimension
        if learnable_projection:
            self.projection = nn.Linear(d_model, m_bits, bias=False)
        else:
            # Use identity or padding
            if d_model == m_bits:
                self.projection = nn.Identity()
            elif d_model < m_bits:
                # Pad with zeros
                self.projection = lambda x: F.pad(x, (0, m_bits - d_model))
            else:
                # Truncate
                self.projection = lambda x: x[..., :m_bits]
        
        # Calculate capacity
        self.capacity = channel_capacity(m_bits, noise_std)
        
    def forward(
        self, 
        x: torch.Tensor, 
        return_continuous: bool = False,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """
        Encode input to REWA representation.
        
        Args:
            x: [B, N, d_model] input embeddings
            return_continuous: If True, return pre-quantization values
            add_noise: If True, add diagonal noise
        
        Returns:
            [B, N, m_bits] encoded representation
            - If return_continuous=False: binary {0,1}
            - If return_continuous=True: continuous R^m
        """
        # 1. Project to m_bits dimension
        if isinstance(self.projection, nn.Linear):
            h = self.projection(x)  # [B, N, m_bits]
        else:
            h = self.projection(x)
        
        # 2. Hadamard transform (deterministic mixing)
        h = hadamard_transform_batched(h, normalize=True)  # [B, N, m_bits]
        
        # 3. Add diagonal noise
        if add_noise and self.training:
            h = add_diagonal_noise(h, self.noise_std)
        
        # 4. Monoid-specific processing
        if self.monoid == 'boolean':
            if return_continuous:
                return h
            else:
                # Quantize to {0, 1}
                return bit_quantize(h, self.m_bits, method='sign')
        
        elif self.monoid == 'tropical':
            # Tropical: use raw values (will be used in min-plus operations)
            # No quantization needed for tropical
            return h
        
        elif self.monoid == 'real':
            if return_continuous:
                return h
            else:
                # Quantize to {0, 1} but can be used as continuous
                return bit_quantize(h, self.m_bits, method='sign')
        
        else:
            raise ValueError(f"Unknown monoid type: {self.monoid}")
    
    def encode_and_decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode then decode (for testing reconstruction).
        
        Args:
            x: [B, N, d_model]
        
        Returns:
            [B, N, d_model] reconstructed
        """
        # Encode
        encoded = self.forward(x, return_continuous=False, add_noise=False)
        
        # Decode: inverse Hadamard + inverse projection
        # For testing purposes only
        decoded = hadamard_transform_batched(encoded, normalize=True)
        
        # If projection is linear, we can't perfectly invert
        # This is just for visualization
        return decoded
    
    def get_capacity(self) -> float:
        """Return channel capacity in bits."""
        return self.capacity
    
    def get_compression_ratio(self) -> float:
        """
        Calculate compression ratio vs. full floating point.
        
        Standard attention: d_model * 32 bits (float32)
        BREWA: m_bits * 1 bit
        
        Returns:
            Compression ratio (e.g., 32.0 for 32× compression)
        """
        standard_bits = self.d_model * 32  # float32
        brewa_bits = self.m_bits * 1  # binary
        return standard_bits / brewa_bits


class MultiMonoidREWAEncoder(nn.Module):
    """
    Encode to multiple monoid types simultaneously.
    
    This allows different attention heads to use different monoids:
    - Head 1: Boolean (exact matching)
    - Head 2: Tropical (shortest paths)
    - Head 3: Real (continuous similarity)
    """
    
    def __init__(
        self,
        d_model: int,
        m_bits: int = 32,
        monoid_types: list[str] = ['boolean', 'tropical', 'real'],
        noise_std: float = 0.1,
    ):
        """
        Args:
            d_model: Input dimension
            m_bits: Bits per monoid encoding
            monoid_types: List of monoid types to use
            noise_std: Noise standard deviation
        """
        super().__init__()
        
        self.d_model = d_model
        self.m_bits = m_bits
        self.monoid_types = monoid_types
        
        # Create encoder for each monoid type
        self.encoders = nn.ModuleDict({
            monoid: REWAEncoder(
                d_model=d_model,
                m_bits=m_bits,
                monoid=monoid,
                noise_std=noise_std,
                learnable_projection=True,
            )
            for monoid in monoid_types
        })
    
    def forward(
        self, 
        x: torch.Tensor,
        return_continuous: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Encode to all monoid types.
        
        Args:
            x: [B, N, d_model]
            return_continuous: Return continuous or quantized
        
        Returns:
            Dictionary mapping monoid type to encoding:
            {
                'boolean': [B, N, m_bits],
                'tropical': [B, N, m_bits],
                'real': [B, N, m_bits],
            }
        """
        encodings = {}
        
        for monoid_type, encoder in self.encoders.items():
            encodings[monoid_type] = encoder(
                x, 
                return_continuous=return_continuous,
                add_noise=self.training,
            )
        
        return encodings
    
    def get_total_compression_ratio(self) -> float:
        """
        Total compression considering all monoid encodings.
        
        If we have 3 monoids, we use 3× the bits,
        but still much less than full attention.
        """
        num_monoids = len(self.monoid_types)
        single_compression = self.encoders[self.monoid_types[0]].get_compression_ratio()
        
        # Total bits = num_monoids * m_bits
        # Compression = (d_model * 32) / (num_monoids * m_bits)
        return single_compression / num_monoids


# ============================================================
# Witness Similarity Functions
# ============================================================

def compute_witness_similarity(
    w1: torch.Tensor,
    w2: torch.Tensor,
    monoid: str = 'boolean',
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute similarity between witness encodings.
    
    Args:
        w1: [B, N, m] witness encodings
        w2: [B, M, m] witness encodings
        monoid: Type of monoid
        temperature: Temperature scaling
    
    Returns:
        [B, N, M] similarity scores
    """
    if monoid == 'boolean':
        # Hamming similarity
        from brewa_utils import hamming_similarity_efficient
        sim = hamming_similarity_efficient(w1, w2)  # [B, N, M]
        
    elif monoid == 'tropical':
        # Tropical distance (min-plus)
        from brewa_utils import tropical_matmul
        # Negative distance → similarity
        sim = -tropical_matmul(w1, w2.transpose(1, 2))
        
    elif monoid == 'real':
        # Dot product (standard)
        sim = torch.bmm(w1, w2.transpose(1, 2))  # [B, N, M]
        
    else:
        raise ValueError(f"Unknown monoid: {monoid}")
    
    # Scale by temperature
    sim = sim / temperature
    
    return sim


if __name__ == "__main__":
    print("="*60)
    print("Testing BREWA Encoder")
    print("="*60)
    
    # Test single encoder
    print("\n1. Single Boolean Encoder")
    encoder = REWAEncoder(d_model=128, m_bits=32, monoid='boolean')
    x = torch.randn(2, 10, 128)
    
    # Encode
    encoded = encoder(x, return_continuous=False)
    print(f"Input: {x.shape}, Encoded: {encoded.shape}")
    print(f"Encoded values (first 8 bits): {encoded[0, 0, :8]}")
    print(f"Capacity: {encoder.get_capacity():.2f} bits")
    print(f"Compression ratio: {encoder.get_compression_ratio():.1f}×")
    
    # Test multi-monoid encoder
    print("\n2. Multi-Monoid Encoder")
    multi_encoder = MultiMonoidREWAEncoder(
        d_model=128,
        m_bits=32,
        monoid_types=['boolean', 'tropical', 'real'],
    )
    
    encodings = multi_encoder(x, return_continuous=False)
    print(f"Encoded to {len(encodings)} monoid types:")
    for monoid_type, enc in encodings.items():
        print(f"  {monoid_type}: {enc.shape}")
    
    print(f"Total compression: {multi_encoder.get_total_compression_ratio():.1f}×")
    
    # Test similarity computation
    print("\n3. Witness Similarity")
    w1 = encodings['boolean'][:, :5, :]  # First 5 tokens
    w2 = encodings['boolean'][:, 5:, :]  # Last 5 tokens
    
    sim = compute_witness_similarity(w1, w2, monoid='boolean')
    print(f"Similarity matrix: {sim.shape}")
    print(f"Sample similarities:\n{sim[0, :3, :3]}")
    
    print("\n" + "="*60)
    print("BREWA Encoder Tests Complete!")
    print("="*60)
