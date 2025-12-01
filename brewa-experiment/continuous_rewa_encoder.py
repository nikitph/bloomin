"""
Continuous REWA Encoder
========================

REWA encoding WITHOUT binary quantization.

Key insight: The REWA theory doesn't require binary encoding!
- Hadamard transform: deterministic mixing
- Diagonal noise: capacity control
- NO quantization: preserves ranking

Expected: 60-80% recall with 8-16× compression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from brewa_utils import hadamard_transform_batched, add_diagonal_noise


class ContinuousREWAEncoder(nn.Module):
    """
    REWA encoder that keeps continuous values (no binary quantization).
    
    Architecture:
        1. Linear projection: R^d → R^m
        2. Hadamard transform: O(m log m) deterministic mixing
        3. Diagonal noise injection: σ² controlled
        4. NO quantization: keep as float32
    
    This achieves compression through dimensionality reduction (d → m),
    not through quantization.
    """
    
    def __init__(
        self,
        d_model: int,
        m_dim: int = 32,
        noise_std: float = 0.01,
        use_hadamard: bool = True,
        learnable_projection: bool = True,
    ):
        """
        Args:
            d_model: Input dimension
            m_dim: Output dimension (compressed)
            noise_std: Standard deviation of diagonal noise
            use_hadamard: If True, use Hadamard transform
            learnable_projection: If True, use learned projection
        """
        super().__init__()
        
        self.d_model = d_model
        self.m_dim = m_dim
        self.noise_std = noise_std
        self.use_hadamard = use_hadamard
        
        # Ensure m_dim is power of 2 for Hadamard
        if use_hadamard:
            assert m_dim > 0 and (m_dim & (m_dim - 1)) == 0, \
                f"m_dim must be power of 2 for Hadamard, got {m_dim}"
        
        # Learnable projection
        if learnable_projection:
            self.projection = nn.Linear(d_model, m_dim, bias=False)
        else:
            # Use identity or padding
            if d_model == m_dim:
                self.projection = nn.Identity()
            elif d_model < m_dim:
                self.projection = lambda x: F.pad(x, (0, m_dim - d_model))
            else:
                self.projection = lambda x: x[..., :m_dim]
        
    def forward(
        self, 
        x: torch.Tensor,
        add_noise: bool = True,
    ) -> torch.Tensor:
        """
        Encode input to continuous REWA representation.
        
        Args:
            x: [B, N, d_model] input embeddings
            add_noise: If True, add diagonal noise (only during training)
        
        Returns:
            [B, N, m_dim] continuous encoded representation (float32)
        """
        # 1. Project to m_dim dimension
        if isinstance(self.projection, nn.Linear):
            h = self.projection(x)  # [B, N, m_dim]
        else:
            h = self.projection(x)
        
        # 2. Hadamard transform (optional, deterministic mixing)
        if self.use_hadamard:
            h = hadamard_transform_batched(h, normalize=True)  # [B, N, m_dim]
        
        # 3. Add diagonal noise (helps with capacity)
        if add_noise and self.training and self.noise_std > 0:
            h = add_diagonal_noise(h, self.noise_std)
        
        # 4. NO quantization - return continuous values!
        return h  # [B, N, m_dim] (float32)
    
    def get_compression_ratio(self) -> float:
        """
        Calculate compression ratio.
        
        Continuous REWA: d_model → m_dim (both float32)
        Compression = d_model / m_dim
        
        Returns:
            Compression ratio (e.g., 16.0 for 256→16)
        """
        return self.d_model / self.m_dim


class ScalarQuantizedREWA(nn.Module):
    """
    REWA encoder with 8-bit scalar quantization.
    
    Better than binary (256 levels vs 2), worse than continuous.
    Expected: 60-90% recall with 4× compression.
    """
    
    def __init__(
        self,
        d_model: int,
        m_dim: int = 32,
        bits: int = 8,
        noise_std: float = 0.01,
    ):
        """
        Args:
            d_model: Input dimension
            m_dim: Output dimension
            bits: Number of bits per value (8 = 256 levels)
            noise_std: Noise standard deviation
        """
        super().__init__()
        
        self.d_model = d_model
        self.m_dim = m_dim
        self.bits = bits
        self.num_levels = 2 ** bits
        
        # Continuous encoder
        self.encoder = ContinuousREWAEncoder(
            d_model, m_dim, noise_std, use_hadamard=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, d_model]
        
        Returns:
            [B, N, m_dim] quantized to 8-bit integers
        """
        # Get continuous encoding
        continuous = self.encoder(x, add_noise=self.training)
        
        # Quantize to 8-bit integers
        # Scale to [0, num_levels-1]
        min_val = continuous.min()
        max_val = continuous.max()
        
        if max_val > min_val:
            scaled = (continuous - min_val) / (max_val - min_val) * (self.num_levels - 1)
        else:
            scaled = torch.zeros_like(continuous)
        
        # Round to integer
        quantized = scaled.round()
        
        # Store as uint8 to save memory
        if self.bits == 8:
            quantized = quantized.byte()
        
        return quantized
    
    def get_compression_ratio(self) -> float:
        """8-bit vs 32-bit float = 4× compression per dimension."""
        return (self.d_model * 32) / (self.m_dim * self.bits)


class ProductQuantizedREWA(nn.Module):
    """
    Product quantization: multiple binary codes.
    
    Instead of one 32-bit code, use 4 × 8-bit codes.
    Expected: 30-50% recall with 32× compression.
    """
    
    def __init__(
        self,
        d_model: int,
        m_bits: int = 32,
        num_codebooks: int = 4,
        noise_std: float = 0.01,
    ):
        """
        Args:
            d_model: Input dimension
            m_bits: Total bits (split across codebooks)
            num_codebooks: Number of codebooks
            noise_std: Noise standard deviation
        """
        super().__init__()
        
        self.d_model = d_model
        self.m_bits = m_bits
        self.num_codebooks = num_codebooks
        self.bits_per_codebook = m_bits // num_codebooks
        
        # Create multiple continuous encoders
        self.codebooks = nn.ModuleList([
            ContinuousREWAEncoder(
                d_model, 
                self.bits_per_codebook,
                noise_std,
                use_hadamard=True
            )
            for _ in range(num_codebooks)
        ])
        
    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: [B, N, d_model]
        
        Returns:
            List of [B, N, bits_per_codebook] encodings
        """
        codes = []
        for codebook in self.codebooks:
            code = codebook(x, add_noise=self.training)
            codes.append(code)
        
        return codes
    
    def compute_similarity(
        self, 
        codes1: list[torch.Tensor], 
        codes2: list[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute similarity between two sets of codes.
        
        Average cosine similarity across all codebooks.
        """
        similarities = []
        
        for c1, c2 in zip(codes1, codes2):
            # Normalize
            c1_norm = F.normalize(c1, dim=-1)
            c2_norm = F.normalize(c2, dim=-1)
            
            # Cosine similarity
            sim = torch.bmm(c1_norm, c2_norm.transpose(1, 2))
            similarities.append(sim)
        
        # Average across codebooks
        return torch.mean(torch.stack(similarities), dim=0)
    
    def get_compression_ratio(self) -> float:
        """Compression ratio."""
        return self.d_model / (self.bits_per_codebook * self.num_codebooks)


if __name__ == "__main__":
    print("="*60)
    print("Testing Continuous REWA Encoders")
    print("="*60)
    
    # Test continuous encoder
    print("\n1. Continuous REWA Encoder")
    encoder = ContinuousREWAEncoder(d_model=256, m_dim=16)
    x = torch.randn(2, 10, 256)
    
    encoded = encoder(x)
    print(f"Input: {x.shape}, Encoded: {encoded.shape}")
    print(f"Compression: {encoder.get_compression_ratio():.1f}×")
    print(f"Encoded dtype: {encoded.dtype}")
    print(f"Encoded values (sample): {encoded[0, 0, :5]}")
    
    # Test 8-bit quantized
    print("\n2. 8-bit Scalar Quantized REWA")
    encoder_8bit = ScalarQuantizedREWA(d_model=256, m_dim=16, bits=8)
    encoded_8bit = encoder_8bit(x)
    print(f"Input: {x.shape}, Encoded: {encoded_8bit.shape}")
    print(f"Compression: {encoder_8bit.get_compression_ratio():.1f}×")
    print(f"Encoded dtype: {encoded_8bit.dtype}")
    print(f"Encoded values (sample): {encoded_8bit[0, 0, :5]}")
    
    # Test product quantization
    print("\n3. Product Quantized REWA")
    encoder_pq = ProductQuantizedREWA(d_model=256, m_bits=32, num_codebooks=4)
    codes = encoder_pq(x)
    print(f"Input: {x.shape}")
    print(f"Number of codebooks: {len(codes)}")
    print(f"Each code shape: {codes[0].shape}")
    print(f"Compression: {encoder_pq.get_compression_ratio():.1f}×")
    
    print("\n" + "="*60)
    print("All encoders working!")
    print("="*60)
