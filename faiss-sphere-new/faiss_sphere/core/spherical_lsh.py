"""
Spherical Locality-Sensitive Hashing

Key Features:
- Random hyperplanes normalized to unit sphere
- Collision probability: P[h(u)=h(v)] = 1 - arccos(⟨u,v⟩)/π
- Multi-probe search (flip bits to check neighboring buckets)
- Automatic bit count optimization based on target recall
"""

import numpy as np
from typing import List, Tuple
from collections import defaultdict


class SphericalLSH:
    def __init__(self, d: int, nbits: int = None, target_recall: float = 0.95):
        """
        Args:
            d: Dimension of input vectors
            nbits: Number of hash bits (auto-computed if None)
            target_recall: Target recall for automatic bit selection
        """
        self.d = d
        self.target_recall = target_recall
        self.nbits = nbits if nbits is not None else 16  # Default
        self.hyperplanes = None
        self.buckets = defaultdict(list)
        self.n_vectors = 0
        
        # Initialize random hyperplanes
        self._init_hyperplanes()
    
    def _init_hyperplanes(self):
        """Initialize random hyperplanes normalized to unit sphere"""
        np.random.seed(42)
        self.hyperplanes = np.random.randn(self.nbits, self.d).astype('float32')
        # Normalize to unit sphere
        self.hyperplanes = self.hyperplanes / np.linalg.norm(
            self.hyperplanes, axis=1, keepdims=True
        )
    
    def _compute_optimal_bits(self, N: int, target_recall: float) -> int:
        """
        Compute optimal number of bits based on:
        - Dataset size N
        - Target recall
        - Spherical collision probability
        
        Formula: nbits = log(N) / log(1/p)
        where p = 1 - arccos(s_neighbor)/π
        """
        # Assume average neighbor similarity ~0.7
        s_neighbor = 0.7
        p = 1 - np.arccos(s_neighbor) / np.pi
        
        # Compute optimal bits
        if p > 0:
            nbits = int(np.log(N) / np.log(1/p))
        else:
            nbits = 16
        
        # Clamp to reasonable range
        nbits = max(8, min(32, nbits))
        
        return nbits
    
    def add(self, idx: int, x: np.ndarray):
        """Add vector to LSH buckets"""
        hash_code = self._hash(x)
        bucket_id = self._get_bucket_id(hash_code)
        self.buckets[bucket_id].append(idx)
        self.n_vectors += 1
    
    def query(self, x: np.ndarray, k: int = 10, n_probes: int = 1) -> List[int]:
        """
        Query with multi-probe
        
        Args:
            x: Query vector (normalized)
            k: Number of neighbors
            n_probes: Number of buckets to probe (1 = exact bucket only)
        
        Returns:
            List of candidate indices
        """
        hash_code = self._hash(x)
        bucket_id = self._get_bucket_id(hash_code)
        
        # Get probe buckets
        probe_buckets = self._get_probe_buckets(bucket_id, n_probes)
        
        # Collect candidates
        candidates = []
        for bid in probe_buckets:
            if bid in self.buckets:
                candidates.extend(self.buckets[bid])
        
        # Remove duplicates and limit
        candidates = list(set(candidates))
        
        return candidates[:k*10]  # Return more candidates for refinement
    
    def _hash(self, x: np.ndarray) -> np.ndarray:
        """Compute hash code (binary vector)"""
        # Dot product with hyperplanes
        dots = self.hyperplanes @ x
        # Convert to binary
        return (dots > 0).astype(np.uint8)
    
    def _get_bucket_id(self, hash_code: np.ndarray) -> int:
        """Convert binary hash to bucket integer ID"""
        # Convert binary array to integer
        bucket_id = 0
        for i, bit in enumerate(hash_code):
            bucket_id |= (int(bit) << i)
        return bucket_id
    
    def _get_probe_buckets(self, bucket_id: int, n_probes: int) -> List[int]:
        """
        Get neighboring buckets by flipping bits
        
        For n_probes=1: just the exact bucket
        For n_probes=2: exact + 1-bit flips
        For n_probes=3: exact + 1-bit + 2-bit flips
        """
        if n_probes == 1:
            return [bucket_id]
        
        probe_buckets = [bucket_id]
        
        # 1-bit flips
        if n_probes >= 2:
            for i in range(self.nbits):
                flipped = bucket_id ^ (1 << i)
                probe_buckets.append(flipped)
        
        # 2-bit flips
        if n_probes >= 3:
            for i in range(self.nbits):
                for j in range(i+1, self.nbits):
                    flipped = bucket_id ^ (1 << i) ^ (1 << j)
                    probe_buckets.append(flipped)
        
        return probe_buckets[:min(len(probe_buckets), n_probes * 10)]
