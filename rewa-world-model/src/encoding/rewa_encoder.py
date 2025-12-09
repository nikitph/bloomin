"""
REWA Encoding Module

Implements capacity-based encoding with Shannon formula:
m ~ O(1/Δ² · log N)

Supports multiple monoid operations:
- Boolean: OR aggregation
- Natural: Sum aggregation
- Real: Weighted sum
- Tropical: Min aggregation
"""

import numpy as np
from typing import List, Dict, Callable
from dataclasses import dataclass
import hashlib

from witnesses import Witness, WitnessType

@dataclass
class REWAConfig:
    """REWA encoding configuration"""
    input_dim: int  # Number of unique witnesses
    num_positions: int  # m (index size)
    num_hashes: int  # K (number of hash functions)
    delta_gap: float  # Minimum separation for capacity
    seed: int = 42

class REWAEncoder:
    """REWA encoder with capacity-based sizing"""
    
    def __init__(self, config: REWAConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        
        # Generate K hash functions
        self.hash_seeds = [config.seed + i for i in range(config.num_hashes)]
    
    @staticmethod
    def estimate_m(delta: float, N: int, K: int = 1) -> int:
        """
        Estimate required index size using Shannon-REWA capacity formula:
        m ~ O(1/Δ² · log N)
        
        Args:
            delta: Minimum gap between items
            N: Number of items to encode
            K: Number of hash functions
        
        Returns:
            Estimated m (number of positions)
        """
        # Shannon capacity: C(Δ) ~ Δ² / K
        # Required: m ≥ log(N) / C(Δ)
        capacity = (delta ** 2) / K
        m_required = int(np.ceil(np.log(N) / capacity))
        
        # Add safety margin
        return int(m_required * 1.5)
    
    def _hash_witness(self, witness_id: str, k: int) -> int:
        """Hash witness to position using k-th hash function"""
        # Use SHA256 for cryptographic hashing
        h = hashlib.sha256()
        h.update(f"{witness_id}_{k}_{self.hash_seeds[k]}".encode())
        hash_value = int.from_bytes(h.digest()[:4], 'big')
        return hash_value % self.config.num_positions
    
    def encode(self, witnesses: List[Witness]) -> np.ndarray:
        """
        Encode witnesses into REWA index
        
        Returns:
            Binary signature of shape (num_positions,)
        """
        m = self.config.num_positions
        K = self.config.num_hashes
        
        # Initialize index based on witness type
        if len(witnesses) == 0:
            return np.zeros(m)
        
        witness_type = witnesses[0].witness_type
        
        if witness_type == WitnessType.BOOLEAN:
            index = self._encode_boolean(witnesses)
        elif witness_type == WitnessType.NATURAL:
            index = self._encode_natural(witnesses)
        elif witness_type == WitnessType.REAL:
            index = self._encode_real(witnesses)
        elif witness_type == WitnessType.TROPICAL:
            index = self._encode_tropical(witnesses)
        else:
            raise ValueError(f"Unknown witness type: {witness_type}")
        
        return index
    
    def _encode_boolean(self, witnesses: List[Witness]) -> np.ndarray:
        """Boolean monoid: OR aggregation"""
        index = np.zeros(self.config.num_positions, dtype=bool)
        
        for witness in witnesses:
            for k in range(self.config.num_hashes):
                pos = self._hash_witness(witness.id, k)
                index[pos] = True  # OR operation
        
        return index.astype(float)
    
    def _encode_natural(self, witnesses: List[Witness]) -> np.ndarray:
        """Natural monoid: Sum aggregation"""
        index = np.zeros(self.config.num_positions)
        
        for witness in witnesses:
            for k in range(self.config.num_hashes):
                pos = self._hash_witness(witness.id, k)
                index[pos] += witness.value  # Sum operation
        
        return index
    
    def _encode_real(self, witnesses: List[Witness]) -> np.ndarray:
        """Real monoid: Weighted sum"""
        index = np.zeros(self.config.num_positions)
        
        for witness in witnesses:
            for k in range(self.config.num_hashes):
                pos = self._hash_witness(witness.id, k)
                index[pos] += witness.value  # Weighted sum
        
        return index
    
    def _encode_tropical(self, witnesses: List[Witness]) -> np.ndarray:
        """Tropical monoid: Min aggregation"""
        index = np.full(self.config.num_positions, np.inf)
        
        for witness in witnesses:
            for k in range(self.config.num_hashes):
                pos = self._hash_witness(witness.id, k)
                index[pos] = min(index[pos], witness.value)  # Min operation
        
        # Replace inf with large value
        index[np.isinf(index)] = 1e6
        
        return index

def hamming_distance(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Compute Hamming distance between binary signatures"""
    return np.sum(sig1 != sig2)

def l1_distance(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Compute L1 distance for Natural/Real monoids"""
    return np.sum(np.abs(sig1 - sig2))

def tropical_distance(sig1: np.ndarray, sig2: np.ndarray) -> float:
    """Compute tropical distance (max of min differences)"""
    return np.max(np.minimum(sig1, sig2))
