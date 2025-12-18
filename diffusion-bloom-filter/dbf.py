
import numpy as np
from .lsh import LSHEncoder

class DiffusionBloomFilterFloat:
    def __init__(self, size: int, input_dim: int, sigma: float = 5.0, decay_rate: float = 0.99):
        self.size = size
        self.array = np.zeros(size, dtype=np.float32)
        self.sigma = sigma
        self.encoder = LSHEncoder(input_dim, size)
        self.decay_rate = decay_rate

    def _get_kernel(self, center_idx: int):
        # Create a Gaussian kernel wrapping around the cyclic array
        # Optimization: only compute for significant range, e.g., +/- 3*sigma
        radius = int(3 * self.sigma)
        indices = np.arange(center_idx - radius, center_idx + radius + 1)
        
        # Distance handling cyclic boundary
        # The simplest way for calculation is to compute dist in linear space, 
        # then map indices modulo size.
        dists = indices - center_idx
        weights = np.exp(-0.5 * (dists / self.sigma) ** 2)
        
        wrapped_indices = indices % self.size
        return wrapped_indices, weights

    def insert(self, vector: np.ndarray):
        center_idx = self.encoder.hash(vector)
        indices, weights = self._get_kernel(center_idx)
        
        # Add heat
        self.array[indices] += weights
        
        # Clip to max 1.0
        self.array[indices] = np.clip(self.array[indices], 0.0, 1.0)

    def query(self, vector: np.ndarray) -> float:
        idx = self.encoder.hash(vector)
        return self.array[idx]
        
    def decay(self):
        self.array *= self.decay_rate


class DiffusionBloomFilterBitset:
    def __init__(self, size: int, input_dim: int, sigma: float = 5.0):
        self.size = size
        # Using int8 for bitset simulation (0 or 1) for simplicity in Python, 
        # but logically it's a bitset.
        self.array = np.zeros(size, dtype=np.int8)
        self.sigma = sigma
        self.encoder = LSHEncoder(input_dim, size)

    def _get_kernel_probs(self, center_idx: int):
        radius = int(3 * self.sigma)
        indices = np.arange(center_idx - radius, center_idx + radius + 1)
        dists = indices - center_idx
        probs = np.exp(-0.5 * (dists / self.sigma) ** 2)
        wrapped_indices = indices % self.size
        return wrapped_indices, probs

    def insert(self, vector: np.ndarray):
        center_idx = self.encoder.hash(vector)
        indices, probs = self._get_kernel_probs(center_idx)
        
        # Stochastic Dithering: Flip bit to 1 with probability 'probs'
        # We only flip if it is currently 0. If it's already 1, it stays 1.
        # But wait, if we treat it as purely stochastic, we should just roll dice.
        # "flip bits probabilistically based on kernel intensity"
        
        random_rolls = np.random.rand(len(probs))
        should_flip = random_rolls < probs
        
        self.array[indices[should_flip]] = 1

    def query(self, vector: np.ndarray) -> float:
        # Recover heat by averaging bits in local window 
        # Window size roughly corresponds to effective width of the kernel
        idx = self.encoder.hash(vector)
        
        # Let's say we smooth over 1 sigma radius
        radius = int(self.sigma)
        indices = np.arange(idx - radius, idx + radius + 1) % self.size
        
        bits = self.array[indices]
        # Calculate density
        density = np.mean(bits)
        
        # The density roughly correlates to original intensity. 
        # If we had a full gaussian set to 1s with probability exp(-x^2), 
        # the local density reflects that probability mass.
        return density
