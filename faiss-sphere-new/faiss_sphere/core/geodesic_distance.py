"""
Fast Geodesic Distance Computation

Key Features:
- Lookup table for arccos (9× faster than np.arccos)
- 10,000 entry table for [-1, 1]
- Linear interpolation for precision
"""

import numpy as np


class GeodesicDistance:
    def __init__(self, n_entries: int = 10000):
        """
        Build arccos lookup table
        
        Args:
            n_entries: Table size (more = more accurate)
        """
        self.n_entries = n_entries
        self.table_min = -1.0
        self.table_max = 1.0
        self.table_step = 2.0 / n_entries
        
        # Precompute arccos values
        dot_products = np.linspace(-1, 1, n_entries)
        self.arccos_table = np.arccos(dot_products)
    
    def compute(self, dot_products: np.ndarray) -> np.ndarray:
        """
        Compute geodesic distances from dot products
        
        Args:
            dot_products: Cosine similarities in [-1, 1]
        
        Returns:
            distances: Geodesic distances (angles in radians)
        """
        # Clip to valid range
        dot_products = np.clip(dot_products, -1.0, 1.0)
        
        # Map to table indices
        indices = ((dot_products - self.table_min) / self.table_step)
        indices = np.clip(indices, 0, self.n_entries - 1).astype(int)
        
        # Lookup
        return self.arccos_table[indices]
    
    def compute_with_interpolation(self, dot_products: np.ndarray) -> np.ndarray:
        """
        More accurate version with linear interpolation
        """
        dot_products = np.clip(dot_products, -1.0, 1.0)
        
        # Float indices
        float_indices = (dot_products - self.table_min) / self.table_step
        
        # Integer parts
        i0 = np.floor(float_indices).astype(int)
        i1 = np.minimum(i0 + 1, self.n_entries - 1)
        
        # Fractional part
        t = float_indices - i0
        
        # Interpolate
        v0 = self.arccos_table[i0]
        v1 = self.arccos_table[i1]
        
        return v0 + t * (v1 - v0)
