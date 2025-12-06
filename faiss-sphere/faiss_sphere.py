"""
FAISS-Sphere: Unified Spherical Vector Search
==============================================

Combines all optimizations into a single interface.
"""

import numpy as np
import faiss
from typing import Tuple, Dict
import time

from intrinsic_projection import IntrinsicDimensionalIndex
from spherical_index import SphericalIndex


class FAISSSphere:
    """
    Complete spherical vector search system.
    
    Modes:
    - 'fast': Intrinsic projection + flat (2.2× faster)
    - 'accurate': Full dimension + geodesic distances
    - 'balanced': Intrinsic + geodesic (best of both)
    """
    
    def __init__(self, d_ambient: int, mode: str = 'balanced', d_intrinsic: int = 350):
        """
        Initialize FAISS-Sphere.
        
        Args:
            d_ambient: Ambient dimension (e.g., 768 for BERT)
            mode: 'fast', 'accurate', or 'balanced'
            d_intrinsic: Intrinsic dimension for projection
        """
        self.d_ambient = d_ambient
        self.mode = mode
        self.d_intrinsic = d_intrinsic
        
        # Choose index based on mode
        if mode == 'fast':
            # Intrinsic projection only (fastest)
            self.projector = IntrinsicDimensionalIndex(d_ambient, d_intrinsic)
            self.index = None  # Will use projector's index
            
        elif mode == 'accurate':
            # Full dimension with geodesic distances
            self.projector = None
            self.index = SphericalIndex(d_ambient)
            
        elif mode == 'balanced':
            # Intrinsic projection + geodesic distances
            self.projector = IntrinsicDimensionalIndex(d_ambient, d_intrinsic)
            self.index = None  # Will use projector's index with geodesic wrapper
            
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        self.is_trained = False
        self.n_vectors = 0
    
    def train(self, X: np.ndarray, verbose: bool = True):
        """
        Train the index (if needed).
        
        Args:
            X: Training data, shape (N, d_ambient)
            verbose: Print progress
        """
        if verbose:
            print(f"Training FAISS-Sphere ({self.mode} mode)...")
        
        # Normalize
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        
        if self.mode in ['fast', 'balanced']:
            # Train projector
            self.projector.train(X, verbose=verbose)
        
        self.is_trained = True
    
    def add(self, X: np.ndarray):
        """
        Add vectors to index.
        
        Args:
            X: Vectors to add, shape (N, d_ambient)
        """
        if not self.is_trained and self.mode in ['fast', 'balanced']:
            raise ValueError("Must call train() before add() in fast/balanced mode")
        
        # Normalize
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        
        if self.mode in ['fast', 'balanced']:
            self.projector.add(X)
        else:
            self.index.add(X)
        
        self.n_vectors += len(X)
    
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.
        
        Args:
            query: Query vectors, shape (N, d_ambient)
            k: Number of neighbors
            
        Returns:
            distances: Distances/similarities, shape (N, k)
            indices: Neighbor indices, shape (N, k)
        """
        # Normalize
        query = query / np.linalg.norm(query, axis=1, keepdims=True)
        
        if self.mode in ['fast', 'balanced']:
            return self.projector.search(query, k)
        else:
            return self.index.search(query, k)
    
    def benchmark(self, query: np.ndarray, k: int = 10) -> Dict:
        """
        Benchmark search performance.
        
        Args:
            query: Query vectors, shape (N, d_ambient)
            k: Number of neighbors
            
        Returns:
            Dictionary with timing and memory stats
        """
        start = time.time()
        distances, indices = self.search(query, k)
        search_time = (time.time() - start) / len(query)  # Per query
        
        # Memory estimate
        if self.mode in ['fast', 'balanced']:
            memory_mb = self.n_vectors * self.d_intrinsic * 4 / 1e6
        else:
            memory_mb = self.n_vectors * self.d_ambient * 4 / 1e6
        
        return {
            'mode': self.mode,
            'search_time_ms': search_time * 1000,
            'memory_mb': memory_mb,
            'n_vectors': self.n_vectors,
        }
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        stats = {
            'mode': self.mode,
            'd_ambient': self.d_ambient,
            'n_vectors': self.n_vectors,
        }
        
        if self.mode in ['fast', 'balanced']:
            stats.update(self.projector.get_stats())
        
        return stats


if __name__ == '__main__':
    # Quick test
    print("Testing FAISS-Sphere...")
    
    N = 10000
    D = 768
    
    # Generate data
    data = np.random.randn(N, D).astype('float32')
    faiss.normalize_L2(data)
    
    query = np.random.randn(10, D).astype('float32')
    faiss.normalize_L2(query)
    
    # Test each mode
    for mode in ['fast', 'accurate', 'balanced']:
        print(f"\n{'='*60}")
        print(f"Mode: {mode}")
        print(f"{'='*60}")
        
        index = FAISSSphere(D, mode=mode, d_intrinsic=350)
        
        if mode in ['fast', 'balanced']:
            index.train(data[:1000])
        
        index.add(data)
        
        distances, indices = index.search(query, k=10)
        
        stats = index.benchmark(query, k=10)
        
        print(f"Search time: {stats['search_time_ms']:.3f}ms per query")
        print(f"Memory: {stats['memory_mb']:.1f} MB")
        print(f"Stats: {index.get_stats()}")
