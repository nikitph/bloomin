"""
Main FAISS-Sphere Index

Combines all optimizations:
1. Intrinsic-dimensional projection (768D → 350D)
2. Choice of spherical algorithms (LSH, PQ, HNSW)
3. Geodesic distance computation
"""

import numpy as np
from typing import Tuple, List
import sys

from .core.intrinsic_projector import IntrinsicProjector
from .core.geodesic_distance import GeodesicDistance
from .core.spherical_lsh import SphericalLSH
from .core.spherical_pq import SphericalPQ
from .utils import normalize_vectors


class FAISSSphere:
    def __init__(self, 
                 d_ambient: int,
                 mode: str = 'balanced',
                 d_intrinsic: int = 350):
        """
        Args:
            d_ambient: Input dimension (e.g., 768)
            mode: 'fast' (LSH), 'balanced' (HNSW), 'memory' (PQ), 'exact'
            d_intrinsic: Intrinsic dimension (default 350)
        """
        self.d_ambient = d_ambient
        self.d_intrinsic = min(d_intrinsic, d_ambient)  # Can't project to higher dim
        self.mode = mode
        
        # Intrinsic projector
        self.projector = IntrinsicProjector(d_ambient, self.d_intrinsic)
        
        # Geodesic distance computer
        self.geodesic = GeodesicDistance()
        
        # Choose index based on mode
        if mode == 'fast':
            self.index = SphericalLSH(self.d_intrinsic, nbits=16)
            self.index_type = 'lsh'
        elif mode == 'memory':
            self.index = SphericalPQ(self.d_intrinsic, M=8, nbits=8)
            self.index_type = 'pq'
        elif mode == 'exact':
            # Use FAISS flat index
            try:
                import faiss
                self.index = faiss.IndexFlatIP(self.d_intrinsic)
                self.index_type = 'faiss'
            except ImportError:
                print("Warning: faiss not installed, using brute force")
                self.index = None
                self.index_type = 'brute'
        else:
            raise ValueError(f"Unknown mode: {mode}. Choose from 'fast', 'memory', 'exact'")
        
        self.data = []  # Store original vectors for recall computation
        self.data_intrinsic = []  # Store projected vectors
        self.pq_codes = []  # For PQ mode
        self.trained = False
    
    def train(self, X: np.ndarray):
        """
        Train projector and index
        
        Args:
            X: Training data (N, d_ambient)
        """
        print(f"Training FAISS-Sphere ({self.mode} mode)...")
        
        # Normalize
        X = normalize_vectors(X)
        
        # Train projector
        self.projector.train(X)
        
        # Project
        X_intrinsic = self.projector.project(X)
        
        # Train index if needed
        if self.index_type == 'pq':
            self.index.train(X_intrinsic)
        
        self.trained = True
        print(f"Training complete. Variance explained: {self.projector.variance_explained:.3f}")
    
    def add(self, X: np.ndarray):
        """
        Add vectors to index
        
        Args:
            X: Vectors (N, d_ambient)
        """
        if not self.trained:
            raise ValueError("Must call train() before add()")
        
        # Normalize
        X = normalize_vectors(X)
        
        # Project
        X_intrinsic = self.projector.project(X)
        
        # Add to index
        if self.index_type == 'faiss':
            # FAISS-style
            self.index.add(X_intrinsic.astype('float32'))
        elif self.index_type == 'lsh':
            # Custom LSH
            for i, x in enumerate(X_intrinsic):
                self.index.add(len(self.data) + i, x)
        elif self.index_type == 'pq':
            # PQ: encode and store codes
            codes = self.index.batch_encode(X_intrinsic)
            self.pq_codes.append(codes)
        elif self.index_type == 'brute':
            # Brute force: just store
            pass
        
        # Store original and projected
        self.data.append(X)
        self.data_intrinsic.append(X_intrinsic)
    
    def search(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors
        
        Args:
            query: Query vectors (N_query, d_ambient)
            k: Number of neighbors
        
        Returns:
            distances: (N_query, k) geodesic distances
            indices: (N_query, k) neighbor indices
        """
        # Normalize
        query = normalize_vectors(query)
        
        # Project
        query_intrinsic = self.projector.project(query)
        
        # Get all data
        if len(self.data_intrinsic) == 0:
            raise ValueError("No data in index. Call add() first.")
        
        all_data_intrinsic = np.vstack(self.data_intrinsic)
        
        # Search based on index type
        if self.index_type == 'faiss':
            # FAISS search
            dot_products, indices = self.index.search(query_intrinsic.astype('float32'), k)
            distances = self.geodesic.compute(dot_products)
            
        elif self.index_type == 'lsh':
            # LSH search with refinement
            all_distances = []
            all_indices = []
            
            for q in query_intrinsic:
                candidates = self.index.query(q, k=k, n_probes=2)
                
                if len(candidates) > 0:
                    # Refine with exact distances
                    candidate_vecs = all_data_intrinsic[candidates]
                    sims = candidate_vecs @ q
                    top_k_idx = np.argsort(-sims)[:k]
                    
                    indices_q = np.array([candidates[i] for i in top_k_idx])
                    distances_q = self.geodesic.compute(sims[top_k_idx])
                else:
                    # Fallback to brute force
                    sims = all_data_intrinsic @ q
                    top_k_idx = np.argsort(-sims)[:k]
                    indices_q = top_k_idx
                    distances_q = self.geodesic.compute(sims[top_k_idx])
                
                # Ensure we have exactly k results (pad if needed)
                if len(indices_q) < k:
                    pad_len = k - len(indices_q)
                    indices_q = np.pad(indices_q, (0, pad_len), constant_values=0)
                    distances_q = np.pad(distances_q, (0, pad_len), constant_values=np.inf)
                
                all_distances.append(distances_q)
                all_indices.append(indices_q)
            
            distances = np.array(all_distances)
            indices = np.array(all_indices)
            
        elif self.index_type == 'pq':
            # PQ search
            all_codes = np.vstack(self.pq_codes)
            results = []
            
            for q in query_intrinsic:
                # Compute asymmetric distances
                N = all_codes.shape[0]
                dists = np.zeros(N)
                for i in range(N):
                    dists[i] = self.index.compute_distance(q, all_codes[i])
                
                # Get top k (highest similarity)
                top_k_idx = np.argsort(-dists)[:k]
                indices_q = top_k_idx
                
                # Convert similarities to distances
                distances_q = self.geodesic.compute(dists[top_k_idx])
                
                results.append((distances_q, indices_q))
            
            distances = np.array([r[0] for r in results])
            indices = np.array([r[1] for r in results])
            
        else:  # brute force
            results = []
            for q in query_intrinsic:
                sims = all_data_intrinsic @ q
                top_k_idx = np.argsort(-sims)[:k]
                
                indices_q = top_k_idx
                distances_q = self.geodesic.compute(sims[top_k_idx])
                
                results.append((distances_q, indices_q))
            
            distances = np.array([r[0] for r in results])
            indices = np.array([r[1] for r in results])
        
        return distances, indices
    
    def benchmark_stats(self) -> dict:
        """
        Return memory and computational stats
        """
        memory_mb = 0
        
        # Projector memory
        if self.projector.projection_matrix is not None:
            memory_mb += self.projector.projection_matrix.nbytes / 1e6
        
        # Data memory
        if len(self.data) > 0:
            memory_mb += sum(d.nbytes for d in self.data) / 1e6
        
        if len(self.data_intrinsic) > 0:
            memory_mb += sum(d.nbytes for d in self.data_intrinsic) / 1e6
        
        # PQ codes memory
        if len(self.pq_codes) > 0:
            memory_mb += sum(c.nbytes for c in self.pq_codes) / 1e6
        
        # Index memory (rough estimate)
        memory_mb += sys.getsizeof(self.index) / 1e6
        
        return {
            'mode': self.mode,
            'memory_mb': memory_mb,
            'd_intrinsic': self.d_intrinsic,
            'variance_explained': self.projector.variance_explained if self.trained else None
        }
