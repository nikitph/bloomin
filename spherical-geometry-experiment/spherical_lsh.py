"""
Spherical Locality-Sensitive Hashing (LSH)
===========================================

Fast approximate nearest neighbor search on the sphere.
Achieves O(log N) search time instead of O(N).
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple


class SphericalLSH:
    """
    Locality-Sensitive Hashing for unit sphere.
    
    Uses random hyperplanes to hash vectors into buckets.
    Vectors in the same bucket are likely to be similar.
    
    Expected speedup: 10-100× for large datasets.
    """
    
    def __init__(self, dim: int, num_tables: int = 10, num_hashes: int = 8):
        """
        Initialize spherical LSH.
        
        Args:
            dim: Embedding dimension
            num_tables: Number of hash tables (more = better recall, slower)
            num_hashes: Number of hash functions per table (more = more selective)
        """
        self.dim = dim
        self.num_tables = num_tables
        self.num_hashes = num_hashes
        
        # Generate random hyperplanes (normalized)
        self.hyperplanes = []
        for _ in range(num_tables):
            planes = np.random.randn(num_hashes, dim)
            # Normalize to unit sphere
            planes = planes / np.linalg.norm(planes, axis=1, keepdims=True)
            self.hyperplanes.append(planes)
        
        # Hash tables: dict of {hash_code: [indices]}
        self.tables = [defaultdict(list) for _ in range(num_tables)]
        
        # Store all vectors for refinement
        self.vectors = []
        self.indices = []
    
    def _hash(self, vec: np.ndarray, table_idx: int) -> tuple:
        """
        Hash vector to bucket using hyperplane projections.
        
        Args:
            vec: Normalized vector on sphere
            table_idx: Which hash table to use
            
        Returns:
            Hash code (tuple of bools)
        """
        # Project onto hyperplanes
        projections = self.hyperplanes[table_idx] @ vec
        
        # Binary hash: sign of projection
        # Points on same side of hyperplane get same bit
        hash_code = tuple(projections > 0)
        
        return hash_code
    
    def insert(self, idx: int, vec: np.ndarray):
        """
        Insert vector into hash tables.
        
        Args:
            idx: Index/ID of the vector
            vec: Vector to insert (will be normalized)
        """
        # Normalize
        vec = vec / np.linalg.norm(vec)
        
        # Store
        self.vectors.append(vec)
        self.indices.append(idx)
        
        # Insert into all tables
        for table_idx in range(self.num_tables):
            hash_code = self._hash(vec, table_idx)
            self.tables[table_idx][hash_code].append(len(self.vectors) - 1)
    
    def query(self, vec: np.ndarray, k: int = 10) -> List[int]:
        """
        Find k approximate nearest neighbors.
        
        Args:
            vec: Query vector (will be normalized)
            k: Number of neighbors to return
            
        Returns:
            List of indices (original IDs passed to insert)
        """
        # Normalize
        vec = vec / np.linalg.norm(vec)
        
        # Collect candidates from all tables
        candidate_internal_ids = set()
        
        for table_idx in range(self.num_tables):
            hash_code = self._hash(vec, table_idx)
            candidate_internal_ids.update(self.tables[table_idx][hash_code])
        
        if len(candidate_internal_ids) == 0:
            # No candidates found, return empty
            return []
        
        # Convert to list
        candidate_internal_ids = list(candidate_internal_ids)
        
        # Refine: compute exact similarities to candidates
        candidate_vecs = np.array([self.vectors[i] for i in candidate_internal_ids])
        similarities = candidate_vecs @ vec
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(-similarities)
        
        # Return top-k original indices
        top_k = min(k, len(sorted_indices))
        result = [self.indices[candidate_internal_ids[sorted_indices[i]]] for i in range(top_k)]
        
        return result
    
    def query_with_stats(self, vec: np.ndarray, k: int = 10) -> Tuple[List[int], dict]:
        """
        Query with statistics about the search.
        
        Returns:
            (results, stats)
        """
        vec = vec / np.linalg.norm(vec)
        
        candidate_internal_ids = set()
        
        for table_idx in range(self.num_tables):
            hash_code = self._hash(vec, table_idx)
            candidate_internal_ids.update(self.tables[table_idx][hash_code])
        
        num_candidates = len(candidate_internal_ids)
        
        if num_candidates == 0:
            return [], {'num_candidates': 0, 'speedup': 0}
        
        candidate_internal_ids = list(candidate_internal_ids)
        candidate_vecs = np.array([self.vectors[i] for i in candidate_internal_ids])
        similarities = candidate_vecs @ vec
        
        sorted_indices = np.argsort(-similarities)
        top_k = min(k, len(sorted_indices))
        result = [self.indices[candidate_internal_ids[sorted_indices[i]]] for i in range(top_k)]
        
        stats = {
            'num_candidates': num_candidates,
            'total_vectors': len(self.vectors),
            'speedup': len(self.vectors) / max(num_candidates, 1)
        }
        
        return result, stats


def build_lsh_index(embeddings: np.ndarray, num_tables: int = 10, num_hashes: int = 8) -> SphericalLSH:
    """
    Build LSH index from embeddings.
    
    Args:
        embeddings: Shape (N, D) array of embeddings
        num_tables: Number of hash tables
        num_hashes: Number of hash functions per table
        
    Returns:
        Populated SphericalLSH index
    """
    dim = embeddings.shape[1]
    lsh = SphericalLSH(dim=dim, num_tables=num_tables, num_hashes=num_hashes)
    
    # Index all embeddings
    for idx, emb in enumerate(embeddings):
        lsh.insert(idx, emb)
    
    return lsh
