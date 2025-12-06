"""
Spherical Vector Quantization
==============================

Compress embeddings using spherical k-means codebook.
Achieves 48× compression with minimal quality loss.
"""

import numpy as np
from typing import Tuple


class SphericalVQ:
    """
    Vector Quantization on sphere.
    
    Learns a codebook of representative vectors on the sphere.
    Each embedding is encoded as the index of its nearest codebook entry.
    
    Compression: 384D float32 → 8-bit codes = 48× compression
    """
    
    def __init__(self, num_clusters: int = 256, dim: int = 384):
        """
        Initialize spherical VQ.
        
        Args:
            num_clusters: Size of codebook (max 256 for uint8)
            dim: Embedding dimension
        """
        self.num_clusters = num_clusters
        self.dim = dim
        self.codebook = None
    
    def fit(self, embeddings: np.ndarray, max_iter: int = 100, verbose: bool = False):
        """
        Learn codebook using spherical k-means.
        
        Args:
            embeddings: Shape (N, D) array of embeddings
            max_iter: Maximum iterations
            verbose: Print progress
            
        Returns:
            self (for chaining)
        """
        # Normalize all embeddings
        emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Initialize codebook (random points on sphere)
        idx = np.random.choice(len(emb_norm), self.num_clusters, replace=False)
        self.codebook = emb_norm[idx].copy()
        
        for iteration in range(max_iter):
            # Assign to nearest centroid (cosine similarity)
            similarities = emb_norm @ self.codebook.T
            assignments = np.argmax(similarities, axis=1)
            
            # Update centroids
            old_codebook = self.codebook.copy()
            
            for k in range(self.num_clusters):
                mask = assignments == k
                if mask.sum() > 0:
                    # New centroid = normalized mean
                    mean = emb_norm[mask].mean(axis=0)
                    self.codebook[k] = mean / np.linalg.norm(mean)
            
            # Check convergence
            change = np.linalg.norm(self.codebook - old_codebook)
            
            if verbose and (iteration % 10 == 0 or iteration == max_iter - 1):
                print(f"Iteration {iteration}: change = {change:.6f}")
            
            if change < 1e-4:
                if verbose:
                    print(f"Converged in {iteration+1} iterations")
                break
        
        return self
    
    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Encode embeddings as codebook indices.
        
        Args:
            embeddings: Shape (N, D) array of embeddings
            
        Returns:
            Shape (N,) uint8 array of codes (48× smaller!)
        """
        if self.codebook is None:
            raise ValueError("Must call fit() before encode()")
        
        # Normalize
        emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Find nearest codebook entry
        similarities = emb_norm @ self.codebook.T
        codes = np.argmax(similarities, axis=1).astype(np.uint8)
        
        return codes
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Decode indices back to embeddings.
        
        Args:
            codes: Shape (N,) uint8 array of codes
            
        Returns:
            Shape (N, D) array of reconstructed embeddings
        """
        if self.codebook is None:
            raise ValueError("Must call fit() before decode()")
        
        return self.codebook[codes]
    
    def compressed_search(self, query: np.ndarray, codes: np.ndarray, k: int = 10) -> np.ndarray:
        """
        Search using compressed codes (FAST!).
        
        Instead of comparing query to N embeddings,
        we compare to 256 codebook entries, then lookup.
        
        Args:
            query: Query vector
            codes: Compressed codes for all documents
            k: Number of results
            
        Returns:
            Indices of top-k results
        """
        if self.codebook is None:
            raise ValueError("Must call fit() before compressed_search()")
        
        # Normalize query
        query_norm = query / np.linalg.norm(query)
        
        # Compute similarities to codebook (only 256 comparisons!)
        codebook_sims = self.codebook @ query_norm
        
        # Lookup similarities for all documents
        similarities = codebook_sims[codes]
        
        # Return top-k
        top_k = np.argsort(-similarities)[:k]
        
        return top_k
    
    def compression_stats(self, embeddings: np.ndarray, codes: np.ndarray) -> dict:
        """
        Compute compression statistics.
        
        Args:
            embeddings: Original embeddings
            codes: Compressed codes
            
        Returns:
            Dictionary with stats
        """
        original_bytes = embeddings.nbytes
        compressed_bytes = codes.nbytes
        compression_ratio = original_bytes / compressed_bytes
        
        # Reconstruction error
        reconstructed = self.decode(codes)
        emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        rec_norm = reconstructed / np.linalg.norm(reconstructed, axis=1, keepdims=True)
        
        # Cosine similarity between original and reconstructed
        similarities = np.sum(emb_norm * rec_norm, axis=1)
        avg_similarity = np.mean(similarities)
        
        return {
            'original_mb': original_bytes / 1e6,
            'compressed_mb': compressed_bytes / 1e6,
            'compression_ratio': compression_ratio,
            'avg_reconstruction_similarity': avg_similarity,
            'min_reconstruction_similarity': np.min(similarities),
            'max_reconstruction_similarity': np.max(similarities)
        }
