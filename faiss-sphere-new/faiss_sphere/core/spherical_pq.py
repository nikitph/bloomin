"""
Spherical Product Quantization

Key Features:
- Splits D dimensions into M subspaces
- Spherical k-means in each subspace (normalize centroids)
- Cosine similarity instead of Euclidean distance
- Centroids lie on sphere
"""

import numpy as np
from typing import Tuple


class SphericalPQ:
    def __init__(self, d: int, M: int = 8, nbits: int = 8):
        """
        Args:
            d: Input dimension
            M: Number of subspaces
            nbits: Bits per subspace (2^nbits centroids)
        """
        self.d = d
        self.M = M
        self.nbits = nbits
        self.n_clusters = 2 ** nbits
        self.d_sub = d // M
        
        # Codebooks: M subspaces, each with n_clusters centroids
        self.codebooks = None  # Will be (M, n_clusters, d_sub)
        
    def train(self, X: np.ndarray, max_iter: int = 100):
        """
        Train spherical codebooks using spherical k-means
        
        Args:
            X: Training data (N, d), will be normalized
            max_iter: Max iterations for k-means
        """
        # Normalize
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        
        N = X.shape[0]
        self.codebooks = np.zeros((self.M, self.n_clusters, self.d_sub), dtype='float32')
        
        print(f"Training PQ with {self.M} subspaces, {self.n_clusters} clusters each...")
        
        for m in range(self.M):
            # Extract subspace
            start_idx = m * self.d_sub
            end_idx = start_idx + self.d_sub
            X_sub = X[:, start_idx:end_idx]
            
            # Train spherical k-means for this subspace
            centroids = self._train_subspace(X_sub, self.n_clusters, max_iter)
            self.codebooks[m] = centroids
            
            if (m + 1) % 2 == 0:
                print(f"  Trained subspace {m+1}/{self.M}")
        
        print("PQ training complete")
    
    def _train_subspace(self, X_sub: np.ndarray, n_clusters: int, max_iter: int = 100) -> np.ndarray:
        """
        Spherical k-means for one subspace
        
        Returns:
            centroids: (n_clusters, d_sub) normalized to unit sphere
        """
        N, d_sub = X_sub.shape
        
        # Adjust n_clusters if we have fewer samples
        actual_clusters = min(n_clusters, N)
        
        # Initialize centroids randomly
        indices = np.random.choice(N, actual_clusters, replace=False)
        centroids = X_sub[indices].copy()
        
        # If we have fewer clusters than requested, pad with random vectors
        if actual_clusters < n_clusters:
            extra = n_clusters - actual_clusters
            extra_centroids = np.random.randn(extra, d_sub).astype('float32')
            extra_centroids = extra_centroids / np.linalg.norm(extra_centroids, axis=1, keepdims=True)
            centroids = np.vstack([centroids, extra_centroids])
        
        # Normalize centroids
        centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        
        for iteration in range(max_iter):
            # Assign to nearest centroid (using cosine similarity)
            similarities = X_sub @ centroids.T  # (N, n_clusters)
            assignments = np.argmax(similarities, axis=1)
            
            # Update centroids
            old_centroids = centroids.copy()
            for k in range(n_clusters):
                mask = assignments == k
                if mask.sum() > 0:
                    # Mean of assigned vectors
                    centroids[k] = X_sub[mask].mean(axis=0)
                    # Normalize to sphere
                    norm = np.linalg.norm(centroids[k])
                    if norm > 0:
                        centroids[k] /= norm
            
            # Check convergence
            diff = np.linalg.norm(centroids - old_centroids)
            if diff < 1e-4:
                break
        
        return centroids
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode vector to PQ codes
        
        Args:
            x: Vector (d,) normalized
        
        Returns:
            codes: (M,) uint8 codes
        """
        codes = np.zeros(self.M, dtype=np.uint8)
        
        for m in range(self.M):
            start_idx = m * self.d_sub
            end_idx = start_idx + self.d_sub
            x_sub = x[start_idx:end_idx]
            
            # Find nearest centroid
            similarities = self.codebooks[m] @ x_sub
            codes[m] = np.argmax(similarities)
        
        return codes
    
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """
        Reconstruct vector from codes
        
        Args:
            codes: (M,) uint8 codes
        
        Returns:
            x_reconstructed: (d,) normalized vector
        """
        x_reconstructed = np.zeros(self.d, dtype='float32')
        
        for m in range(self.M):
            start_idx = m * self.d_sub
            end_idx = start_idx + self.d_sub
            x_reconstructed[start_idx:end_idx] = self.codebooks[m][codes[m]]
        
        # Normalize
        x_reconstructed = x_reconstructed / np.linalg.norm(x_reconstructed)
        
        return x_reconstructed
    
    def compute_distance(self, x: np.ndarray, codes: np.ndarray) -> float:
        """
        Asymmetric distance: query (full) vs database (codes)
        
        Returns average cosine similarity across subspaces
        """
        similarity = 0.0
        
        for m in range(self.M):
            start_idx = m * self.d_sub
            end_idx = start_idx + self.d_sub
            x_sub = x[start_idx:end_idx]
            
            # Get centroid for this code
            centroid = self.codebooks[m][codes[m]]
            
            # Cosine similarity
            similarity += np.dot(x_sub, centroid)
        
        # Average similarity across subspaces
        return similarity / self.M
    
    def batch_encode(self, X: np.ndarray) -> np.ndarray:
        """
        Encode batch of vectors
        
        Args:
            X: (N, d)
        
        Returns:
            codes: (N, M) uint8
        """
        N = X.shape[0]
        codes = np.zeros((N, self.M), dtype=np.uint8)
        
        for i in range(N):
            codes[i] = self.encode(X[i])
        
        return codes
