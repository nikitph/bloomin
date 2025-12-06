"""
Intrinsic Dimensional Projection

Key Insight: Semantic manifold has intrinsic dimension ~350D
even when embeddings are 768D or 12,288D

Strategy:
1. Learn 768D → 350D projection via spherical PCA
2. Retain 95-99% of variance
3. 2× speedup, 2× memory reduction
"""

import numpy as np
from sklearn.decomposition import PCA


class IntrinsicProjector:
    def __init__(self, d_ambient: int, d_intrinsic: int = 350):
        """
        Args:
            d_ambient: Input dimension (e.g., 768 for BERT)
            d_intrinsic: Target intrinsic dimension (default 350)
        """
        self.d_ambient = d_ambient
        self.d_intrinsic = d_intrinsic
        self.projection_matrix = None  # (d_intrinsic, d_ambient)
        self.variance_explained = None
        
    def train(self, X: np.ndarray):
        """
        Learn projection via PCA on normalized vectors
        
        Args:
            X: Training data (N, d_ambient), will be normalized
        """
        # Normalize
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        
        # Fit PCA
        pca = PCA(n_components=self.d_intrinsic)
        pca.fit(X)
        
        # Store projection matrix
        self.projection_matrix = pca.components_  # (350, 768)
        self.variance_explained = pca.explained_variance_ratio_.sum()
        
        print(f"Variance explained by {self.d_intrinsic}D: {self.variance_explained:.3f}")
    
    def project(self, X: np.ndarray) -> np.ndarray:
        """
        Project to intrinsic subspace
        
        Args:
            X: (N, d_ambient) normalized
        
        Returns:
            X_intrinsic: (N, d_intrinsic) normalized
        """
        if self.projection_matrix is None:
            raise ValueError("Must call train() first")
        
        # Project
        X_intrinsic = X @ self.projection_matrix.T
        
        # Re-normalize (stay on sphere)
        X_intrinsic = X_intrinsic / np.linalg.norm(
            X_intrinsic, axis=1, keepdims=True
        )
        
        return X_intrinsic
    
    def save(self, path: str):
        """Save projection matrix"""
        np.savez(path, 
                 projection=self.projection_matrix,
                 variance=self.variance_explained)
    
    def load(self, path: str):
        """Load projection matrix"""
        data = np.load(path)
        self.projection_matrix = data['projection']
        self.variance_explained = data['variance']
