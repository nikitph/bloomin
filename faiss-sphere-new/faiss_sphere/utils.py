"""Utility functions for FAISS-Sphere"""

import numpy as np


def normalize_vectors(X: np.ndarray) -> np.ndarray:
    """
    Normalize vectors to unit sphere
    
    Args:
        X: Vectors (N, d)
    
    Returns:
        X_normalized: (N, d) with unit norm
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-12)
    return X / norms


def cosine_similarity(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between two sets of vectors
    
    Args:
        X: (N, d) normalized vectors
        Y: (M, d) normalized vectors
    
    Returns:
        similarities: (N, M) cosine similarities
    """
    return X @ Y.T


def geodesic_distance_exact(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute exact geodesic distances (without lookup table)
    
    Args:
        X: (N, d) normalized vectors
        Y: (M, d) normalized vectors
    
    Returns:
        distances: (N, M) geodesic distances in radians
    """
    similarities = cosine_similarity(X, Y)
    similarities = np.clip(similarities, -1.0, 1.0)
    return np.arccos(similarities)
