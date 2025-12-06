"""Unit tests for Spherical PQ"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faiss_sphere.core.spherical_pq import SphericalPQ


def test_pq_initialization():
    """Test PQ initialization"""
    pq = SphericalPQ(d=128, M=8, nbits=8)
    assert pq.d == 128
    assert pq.M == 8
    assert pq.nbits == 8
    assert pq.n_clusters == 256
    assert pq.d_sub == 16


def test_pq_training():
    """Test PQ training"""
    pq = SphericalPQ(d=64, M=4, nbits=4)
    
    # Generate training data
    X = np.random.randn(200, 64).astype('float32')
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    # Train
    pq.train(X, max_iter=20)
    
    assert pq.codebooks is not None
    assert pq.codebooks.shape == (4, 16, 16)  # (M, n_clusters, d_sub)
    
    # Check centroids are normalized
    for m in range(pq.M):
        norms = np.linalg.norm(pq.codebooks[m], axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4), "Centroids should be normalized"


def test_pq_encode_decode():
    """Test encoding and decoding"""
    pq = SphericalPQ(d=64, M=4, nbits=4)
    
    # Generate and train
    X = np.random.randn(200, 64).astype('float32')
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    pq.train(X, max_iter=20)
    
    # Encode a vector
    x = np.random.randn(64).astype('float32')
    x = x / np.linalg.norm(x)
    
    codes = pq.encode(x)
    
    assert codes.shape == (4,)
    assert codes.dtype == np.uint8
    assert all(0 <= code < 16 for code in codes)
    
    # Decode
    x_reconstructed = pq.decode(codes)
    
    assert x_reconstructed.shape == (64,)
    assert np.allclose(np.linalg.norm(x_reconstructed), 1.0), "Reconstructed should be normalized"


def test_pq_distance_computation():
    """Test asymmetric distance computation"""
    pq = SphericalPQ(d=64, M=4, nbits=4)
    
    # Generate and train
    X = np.random.randn(200, 64).astype('float32')
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    pq.train(X, max_iter=20)
    
    # Encode a vector
    x = np.random.randn(64).astype('float32')
    x = x / np.linalg.norm(x)
    codes = pq.encode(x)
    
    # Compute distance
    dist = pq.compute_distance(x, codes)
    
    # Distance should be a similarity (0 to 1)
    assert -1 <= dist <= 1


if __name__ == '__main__':
    test_pq_initialization()
    test_pq_training()
    test_pq_encode_decode()
    test_pq_distance_computation()
    print("All PQ tests passed!")
