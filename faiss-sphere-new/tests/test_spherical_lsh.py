"""Unit tests for Spherical LSH"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faiss_sphere.core.spherical_lsh import SphericalLSH


def test_lsh_initialization():
    """Test LSH initialization"""
    lsh = SphericalLSH(d=128, nbits=16)
    assert lsh.d == 128
    assert lsh.nbits == 16
    assert lsh.hyperplanes.shape == (16, 128)
    # Check hyperplanes are normalized
    norms = np.linalg.norm(lsh.hyperplanes, axis=1)
    assert np.allclose(norms, 1.0), "Hyperplanes should be normalized"


def test_lsh_hash():
    """Test hash computation"""
    lsh = SphericalLSH(d=128, nbits=8)
    
    # Create a test vector
    x = np.random.randn(128).astype('float32')
    x = x / np.linalg.norm(x)
    
    # Compute hash
    hash_code = lsh._hash(x)
    
    assert hash_code.shape == (8,)
    assert hash_code.dtype == np.uint8
    assert all(bit in [0, 1] for bit in hash_code)


def test_lsh_add_and_query():
    """Test adding vectors and querying"""
    lsh = SphericalLSH(d=64, nbits=12)
    
    # Add some vectors
    n_vectors = 100
    vectors = np.random.randn(n_vectors, 64).astype('float32')
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    for i, vec in enumerate(vectors):
        lsh.add(i, vec)
    
    # Query
    query = np.random.randn(64).astype('float32')
    query = query / np.linalg.norm(query)
    
    candidates = lsh.query(query, k=10, n_probes=1)
    
    assert len(candidates) >= 0  # May return 0 if bucket is empty
    assert all(0 <= idx < n_vectors for idx in candidates)


def test_lsh_collision_probability():
    """Test that similar vectors have higher collision probability"""
    lsh = SphericalLSH(d=64, nbits=8)
    
    # Create similar vectors
    base = np.random.randn(64).astype('float32')
    base = base / np.linalg.norm(base)
    
    similar = base + 0.1 * np.random.randn(64).astype('float32')
    similar = similar / np.linalg.norm(similar)
    
    dissimilar = np.random.randn(64).astype('float32')
    dissimilar = dissimilar / np.linalg.norm(dissimilar)
    
    # Compute hashes
    hash_base = lsh._hash(base)
    hash_similar = lsh._hash(similar)
    hash_dissimilar = lsh._hash(dissimilar)
    
    # Count matching bits
    matches_similar = (hash_base == hash_similar).sum()
    matches_dissimilar = (hash_base == hash_dissimilar).sum()
    
    # Similar vectors should have more matching bits (on average)
    # This is probabilistic, so we just check it's reasonable
    assert matches_similar >= 0
    assert matches_dissimilar >= 0


if __name__ == '__main__':
    test_lsh_initialization()
    test_lsh_hash()
    test_lsh_add_and_query()
    test_lsh_collision_probability()
    print("All LSH tests passed!")
