"""Unit tests for Intrinsic Projector"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faiss_sphere.core.intrinsic_projector import IntrinsicProjector


def test_projector_initialization():
    """Test projector initialization"""
    proj = IntrinsicProjector(d_ambient=768, d_intrinsic=350)
    assert proj.d_ambient == 768
    assert proj.d_intrinsic == 350
    assert proj.projection_matrix is None


def test_projector_training():
    """Test projector training"""
    proj = IntrinsicProjector(d_ambient=128, d_intrinsic=64)
    
    # Generate training data
    X = np.random.randn(1000, 128).astype('float32')
    
    # Train
    proj.train(X)
    
    assert proj.projection_matrix is not None
    assert proj.projection_matrix.shape == (64, 128)
    assert proj.variance_explained is not None
    assert 0 < proj.variance_explained <= 1.0


def test_projector_preserves_normalization():
    """Test that projection preserves normalization"""
    proj = IntrinsicProjector(d_ambient=128, d_intrinsic=64)
    
    # Generate and train
    X_train = np.random.randn(500, 128).astype('float32')
    proj.train(X_train)
    
    # Project test data
    X_test = np.random.randn(100, 128).astype('float32')
    X_test = X_test / np.linalg.norm(X_test, axis=1, keepdims=True)
    
    X_projected = proj.project(X_test)
    
    # Check normalization
    norms = np.linalg.norm(X_projected, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Projected vectors should be normalized"


def test_projector_similarity_preservation():
    """Test that projection approximately preserves similarity"""
    proj = IntrinsicProjector(d_ambient=256, d_intrinsic=128)
    
    # Generate and train
    X_train = np.random.randn(1000, 256).astype('float32')
    proj.train(X_train)
    
    # Create two similar vectors
    v1 = np.random.randn(256).astype('float32')
    v1 = v1 / np.linalg.norm(v1)
    
    v2 = v1 + 0.1 * np.random.randn(256).astype('float32')
    v2 = v2 / np.linalg.norm(v2)
    
    # Original similarity
    sim_original = np.dot(v1, v2)
    
    # Project
    v1_proj = proj.project(v1.reshape(1, -1))[0]
    v2_proj = proj.project(v2.reshape(1, -1))[0]
    
    # Projected similarity
    sim_projected = np.dot(v1_proj, v2_proj)
    
    # Should be reasonably close
    assert abs(sim_original - sim_projected) < 0.3, "Similarity should be approximately preserved"


def test_variance_explained():
    """Test that variance explained is high enough"""
    proj = IntrinsicProjector(d_ambient=768, d_intrinsic=350)
    
    # Generate training data with some structure
    X = np.random.randn(2000, 768).astype('float32')
    
    proj.train(X)
    
    # For random data, variance explained should still be reasonable
    # (for real embeddings, it would be much higher)
    assert proj.variance_explained > 0.3, "Variance explained too low: {}".format(proj.variance_explained)


if __name__ == '__main__':
    test_projector_initialization()
    test_projector_training()
    test_projector_preserves_normalization()
    test_projector_similarity_preservation()
    test_variance_explained()
    print("All projector tests passed!")
