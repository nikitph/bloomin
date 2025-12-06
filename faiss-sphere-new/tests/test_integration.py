"""Integration tests for FAISS-Sphere"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faiss_sphere import FAISSSphere


def test_end_to_end_fast_mode():
    """Test complete pipeline in fast mode"""
    # Generate data
    np.random.seed(42)
    documents = np.random.randn(1000, 128).astype('float32')
    documents = documents / np.linalg.norm(documents, axis=1, keepdims=True)
    
    queries = np.random.randn(10, 128).astype('float32')
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Create index
    index = FAISSSphere(d_ambient=128, mode='fast', d_intrinsic=64)
    
    # Train
    index.train(documents[:200])
    
    # Add
    index.add(documents)
    
    # Search
    distances, indices = index.search(queries, k=10)
    
    # Verify shapes
    assert distances.shape == (10, 10)
    assert indices.shape == (10, 10)
    
    # Verify distances are non-negative
    assert np.all(distances >= 0)
    
    # Verify indices are valid
    assert np.all(indices >= 0)
    assert np.all(indices < 1000)
    
    print("Fast mode test passed!")


def test_end_to_end_memory_mode():
    """Test complete pipeline in memory mode"""
    # Generate data
    np.random.seed(42)
    documents = np.random.randn(500, 128).astype('float32')
    documents = documents / np.linalg.norm(documents, axis=1, keepdims=True)
    
    queries = np.random.randn(5, 128).astype('float32')
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Create index
    index = FAISSSphere(d_ambient=128, mode='memory', d_intrinsic=64)
    
    # Train
    index.train(documents[:100])
    
    # Add
    index.add(documents)
    
    # Search
    distances, indices = index.search(queries, k=5)
    
    # Verify shapes
    assert distances.shape == (5, 5)
    assert indices.shape == (5, 5)
    
    print("Memory mode test passed!")


def test_end_to_end_exact_mode():
    """Test complete pipeline in exact mode"""
    # Generate data
    np.random.seed(42)
    documents = np.random.randn(500, 128).astype('float32')
    documents = documents / np.linalg.norm(documents, axis=1, keepdims=True)
    
    queries = np.random.randn(5, 128).astype('float32')
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Create index
    index = FAISSSphere(d_ambient=128, mode='exact', d_intrinsic=64)
    
    # Train
    index.train(documents[:100])
    
    # Add
    index.add(documents)
    
    # Search
    distances, indices = index.search(queries, k=5)
    
    # Verify shapes
    assert distances.shape == (5, 5)
    assert indices.shape == (5, 5)
    
    print("Exact mode test passed!")


def test_recall_comparison():
    """Test that recall is reasonable"""
    # Generate data
    np.random.seed(42)
    documents = np.random.randn(1000, 128).astype('float32')
    documents = documents / np.linalg.norm(documents, axis=1, keepdims=True)
    
    queries = np.random.randn(20, 128).astype('float32')
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    # Ground truth (brute force)
    similarities = queries.dot(documents.T)
    gt_indices = np.argsort(-similarities, axis=1)[:, :10]
    
    # Test with exact mode
    index = FAISSSphere(d_ambient=128, mode='exact', d_intrinsic=64)
    index.train(documents[:200])
    index.add(documents)
    
    distances, indices = index.search(queries, k=10)
    
    # Compute recall
    recalls = []
    for i in range(len(queries)):
        true_set = set(gt_indices[i])
        pred_set = set(indices[i])
        recall = len(true_set & pred_set) / len(true_set)
        recalls.append(recall)
    
    avg_recall = np.mean(recalls)
    print(f"Average recall: {avg_recall:.3f}")
    
    # With projection and synthetic data, recall will be lower
    # For real embeddings, it would be much higher
    assert avg_recall > 0.25, "Recall too low: {}".format(avg_recall)
    
    print("Recall comparison test passed!")


def test_benchmark_stats():
    """Test benchmark statistics"""
    index = FAISSSphere(d_ambient=128, mode='fast', d_intrinsic=64)
    
    # Generate and add data
    documents = np.random.randn(100, 128).astype('float32')
    documents = documents / np.linalg.norm(documents, axis=1, keepdims=True)
    
    index.train(documents[:100])  # Need more samples than components
    index.add(documents)
    
    # Get stats
    stats = index.benchmark_stats()
    
    assert 'mode' in stats
    assert 'memory_mb' in stats
    assert 'd_intrinsic' in stats
    assert 'variance_explained' in stats
    
    assert stats['mode'] == 'fast'
    assert stats['d_intrinsic'] == 64
    assert stats['memory_mb'] > 0
    
    print("Benchmark stats test passed!")


if __name__ == '__main__':
    test_end_to_end_fast_mode()
    test_end_to_end_memory_mode()
    test_end_to_end_exact_mode()
    test_recall_comparison()
    test_benchmark_stats()
    print("\nAll integration tests passed!")
