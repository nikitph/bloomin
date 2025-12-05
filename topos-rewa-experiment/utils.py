"""
Utility functions for Topos-REWA experiment
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import softmax as scipy_softmax


def kl_divergence(p, q, epsilon=1e-10):
    """
    Compute KL divergence KL(p || q)
    
    Args:
        p: Probability distribution (numpy array)
        q: Probability distribution (numpy array)
        epsilon: Small constant to avoid log(0)
    
    Returns:
        KL divergence value
    """
    p = np.clip(p, epsilon, 1.0)
    q = np.clip(q, epsilon, 1.0)
    return np.sum(p * np.log(p / q))


def fisher_distance(p, q):
    """
    Approximation of Fisher-Rao distance
    d_F ≈ sqrt(2 * KL(p || q))
    
    Args:
        p: Probability distribution
        q: Probability distribution
    
    Returns:
        Fisher distance approximation
    """
    kl = kl_divergence(p, q)
    return np.sqrt(2 * kl)


def softmax(x, beta=1.0):
    """
    Compute softmax with temperature parameter
    
    Args:
        x: Input array
        beta: Temperature parameter (inverse)
    
    Returns:
        Softmax probabilities
    """
    return scipy_softmax(-x / beta)


def euclidean_distance(x, y):
    """
    Compute Euclidean distance between vectors or sets of vectors
    
    Args:
        x: Vector or matrix (N x D)
        y: Vector or matrix (M x D)
    
    Returns:
        Distance(s)
    """
    if x.ndim == 1 and y.ndim == 1:
        return np.linalg.norm(x - y)
    elif x.ndim == 1:
        return np.linalg.norm(y - x, axis=1)
    elif y.ndim == 1:
        return np.linalg.norm(x - y, axis=1)
    else:
        return cdist(x, y, metric='euclidean')


def evaluate_retrieval(retrieved_items, ground_truth_items):
    """
    Compute precision and recall for retrieval task
    
    Args:
        retrieved_items: List of retrieved item indices
        ground_truth_items: List of ground truth item indices
    
    Returns:
        Dictionary with precision, recall, f1
    """
    retrieved_set = set(retrieved_items)
    ground_truth_set = set(ground_truth_items)
    
    if len(retrieved_set) == 0:
        precision = 0.0
    else:
        precision = len(retrieved_set & ground_truth_set) / len(retrieved_set)
    
    if len(ground_truth_set) == 0:
        recall = 0.0
    else:
        recall = len(retrieved_set & ground_truth_set) / len(ground_truth_set)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_retrieved': len(retrieved_set),
        'n_ground_truth': len(ground_truth_set),
        'n_correct': len(retrieved_set & ground_truth_set)
    }


def nearest_neighbors(query_vec, data_vecs, k=10):
    """
    Find k nearest neighbors using Euclidean distance
    
    Args:
        query_vec: Query vector (D,)
        data_vecs: Data vectors (N x D)
        k: Number of neighbors
    
    Returns:
        Indices of k nearest neighbors
    """
    distances = euclidean_distance(query_vec, data_vecs)
    return np.argsort(distances)[:k]
