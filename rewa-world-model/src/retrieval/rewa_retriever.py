"""
REWA Retrieval Module

Implements fast retrieval using REWA-encoded signatures with different distance metrics.
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

from encoding import hamming_distance, l1_distance, tropical_distance
from witnesses import WitnessType

@dataclass
class RetrievalResult:
    """A single retrieval result"""
    doc_id: str
    distance: float
    score: float  # Normalized similarity score

class REWARetriever:
    """Fast retrieval using REWA signatures"""
    
    def __init__(self, witness_type: WitnessType):
        self.witness_type = witness_type
        self.index: Dict[str, np.ndarray] = {}
        self.doc_ids: List[str] = []
        
        # Choose distance function based on witness type
        if witness_type == WitnessType.BOOLEAN:
            self.distance_fn = hamming_distance
        elif witness_type in [WitnessType.NATURAL, WitnessType.REAL]:
            self.distance_fn = l1_distance
        elif witness_type == WitnessType.TROPICAL:
            self.distance_fn = tropical_distance
        else:
            raise ValueError(f"Unknown witness type: {witness_type}")
    
    def add(self, doc_id: str, signature: np.ndarray):
        """Add a document signature to the index"""
        self.index[doc_id] = signature
        if doc_id not in self.doc_ids:
            self.doc_ids.append(doc_id)
    
    def search(self, query_signature: np.ndarray, k: int = 10) -> List[RetrievalResult]:
        """
        Search for top-k most similar documents
        
        Args:
            query_signature: REWA signature of query
            k: Number of results to return
            
        Returns:
            List of RetrievalResults sorted by similarity
        """
        if len(self.index) == 0:
            return []
        
        # Compute distances to all documents
        distances = []
        for doc_id in self.doc_ids:
            doc_sig = self.index[doc_id]
            dist = self.distance_fn(query_signature, doc_sig)
            distances.append((doc_id, dist))
        
        # Sort by distance (ascending)
        distances.sort(key=lambda x: x[1])
        
        # Take top-k and convert to results
        results = []
        max_dist = max(d[1] for d in distances) if distances else 1.0
        
        for doc_id, dist in distances[:k]:
            # Convert distance to similarity score [0, 1]
            score = 1.0 - (dist / max_dist) if max_dist > 0 else 1.0
            results.append(RetrievalResult(
                doc_id=doc_id,
                distance=dist,
                score=score
            ))
        
        return results
    
    def batch_search(self, query_signatures: List[np.ndarray], k: int = 10) -> List[List[RetrievalResult]]:
        """Search for multiple queries"""
        return [self.search(q, k) for q in query_signatures]
    
    def size(self) -> int:
        """Get number of documents in index"""
        return len(self.index)
    
    def memory_usage(self) -> int:
        """Estimate memory usage in bytes"""
        if len(self.index) == 0:
            return 0
        
        # Signature size
        sig_size = next(iter(self.index.values())).nbytes
        total_sig = sig_size * len(self.index)
        
        # Doc ID size (rough estimate)
        total_ids = sum(len(doc_id) for doc_id in self.doc_ids)
        
        return total_sig + total_ids

def evaluate_retrieval(
    retriever: REWARetriever,
    queries: List[Tuple[np.ndarray, str]],
    ground_truth: Dict[str, List[str]],
    k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Evaluate retrieval performance
    
    Args:
        retriever: REWA retriever
        queries: List of (signature, query_id) tuples
        ground_truth: Dict mapping query_id to list of relevant doc_ids
        k_values: K values for Recall@K
        
    Returns:
        Dict of metrics
    """
    metrics = {}
    
    for k in k_values:
        total_recall = 0.0
        total_precision = 0.0
        
        for query_sig, query_id in queries:
            results = retriever.search(query_sig, k)
            retrieved = {r.doc_id for r in results}
            relevant = set(ground_truth.get(query_id, []))
            
            if len(relevant) > 0:
                recall = len(retrieved & relevant) / len(relevant)
                total_recall += recall
            
            if len(retrieved) > 0:
                precision = len(retrieved & relevant) / len(retrieved)
                total_precision += precision
        
        n_queries = len(queries)
        metrics[f'recall@{k}'] = total_recall / n_queries if n_queries > 0 else 0.0
        metrics[f'precision@{k}'] = total_precision / n_queries if n_queries > 0 else 0.0
    
    return metrics
