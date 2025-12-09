"""
FAISS Baseline Implementation

Creates FAISS indexes (Flat, HNSW, IVF) for comparison with SemantixBit.
"""

import numpy as np
import faiss
import json
import time
from typing import List, Tuple

class FAISSBaseline:
    """FAISS baseline for vector search"""
    
    def __init__(self, dimension: int, index_type: str = "hnsw"):
        """
        Initialize FAISS index
        
        Args:
            dimension: Vector dimension
            index_type: Type of index ('flat', 'hnsw', 'ivf')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.doc_ids = []
        
        if index_type == "flat":
            # Exact search (ground truth)
            self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine for normalized vectors)
        
        elif index_type == "hnsw":
            # HNSW approximate search
            M = 32  # Number of connections per layer
            self.index = faiss.IndexHNSWFlat(dimension, M)
            self.index.hnsw.efConstruction = 200  # Build-time search depth
            self.index.hnsw.efSearch = 100  # Query-time search depth
        
        elif index_type == "ivf":
            # IVF approximate search
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        else:
            raise ValueError(f"Unknown index type: {index_type}")
    
    def add(self, vectors: np.ndarray, doc_ids: List[str]):
        """Add vectors to index"""
        assert vectors.shape[1] == self.dimension
        
        # Train IVF index if needed
        if self.index_type == "ivf" and not self.index.is_trained:
            print("Training IVF index...")
            self.index.train(vectors)
        
        self.index.add(vectors)
        self.doc_ids.extend(doc_ids)
    
    def search(self, query_vectors: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top-k nearest neighbors
        
        Returns:
            distances: Array of shape (num_queries, k)
            indices: Array of shape (num_queries, k)
        """
        distances, indices = self.index.search(query_vectors, k)
        return distances, indices
    
    def get_doc_ids(self, indices: np.ndarray) -> List[List[str]]:
        """Convert indices to document IDs"""
        return [[self.doc_ids[idx] for idx in row] for row in indices]
    
    def memory_usage(self) -> int:
        """Get memory usage in bytes"""
        # Approximate memory usage
        if self.index_type == "flat":
            return self.index.ntotal * self.dimension * 4  # float32
        elif self.index_type == "hnsw":
            # HNSW has overhead for graph structure
            return self.index.ntotal * self.dimension * 4 + self.index.ntotal * 32 * 8
        else:
            return self.index.ntotal * self.dimension * 4
    
    def save(self, path: str):
        """Save index to disk"""
        faiss.write_index(self.index, path)
        
        # Save doc IDs separately
        with open(path + ".ids.json", 'w') as f:
            json.dump(self.doc_ids, f)
    
    def load(self, path: str):
        """Load index from disk"""
        self.index = faiss.read_index(path)
        
        with open(path + ".ids.json", 'r') as f:
            self.doc_ids = json.load(f)


def benchmark_faiss(embeddings: np.ndarray, doc_ids: List[str], 
                    query_embeddings: np.ndarray, k: int = 10):
    """Benchmark FAISS indexes"""
    
    dimension = embeddings.shape[1]
    results = {}
    
    for index_type in ["flat", "hnsw"]:
        print(f"\n=== Benchmarking FAISS {index_type.upper()} ===")
        
        # Build index
        baseline = FAISSBaseline(dimension, index_type)
        
        start = time.time()
        baseline.add(embeddings, doc_ids)
        build_time = time.time() - start
        
        print(f"Build time: {build_time:.2f}s")
        print(f"Memory usage: {baseline.memory_usage() / 1024 / 1024:.2f} MB")
        
        # Search
        start = time.time()
        distances, indices = baseline.search(query_embeddings, k)
        search_time = time.time() - start
        
        qps = len(query_embeddings) / search_time
        latency_per_query = search_time / len(query_embeddings) * 1000  # ms
        
        print(f"Search time: {search_time:.2f}s")
        print(f"Queries per second: {qps:.2f}")
        print(f"Latency per query: {latency_per_query:.2f}ms")
        
        results[index_type] = {
            'build_time': build_time,
            'search_time': search_time,
            'qps': qps,
            'latency_ms': latency_per_query,
            'memory_mb': baseline.memory_usage() / 1024 / 1024,
            'indices': indices,
            'distances': distances
        }
    
    return results


if __name__ == "__main__":
    # Load data
    print("Loading Wikipedia embeddings...")
    embeddings = np.load("data/wikipedia/embeddings.npy")
    
    with open("data/wikipedia/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    doc_ids = [m['id'] for m in metadata]
    
    # Create query set (use first 1000 as queries)
    num_queries = 1000
    query_embeddings = embeddings[:num_queries]
    query_ids = doc_ids[:num_queries]
    
    # Benchmark
    results = benchmark_faiss(embeddings, doc_ids, query_embeddings, k=10)
    
    # Save results
    with open("results/faiss_baseline.json", 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for key, val in results.items():
            serializable_results[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in val.items()
            }
        json.dump(serializable_results, f, indent=2)
    
    print("\nResults saved to results/faiss_baseline.json")
