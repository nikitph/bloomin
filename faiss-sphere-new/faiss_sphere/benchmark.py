"""
Benchmarking utilities
"""

import numpy as np
import time
from typing import Tuple


class Benchmark:
    def __init__(self, name: str):
        self.name = name
        self.results = []
    
    def add_method(self, 
                   method_name: str,
                   index,
                   query: np.ndarray,
                   ground_truth_indices: np.ndarray,
                   k: int = 10):
        """
        Benchmark one method
        
        Args:
            method_name: Name for results
            index: Index object with search() method
            query: Query vectors (N_query, d)
            ground_truth_indices: (N_query, k) true nearest neighbors
            k: Number of neighbors
        """
        print(f"\nBenchmarking {method_name}...")
        
        # Warmup
        _ = index.search(query[:10], k)
        
        # Timing
        start = time.time()
        distances, indices = index.search(query, k)
        elapsed = time.time() - start
        
        # Compute recall
        recall = self._compute_recall(indices, ground_truth_indices)
        
        # Memory
        memory_mb = 0
        if hasattr(index, 'benchmark_stats'):
            stats = index.benchmark_stats()
            memory_mb = stats['memory_mb']
        else:
            # Estimate for FAISS
            import sys
            memory_mb = sys.getsizeof(index) / 1e6
        
        result = {
            'method': method_name,
            'qps': len(query) / elapsed,
            'latency_ms': (elapsed / len(query)) * 1000,
            'recall@10': recall,
            'memory_mb': memory_mb
        }
        
        self.results.append(result)
        print(f"  QPS: {result['qps']:.1f}")
        print(f"  Latency: {result['latency_ms']:.2f} ms")
        print(f"  Recall@10: {result['recall@10']:.3f}")
        print(f"  Memory: {result['memory_mb']:.1f} MB")
        
        return result
    
    def _compute_recall(self, 
                       predicted: np.ndarray, 
                       ground_truth: np.ndarray) -> float:
        """
        Compute recall@k
        
        Args:
            predicted: (N, k) predicted indices
            ground_truth: (N, k) true indices
        
        Returns:
            recall: Average recall across queries
        """
        recalls = []
        for i in range(len(predicted)):
            true_set = set(ground_truth[i])
            pred_set = set(predicted[i])
            recall = len(true_set & pred_set) / len(true_set)
            recalls.append(recall)
        
        return np.mean(recalls)
    
    def print_summary(self):
        """Print formatted results table"""
        import pandas as pd
        
        df = pd.DataFrame(self.results)
        
        # Add relative metrics
        baseline = df[df['method'] == 'FAISS Flat']
        if len(baseline) > 0:
            baseline_qps = baseline['qps'].values[0]
            baseline_memory = baseline['memory_mb'].values[0]
            
            df['speedup'] = df['qps'] / baseline_qps
            df['memory_ratio'] = baseline_memory / df['memory_mb']
        
        print("\n" + "="*80)
        print(f"BENCHMARK RESULTS: {self.name}")
        print("="*80)
        print(df.to_string(index=False))
        print("="*80)
        
        return df
    
    def save_results(self, path: str):
        """Save results to JSON"""
        import json
        
        with open(path, 'w') as f:
            json.dump({
                'benchmark': self.name,
                'results': self.results
            }, f, indent=2)
