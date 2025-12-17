import time
import torch
import numpy as np
from typing import Dict, Any, List
try:
    import faiss
except ImportError:
    faiss = None
    print("FAISS not found, using brute force for ground truth.")

from .engine import WaveRetrievalEngine
from .utils import generate_synthetic_data, sparse_dense_mul

class Benchmark:
    def __init__(self, n_samples: int = 1000, dim: int = 64, n_queries: int = 10, k: int = 10):
        self.n_samples = n_samples
        self.dim = dim
        self.n_queries = n_queries
        self.k = k
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Generating synthetic data (N={n_samples}, D={dim})...")
        self.data = generate_synthetic_data(n_samples, dim).to(self.device)
        self.queries = generate_synthetic_data(n_queries, dim).to(self.device) # Sample from same distribution
        
        # Ground Truth
        print("Computing Ground Truth...")
        self.gt_indices = self._compute_ground_truth()
        
        # Initialize Engine
        print("Initializing Wave Engine...")
        self.engine = WaveRetrievalEngine(self.data, k_neighbors=15, use_cuda=torch.cuda.is_available())
        
    def _compute_ground_truth(self):
        # Brute force exact search
        dists = torch.cdist(self.queries, self.data)
        _, indices = torch.topk(dists, self.k, largest=False)
        return indices.cpu().numpy()
        
    def run_all(self):
        results = {}
        
        # 1. Wave Engine
        print("Running Wave Engine...")
        start_time = time.time()
        wave_preds = []
        for i in range(self.n_queries):
            preds, _ = self.engine.retrieve(self.queries[i], top_k=self.k, 
                                          wave_params={'T_wave': 10, 'c': 1.0, 'sigma': 10.0},
                                          telegrapher_params={'T_damp': 10, 'gamma': 0.5},
                                          poisson_params={'alpha': 0.1})
            wave_preds.append(preds.cpu().numpy())
        wave_time = (time.time() - start_time) / self.n_queries
        results['Wave'] = {'recall': self._calc_recall(wave_preds), 'latency': wave_time}
        
        # 2. Diffusion / Heat Only (Baseline)
        # Use Engine but 0 gamma and no Poisson? Or just specialized heat run.
        # Let's simulate heat by just diffusion on graph: u_t = c^2 L u
        # We can implement a quick heat baseline here
        print("Running Heat Baseline...")
        start_time = time.time()
        heat_preds = []
        for i in range(self.n_queries):
            preds = self._heat_baseline(self.queries[i])
            heat_preds.append(preds.cpu().numpy())
        heat_time = (time.time() - start_time) / self.n_queries
        results['Heat'] = {'recall': self._calc_recall(heat_preds), 'latency': heat_time}
        
        # 3. FAISS (IVF or HNSW)
        if faiss:
            print("Running FAISS HNSW Baseline...")
            index = faiss.IndexHNSWFlat(self.dim, 32)
            # FAISS expects numpy
            data_np = self.data.cpu().numpy().astype('float32')
            queries_np = self.queries.cpu().numpy().astype('float32')
            
            train_start = time.time()
            index.add(data_np)
            train_time = time.time() - train_start
            
            start_time = time.time()
            _, faiss_preds = index.search(queries_np, self.k)
            faiss_time = (time.time() - start_time) / self.n_queries
            
            results['FAISS_HNSW'] = {'recall': self._calc_recall(faiss_preds), 'latency': faiss_time}
            
        return results
        
    def _heat_baseline(self, query):
        # Simple heat diffusion: u_t = -L u
        # Initial: Gaussian
        sigma = 10.0
        dists = torch.cdist(query.view(1, -1), self.data)
        u = torch.exp(-dists.view(-1)**2 / (2 * sigma**2))
        
        # Diffuse
        dt = 0.1
        steps = 10 # Sanity check: Should match GT
        for _ in range(steps):
             # L is positive semi-def, so -L is diffusion
             du = -sparse_dense_mul(self.engine.L, u)
             u = u + dt * du
             
        # Retrieval: highest heat
        _, indices = torch.topk(u, self.k, largest=True)
        return indices
        
    def _calc_recall(self, predictions):
        # predictions: list of (k,) arrays
        total_hits = 0
        total_expected = self.n_queries * self.k
        
        # Debug print for first query
        if self.n_queries > 0:
            print(f"Debug Recall Q0: GT={self.gt_indices[0]}, Pred={predictions[0]}")
        
        for i in range(self.n_queries):
            gt_set = set(self.gt_indices[i])
            pred_set = set(predictions[i])
            total_hits += len(gt_set.intersection(pred_set))
            
        return total_hits / total_expected

if __name__ == "__main__":
    benchmark = Benchmark(n_samples=2000, dim=64, n_queries=10, k=10)
    results = benchmark.run_all()
    
    print("\n--- Results (Recall@10, Latency per query) ---")
    for name, metrics in results.items():
        print(f"{name}: Recall={metrics['recall']:.4f}, Latency={metrics['latency']:.4f}s")
