import time
import torch
import numpy as np
from typing import Dict, Any, List
try:
    import faiss
except ImportError:
    faiss = None

from .engine import WaveRetrievalEngine
from .utils import generate_synthetic_data, sparse_dense_mul

class AdvancedBenchmark:
    def __init__(self, n_samples: int = 2000, dim: int = 64, n_clusters: int = 10, n_queries: int = 20):
        self.n_samples = n_samples
        self.dim = dim
        self.n_clusters = n_clusters
        self.n_queries = n_queries
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generate Data with known clusters
        print(f"Generating clustered data (N={n_samples}, Clusters={n_clusters})...")
        # We need to manually generate data to keep track of cluster labels and centers
        self.data, self.labels, self.centers = self._generate_labeled_data()
        
        # Queries: Use the cluster centers themselves + noise to test basin retrieval
        # Or sample points near centers.
        # Let's use points near centers to see if we retrieve the center.
        self.queries = self._generate_queries_near_centers()
        
        print("Initializing Wave Engine...")
        self.engine = WaveRetrievalEngine(self.data, k_neighbors=15, use_cuda=torch.cuda.is_available())
        
        if faiss:
            self.index_faiss = faiss.IndexHNSWFlat(self.dim, 32)
            self.index_faiss.add(self.data.cpu().numpy().astype('float32'))
            
    def _generate_labeled_data(self):
        # Centers
        centers = torch.randn(self.n_clusters, self.dim).to(self.device) * 5.0
        
        data = []
        labels = []
        samples_per_cluster = self.n_samples // self.n_clusters
        
        for i in range(self.n_clusters):
            # Cluster spread
            cluster_data = torch.randn(samples_per_cluster, self.dim).to(self.device) + centers[i]
            data.append(cluster_data)
            labels.extend([i] * samples_per_cluster)
            
        data = torch.vstack(data)
        labels = torch.tensor(labels).to(self.device)
        return data, labels, centers

    def _generate_queries_near_centers(self):
        # Generate queries that belong to specific clusters but are offset
        # We initiate 2 queries per cluster
        queries = []
        for i in range(self.n_clusters):
            # Take center and add noise
            q1 = self.centers[i] + torch.randn(self.dim).to(self.device) * 1.5
            q2 = self.centers[i] + torch.randn(self.dim).to(self.device) * 1.5
            queries.append(q1)
            queries.append(q2)
        return torch.stack(queries)[:self.n_queries]

    def run_suite(self):
        # Metrics storage
        metrics = {
            'Wave': {'basin_purity': [], 'center_recall': [], 'robustness': []},
            'Heat': {'basin_purity': [], 'center_recall': [], 'robustness': []},
            'FAISS': {'basin_purity': [], 'center_recall': [], 'robustness': []}
        }
        
        print("\n--- Running Basin Purity & Center Recall ---")
        for q_idx, query in enumerate(self.queries):
            # True cluster for this query (based on generation logic)
            # queries are generated 2 per cluster: 0,1 -> cluster 0; 2,3 -> cluster 1...
            true_cluster_id = q_idx // 2
            
            # 1. Wave
            wave_idx, _ = self.engine.retrieve(query, top_k=10,
                                             wave_params={'T_wave': 10, 'sigma': 10.0},
                                             telegrapher_params={'T_damp': 10, 'gamma': 0.5},
                                             poisson_params={'alpha': 0.1})
            
            metrics['Wave']['basin_purity'].append(self._calc_purity(wave_idx, true_cluster_id))
            metrics['Wave']['center_recall'].append(self._calc_center_proximity(wave_idx, true_cluster_id))
            
            # 2. Heat (Diffusion)
            # Heat baseline: simple diffusion steps=10
            heat_idx = self._run_heat(query, steps=10)
            metrics['Heat']['basin_purity'].append(self._calc_purity(heat_idx, true_cluster_id))
            metrics['Heat']['center_recall'].append(self._calc_center_proximity(heat_idx, true_cluster_id))

            # 3. FAISS
            if faiss:
                _, faiss_result = self.index_faiss.search(query.cpu().numpy().reshape(1, -1), 10)
                faiss_idx = torch.tensor(faiss_result[0]).to(self.device)
                metrics['FAISS']['basin_purity'].append(self._calc_purity(faiss_idx, true_cluster_id))
                metrics['FAISS']['center_recall'].append(self._calc_center_proximity(faiss_idx, true_cluster_id))

        self._print_summary("Basin Retrieval", metrics)
        
        print("\n--- Running Robustness Test (Noise Resistance) ---")
        # Add noise to queries and see if result set overlaps with clean result set
        noise_levels = [0.1, 0.5, 1.0, 2.0]
        
        for noise in noise_levels:
            print(f"Noise Level: {noise}")
            r_metrics = {'Wave': [], 'Heat': [], 'FAISS': []}
            
            for q_idx, query in enumerate(self.queries[:5]): # Subset for speed
                noisy_query = query + torch.randn_like(query) * noise
                
                # Baseline Clean Results (pre-computed/re-computed)
                w_clean, _ = self.engine.retrieve(query, top_k=10, wave_params={'T_wave': 10, 'sigma': 10.0})
                h_clean = self._run_heat(query, steps=10)
                if faiss:
                     _, f_res = self.index_faiss.search(query.cpu().numpy().reshape(1, -1), 10)
                     f_clean = torch.tensor(f_res[0]).to(self.device)
                
                # Noisy Results
                w_noisy, _ = self.engine.retrieve(noisy_query, top_k=10, wave_params={'T_wave': 10, 'sigma': 10.0})
                h_noisy = self._run_heat(noisy_query, steps=10)
                
                r_metrics['Wave'].append(self._jaccard(w_clean, w_noisy))
                r_metrics['Heat'].append(self._jaccard(h_clean, h_noisy))
                
                if faiss:
                     _, f_res_n = self.index_faiss.search(noisy_query.cpu().numpy().reshape(1, -1), 10)
                     f_noisy = torch.tensor(f_res_n[0]).to(self.device)
                     r_metrics['FAISS'].append(self._jaccard(f_clean, f_noisy))
            
            print(f"  Wave Stability: {np.mean(r_metrics['Wave']):.2f}")
            print(f"  Heat Stability: {np.mean(r_metrics['Heat']):.2f}")
            if faiss:
                print(f"  FAISS Stability: {np.mean(r_metrics['FAISS']):.2f}")
    
    def _run_heat(self, query, steps=10):
        sigma = 10.0
        dists = torch.cdist(query.view(1, -1), self.data)
        u = torch.exp(-dists.view(-1)**2 / (2 * sigma**2))
        dt = 0.1
        for _ in range(steps):
             du = -sparse_dense_mul(self.engine.L, u)
             u = u + dt * du
        _, indices = torch.topk(u, 10, largest=True)
        return indices

    def _calc_purity(self, indices, true_cluster_id):
        # Fraction of retrieved points that belong to the true cluster
        retrieved_labels = self.labels[indices]
        hits = (retrieved_labels == true_cluster_id).float().sum()
        return (hits / len(indices)).item()

    def _calc_center_proximity(self, indices, true_cluster_id):
        # 1 if ANY retrieved point is very close to true center (index of center closest to mean of cluster)
        # Actually easier: check if retrieved points are closer to true center than other centers
        # Or simpler: Distance to true center
        
        # Let's use: Inverse distance to true center of the retrieved basin
        retrieved_points = self.data[indices]
        center = self.centers[true_cluster_id]
        
        avg_dist = torch.norm(retrieved_points - center, dim=1).mean()
        return 1.0 / (1.0 + avg_dist.item())

    def _jaccard(self, a, b):
        s1 = set(a.cpu().numpy())
        s2 = set(b.cpu().numpy())
        return len(s1.intersection(s2)) / len(s1.union(s2))

    def _print_summary(self, title, metrics):
        print(f"\n[{title}]")
        for method, values in metrics.items():
            if not values['basin_purity']: continue
            purity = np.mean(values['basin_purity'])
            center_score = np.mean(values['center_recall'])
            print(f"{method}: Purity={purity:.2f}, CenterScore={center_score:.2f}")

if __name__ == '__main__':
    bench = AdvancedBenchmark()
    bench.run_suite()
