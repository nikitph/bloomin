import numpy as np
import time
from scipy.linalg import hadamard
from sklearn.metrics.pairwise import euclidean_distances

class SimilarityExperiment:
    def __init__(self, N=2000, d=1024, m_bits=128):
        self.N = N
        self.d = d
        self.m = m_bits
        
        print(f"--- SETTING UP EXPERIMENT (PLANTED CLUSTER) ---")
        
        # 1. Generate Base Data (Background Noise)
        np.random.seed(42)
        self.X = np.random.randn(N, d)
        self.X /= np.linalg.norm(self.X, axis=1, keepdims=True)
        
        # 2. PLANT A CLUSTER (Indices 0-9)
        # Generate a center
        center = np.random.randn(d)
        center /= np.linalg.norm(center)
        
        # Scale noise to be small relative to unit norm
        # sqrt(1024) = 32. 
        # We want noise norm to be ~0.3 (30% of signal).
        # So sigma * 32 = 0.3 => sigma = 0.01
        sigma = 0.01
        
        # Make the first 10 points clustered around this center
        for i in range(10):
            point = center + np.random.randn(d) * sigma
            point /= np.linalg.norm(point)
            self.X[i] = point
            
        # 3. Create Query close to the same center
        # Query is also near the center
        self.Q = (center + np.random.randn(d) * sigma).reshape(1, -1)
        self.Q /= np.linalg.norm(self.Q, axis=1, keepdims=True)

    def get_ground_truth(self, k=10):
        # We expect indices 0-9 to be at the top
        dists = euclidean_distances(self.Q, self.X)
        return np.argsort(dists[0])[:k]

    def run_random_projection(self, ground_truth_indices, k=10):
        start_t = time.time()
        G = np.random.randn(self.d, self.m)
        X_enc = self.X @ G
        Q_enc = self.Q @ G
        encode_time = time.time() - start_t
        
        dists = euclidean_distances(Q_enc, X_enc)
        pred_indices = np.argsort(dists[0])[:k]
        
        # Check intersection
        intersection = len(set(ground_truth_indices).intersection(pred_indices))
        return intersection / k, encode_time

    def run_witness_polar(self, ground_truth_indices, k=10):
        start_t = time.time()
        D = np.random.choice([-1, 1], size=self.d)
        
        # Hadamard Transform
        H = hadamard(self.d)
        X_polar = (self.X * D) @ H
        Q_polar = (self.Q * D) @ H
        
        # Keep top m components
        X_enc = X_polar[:, :self.m]
        Q_enc = Q_polar[:, :self.m]
        
        encode_time = time.time() - start_t
        
        dists = euclidean_distances(Q_enc, X_enc)
        pred_indices = np.argsort(dists[0])[:k]
        
        intersection = len(set(ground_truth_indices).intersection(pred_indices))
        return intersection / k, encode_time

# --- EXECUTION ---
exp = SimilarityExperiment(N=2000, d=1024, m_bits=128)
gt = exp.get_ground_truth(k=10)
print(f"Ground Truth Indices (Should be 0-9): {gt}")

rp_recall, _ = exp.run_random_projection(gt)
print(f"Random Recall: {rp_recall*100:.1f}%")

wp_recall, _ = exp.run_witness_polar(gt)
print(f"Polar Recall:  {wp_recall*100:.1f}%")

print(f"\n--- CONCLUSION ---")
if abs(wp_recall - rp_recall) < 0.2 and wp_recall > 0.8:
    print("SUCCESS: High accuracy and parity achieved.")
else:
    print("FAILURE: Low accuracy or discrepancy.")
