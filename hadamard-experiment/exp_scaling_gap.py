import numpy as np
from scipy.linalg import hadamard
from sklearn.metrics.pairwise import euclidean_distances

def run_scaling_gap(N=2000, d=1024, m=128):
    print(f"--- Experiment 1: Scaling the Gap (Sigma Sweep) ---")
    print(f"{'Sigma':<10} | {'RP Recall':<10} {'WP Recall':<10}")
    print("-" * 40)
    
    sigmas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    for sigma in sigmas:
        np.random.seed(42)
        # Base Data
        X = np.random.randn(N, d)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        
        # Plant Cluster
        center = np.random.randn(d)
        center /= np.linalg.norm(center)
        
        # 10 points around center
        for i in range(10):
            point = center + np.random.randn(d) * sigma
            point /= np.linalg.norm(point)
            X[i] = point
            
        # Query near center
        Q = (center + np.random.randn(d) * sigma).reshape(1, -1)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        
        # Ground Truth
        dists_gt = euclidean_distances(Q, X)[0]
        gt_indices = np.argsort(dists_gt)[:10]
        
        # Random Projection
        G = np.random.randn(d, m)
        X_rp = X @ G
        Q_rp = Q @ G
        dists_rp = euclidean_distances(Q_rp, X_rp)[0]
        rp_indices = np.argsort(dists_rp)[:10]
        rp_recall = len(set(gt_indices).intersection(rp_indices)) / 10
        
        # Witness Polar
        D = np.random.choice([-1, 1], size=d)
        H = hadamard(d)
        X_wp = (X * D) @ H
        Q_wp = (Q * D) @ H
        X_wp = X_wp[:, :m]
        Q_wp = Q_wp[:, :m]
        dists_wp = euclidean_distances(Q_wp, X_wp)[0]
        wp_indices = np.argsort(dists_wp)[:10]
        wp_recall = len(set(gt_indices).intersection(wp_indices)) / 10
        
        print(f"{sigma:<10} | {rp_recall:.2f}       {wp_recall:.2f}")

if __name__ == "__main__":
    run_scaling_gap()
