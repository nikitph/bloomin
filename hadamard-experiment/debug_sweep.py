import numpy as np
from scipy.linalg import hadamard
from sklearn.metrics.pairwise import euclidean_distances
import time

def run_experiment(N, d, m, normalize_projection=False):
    np.random.seed(42)
    # Data
    X = np.random.randn(N, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q = np.random.randn(1, d)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    
    # Ground Truth
    dists_gt = euclidean_distances(Q, X)[0]
    gt_indices = np.argsort(dists_gt)[:10]
    
    # Random Projection
    G = np.random.randn(d, m)
    X_rp = X @ G
    Q_rp = Q @ G
    
    if normalize_projection:
        X_rp /= np.linalg.norm(X_rp, axis=1, keepdims=True)
        Q_rp /= np.linalg.norm(Q_rp, axis=1, keepdims=True)
        
    dists_rp = euclidean_distances(Q_rp, X_rp)[0]
    rp_indices = np.argsort(dists_rp)[:10]
    
    rp_recall = len(set(gt_indices).intersection(rp_indices)) / 10
    
    # Witness Polar
    D = np.random.choice([-1, 1], size=d)
    H = hadamard(d)
    X_polar = (X * D) @ H
    Q_polar = (Q * D) @ H
    
    X_wp = X_polar[:, :m]
    Q_wp = Q_polar[:, :m]
    
    if normalize_projection:
        X_wp /= np.linalg.norm(X_wp, axis=1, keepdims=True)
        Q_wp /= np.linalg.norm(Q_wp, axis=1, keepdims=True)
        
    dists_wp = euclidean_distances(Q_wp, X_wp)[0]
    wp_indices = np.argsort(dists_wp)[:10]
    
    wp_recall = len(set(gt_indices).intersection(wp_indices)) / 10
    
    return rp_recall, wp_recall

print(f"{'N':<6} {'d':<6} {'m':<6} {'Norm?':<6} | {'RP Recall':<10} {'WP Recall':<10}")
print("-" * 60)

# Original
print(f"{2000:<6} {1024:<6} {128:<6} {'No':<6} | {run_experiment(2000, 1024, 128)[0]:.2f}       {run_experiment(2000, 1024, 128)[1]:.2f}")

# Normalize?
print(f"{2000:<6} {1024:<6} {128:<6} {'Yes':<6} | {run_experiment(2000, 1024, 128, True)[0]:.2f}       {run_experiment(2000, 1024, 128, True)[1]:.2f}")

# Smaller N
print(f"{100:<6} {1024:<6} {128:<6} {'No':<6} | {run_experiment(100, 1024, 128)[0]:.2f}       {run_experiment(100, 1024, 128)[1]:.2f}")

# Larger m
print(f"{2000:<6} {1024:<6} {512:<6} {'No':<6} | {run_experiment(2000, 1024, 512)[0]:.2f}       {run_experiment(2000, 1024, 512)[1]:.2f}")

# Larger m + Normalize
print(f"{2000:<6} {1024:<6} {512:<6} {'Yes':<6} | {run_experiment(2000, 1024, 512, True)[0]:.2f}       {run_experiment(2000, 1024, 512, True)[1]:.2f}")
