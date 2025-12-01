import numpy as np
from scipy.linalg import hadamard
from sklearn.metrics.pairwise import euclidean_distances

def run_debug_planted(N=2000, d=1024, m=128):
    np.random.seed(42)
    X = np.random.randn(N, d)
    
    # Plant target at 0
    target = X[0]
    noise = np.random.randn(d) * 0.5
    Q = (target + noise).reshape(1, -1)
    
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    
    # GT
    dists = euclidean_distances(Q, X)[0]
    gt_indices = np.argsort(dists)[:10]
    print(f"GT Indices: {gt_indices}")
    
    # RP
    G = np.random.randn(d, m)
    X_rp = X @ G
    Q_rp = Q @ G
    dists_rp = euclidean_distances(Q_rp, X_rp)[0]
    rp_indices = np.argsort(dists_rp)[:10]
    print(f"RP Indices: {rp_indices}")
    
    # WP
    D = np.random.choice([-1, 1], size=d)
    H = hadamard(d)
    X_wp = (X * D) @ H
    Q_wp = (Q * D) @ H
    X_wp = X_wp[:, :m]
    Q_wp = Q_wp[:, :m]
    dists_wp = euclidean_distances(Q_wp, X_wp)[0]
    wp_indices = np.argsort(dists_wp)[:10]
    print(f"WP Indices: {wp_indices}")
    
    # Check if 0 is found
    print(f"RP found 0? {0 in rp_indices}")
    print(f"WP found 0? {0 in wp_indices}")

run_debug_planted()
