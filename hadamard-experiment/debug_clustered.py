import numpy as np
from scipy.linalg import hadamard
from sklearn.metrics.pairwise import euclidean_distances

def run_clustered_experiment(N=2000, d=1024, m=128, n_clusters=20):
    np.random.seed(42)
    
    # Generate Cluster Centers
    centers = np.random.randn(n_clusters, d)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    
    # Generate Data around centers
    X = []
    labels = []
    for i in range(N):
        c_idx = np.random.randint(n_clusters)
        center = centers[c_idx]
        noise = np.random.randn(d) * 0.1 # Small noise
        point = center + noise
        point /= np.linalg.norm(point)
        X.append(point)
        labels.append(c_idx)
    X = np.array(X)
    
    # Query: Pick a point near a center
    q_c_idx = 0
    Q = centers[q_c_idx] + np.random.randn(d) * 0.1
    Q /= np.linalg.norm(Q)
    Q = Q.reshape(1, -1)
    
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
    X_polar = (X * D) @ H
    Q_polar = (Q * D) @ H
    
    X_wp = X_polar[:, :m]
    Q_wp = Q_polar[:, :m]
    
    dists_wp = euclidean_distances(Q_wp, X_wp)[0]
    wp_indices = np.argsort(dists_wp)[:10]
    
    wp_recall = len(set(gt_indices).intersection(wp_indices)) / 10
    
    return rp_recall, wp_recall

print(f"Clustered Data Recall:")
rp, wp = run_clustered_experiment()
print(f"RP: {rp:.2f}")
print(f"WP: {wp:.2f}")
