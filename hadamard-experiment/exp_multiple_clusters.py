import numpy as np
from scipy.linalg import hadamard
from sklearn.metrics.pairwise import euclidean_distances

def run_multiple_clusters(N=2000, d=1024, m=128, k_clusters=10, points_per_cluster=10):
    print(f"--- Experiment 2: Multiple Clusters ---")
    print(f"Planting {k_clusters} clusters of {points_per_cluster} points each.")
    
    np.random.seed(42)
    X = np.random.randn(N, d)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    
    # Plant Clusters
    cluster_centers = []
    cluster_indices = [] # List of sets
    
    current_idx = 0
    for k in range(k_clusters):
        center = np.random.randn(d)
        center /= np.linalg.norm(center)
        cluster_centers.append(center)
        
        indices = []
        for i in range(points_per_cluster):
            point = center + np.random.randn(d) * 0.05 # Tight clusters
            point /= np.linalg.norm(point)
            X[current_idx] = point
            indices.append(current_idx)
            current_idx += 1
        cluster_indices.append(set(indices))
            
    # Test each cluster
    rp_precisions = []
    wp_precisions = []
    
    # Pre-compute projections
    G = np.random.randn(d, m)
    X_rp = X @ G
    
    D = np.random.choice([-1, 1], size=d)
    H = hadamard(d)
    X_wp = ((X * D) @ H)[:, :m]
    
    for k in range(k_clusters):
        # Query near center k
        Q = (cluster_centers[k] + np.random.randn(d) * 0.05).reshape(1, -1)
        Q /= np.linalg.norm(Q, axis=1, keepdims=True)
        
        # Ground Truth (should be the cluster indices)
        target_indices = cluster_indices[k]
        
        # RP
        Q_rp = Q @ G
        dists_rp = euclidean_distances(Q_rp, X_rp)[0]
        rp_top = set(np.argsort(dists_rp)[:points_per_cluster])
        rp_prec = len(target_indices.intersection(rp_top)) / points_per_cluster
        rp_precisions.append(rp_prec)
        
        # WP
        Q_wp = ((Q * D) @ H)[:, :m]
        dists_wp = euclidean_distances(Q_wp, X_wp)[0]
        wp_top = set(np.argsort(dists_wp)[:points_per_cluster])
        wp_prec = len(target_indices.intersection(wp_top)) / points_per_cluster
        wp_precisions.append(wp_prec)
        
    print(f"Average Precision over {k_clusters} clusters:")
    print(f"Random Projection: {np.mean(rp_precisions)*100:.1f}%")
    print(f"Witness Polar:     {np.mean(wp_precisions)*100:.1f}%")

if __name__ == "__main__":
    run_multiple_clusters()
