import numpy as np
from scipy.linalg import hadamard
from sklearn.metrics.pairwise import euclidean_distances

N = 2000
d = 1024
m = 128

np.random.seed(42)
X = np.random.randn(N, d)
X /= np.linalg.norm(X, axis=1, keepdims=True)
Q = np.random.randn(1, d)
Q /= np.linalg.norm(Q, axis=1, keepdims=True)

# Ground Truth
dists_gt = euclidean_distances(Q, X)[0]
gt_indices = np.argsort(dists_gt)[:10]
print(f"GT Indices: {gt_indices}")
print(f"GT Distances: {dists_gt[gt_indices]}")

# Random Projection
G = np.random.randn(d, m)
X_rp = X @ G
Q_rp = Q @ G
dists_rp = euclidean_distances(Q_rp, X_rp)[0]
rp_indices = np.argsort(dists_rp)[:10]
print(f"RP Indices: {rp_indices}")
print(f"RP Distances (top 10): {dists_rp[rp_indices]}")

intersection = len(set(gt_indices).intersection(rp_indices))
print(f"Intersection: {intersection}")

# Check correlation
correlation = np.corrcoef(dists_gt, dists_rp)[0, 1]
print(f"Correlation between distances: {correlation:.4f}")
