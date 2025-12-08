import numpy as np
from curvature_measurement import measure_curvature

def random_matrix_lowrank(n_samples, dim, rank):
    print(f"Generating {n_samples} samples of Rank {rank} in {dim}D...")
    # U * V
    U = np.random.randn(n_samples, rank)
    V = np.random.randn(rank, dim)
    return np.dot(U, V)

def run_random_control():
    np.random.seed(42)
    n = 10000
    dim = 512
    
    print("\nExperiment 1: Full Rank Random Noise")
    # 1. Random Gaussian
    random_512d = np.random.randn(n, dim)
    # Normalize (handled by measure_curvature but let's be explicit for clarity)
    random_norm = random_512d / np.linalg.norm(random_512d, axis=1, keepdims=True)
    
    res_rand = measure_curvature(random_norm, n_triangles=2000, verbose=False)
    print(f"Random normalized: K = {res_rand['K_mean']:.4f} ± {res_rand['K_std']:.4f}")
    
    print("\nExperiment 2: Low-Rank Random Structure")
    # 2. Low Rank (Rank 50)
    random_lowrank = random_matrix_lowrank(n, dim, rank=50)
    # Normalize
    random_lowrank_norm = random_lowrank / np.linalg.norm(random_lowrank, axis=1, keepdims=True)
    
    res_low = measure_curvature(random_lowrank_norm, n_triangles=2000, verbose=False)
    print(f"Low-rank random: K = {res_low['K_mean']:.4f} ± {res_low['K_std']:.4f}")
    
    print("\n" + "="*50)
    print("CONCLUSION")
    print("="*50)
    if abs(res_rand['K_mean'] - 1.0) < 0.01:
        print("Hypothesis Confirmed: Normalization forces K=1 even for NOISE.")
        print("Curvature measures the 'Container' (Sphere), not the 'Content'.")
    else:
        print("Hypothesis Rejected: Noise has different curvature?")

if __name__ == "__main__":
    run_random_control()
