import numpy as np
import time

def fwht(a):
    """
    Vectorized Fast Walsh-Hadamard Transform.
    Recursive decomposition: H_2n = [[H_n, H_n], [H_n, -H_n]]
    """
    a = np.asarray(a)
    h = 1
    while h < a.shape[-1]:
        # Reshape to (..., N/2h, 2, h)
        a = a.reshape(a.shape[:-1] + (-1, 2, h))
        # Butterfly operations
        # x = a[..., 0, :]
        # y = a[..., 1, :]
        # new_x = x + y
        # new_y = x - y
        a = np.concatenate((a[..., 0, :] + a[..., 1, :], a[..., 0, :] - a[..., 1, :]), axis=-1)
        h *= 2
    return a.reshape(a.shape[:-1] + (-1,))

def run_benchmark():
    print(f"--- Experiment 5: Encoding Time Benchmark ---")
    N = 10000
    d = 1024
    m = 128
    
    print(f"Encoding {N} vectors of dimension {d} -> {m}")
    
    # Pre-allocate data
    X = np.random.randn(N, d).astype(np.float32)
    G = np.random.randn(d, m).astype(np.float32)
    D = np.random.choice([-1.0, 1.0], size=d).astype(np.float32)
    
    times_rp = []
    times_wp = []
    
    # Warmup
    _ = X @ G
    _ = fwht(X * D)
    
    iterations = 20
    print(f"Running {iterations} iterations...")
    
    for i in range(iterations):
        # Random Projection
        start = time.perf_counter()
        B_rp = X @ G
        times_rp.append(time.perf_counter() - start)
        
        # Witness-Polar
        start = time.perf_counter()
        # 1. Diagonal Flip (O(N*d))
        X_flipped = X * D
        # 2. FWHT (O(N*d*log d))
        X_trans = fwht(X_flipped)
        # 3. Select m columns (O(N*m) copy)
        B_wp = X_trans[:, :m]
        times_wp.append(time.perf_counter() - start)
        
    mean_rp = np.mean(times_rp) * 1000
    std_rp = np.std(times_rp) * 1000
    
    mean_wp = np.mean(times_wp) * 1000
    std_wp = np.std(times_wp) * 1000
    
    print(f"Random Projection: {mean_rp:.2f} ± {std_rp:.2f} ms")
    print(f"Witness-Polar:     {mean_wp:.2f} ± {std_wp:.2f} ms")
    
    speedup = mean_rp / mean_wp
    print(f"Speedup: {speedup:.2f}x")

if __name__ == "__main__":
    run_benchmark()
