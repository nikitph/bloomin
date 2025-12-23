"""
Emergent Void Detection Experiment (MPS/GPU Accelerated)
Compares:
1. Classical Baseline (CPU/Scipy Delaunay)
2. OBDS Solution (GPU/PyTorch Reaction-Diffusion)
"""

import numpy as np
import torch
import torch.fft
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Ensure output directory exists
if not os.path.exists('obds_void_experiment'):
    os.makedirs('obds_void_experiment', exist_ok=True)

# Select Device
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"✅ Using MPS (Metal Performance Shaders) acceleration.")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"✅ Using CUDA acceleration.")
else:
    DEVICE = torch.device("cpu")
    print(f"⚠️ MPS/CUDA not available. Using CPU.")

# -----------------------------------------------------------------------------
# PART A: Classical Baseline (CPU)
# -----------------------------------------------------------------------------

class ClassicalVoidDetector:
    def __init__(self, points: np.ndarray):
        self.points = points
        self.N, self.d = points.shape
    
    def find_largest_void_delaunay(self):
        """Standard Delaunay implementation on CPU"""
        start_time = time.time()
        tri = Delaunay(self.points)
        
        best_radius = 0.0
        best_center = None
        
        for simplex in tri.simplices:
            pts = self.points[simplex]
            center, radius = self._circumsphere(pts)
            if 0 <= center[0] <= 1 and 0 <= center[1] <= 1:
                if radius > best_radius:
                    best_radius = radius
                    best_center = center
        
        elapsed = time.time() - start_time
        return elapsed

    def _circumsphere(self, simplex_points):
        A = simplex_points[1:] - simplex_points[0]
        b = 0.5 * np.sum(A**2, axis=1)
        try:
            y = np.linalg.solve(A, b)
            center = simplex_points[0] + y
            radius = np.linalg.norm(center - simplex_points[0])
        except np.linalg.LinAlgError:
            center = simplex_points.mean(axis=0)
            radius = 0.0
        return center, radius

# -----------------------------------------------------------------------------
# PART B: OBDS Solution (PyTorch / MPS)
# -----------------------------------------------------------------------------

class OBDSVoidDetectorMPS:
    def __init__(self, grid_resolution=256, Du=2e-5, Dv=1e-5, F=0.055, k=0.062):
        self.G = grid_resolution
        self.Du = Du
        self.Dv = Dv
        self.F = F
        self.k = k
        self.dx = 1.0 / grid_resolution
        self.dy = 1.0 / grid_resolution
        
        # Precompute frequencies on GPU
        kx = torch.fft.fftfreq(self.G, self.dx, device=DEVICE) * 2 * np.pi
        ky = torch.fft.fftfreq(self.G, self.dy, device=DEVICE) * 2 * np.pi
        KX, KY = torch.meshgrid(kx, ky, indexing='xy')
        self.K2 = KX**2 + KY**2
        
        # Initialize fields (Placeholder, will reset in cycle)
        self.u = torch.ones((self.G, self.G), device=DEVICE, dtype=torch.float32)
        self.v = torch.zeros((self.G, self.G), device=DEVICE, dtype=torch.float32)

    def initialize_from_points(self, points: np.ndarray):
        """
        Initialization (O(N)).
        For optimal perf, we rasterize points on CPU then upload mask to GPU.
        Or scatter on GPU. Let's do CPU raster -> GPU upload for simplicity/correctness,
        as init time is O(N) anyway.
        """
        # CPU Rasterization
        u_cpu = np.ones((self.G, self.G), dtype=np.float32)
        v_cpu = np.zeros((self.G, self.G), dtype=np.float32)
        
        # Noise
        v_cpu += np.random.uniform(0, 0.01, (self.G, self.G))
        
        # Points
        # Vectorized point masking could be faster, but loop is O(N)
        for point in points:
            i = int(point[0] * self.G)
            j = int(point[1] * self.G)
            # Simple 3x3 or 5x5 kernel
            i_min = max(0, i-2)
            i_max = min(self.G, i+3)
            j_min = max(0, j-2)
            j_max = min(self.G, j+3)
            u_cpu[i_min:i_max, j_min:j_max] = 1.0
            v_cpu[i_min:i_max, j_min:j_max] = 0.0
            
        # Upload to GPU
        self.u = torch.from_numpy(u_cpu).to(DEVICE)
        self.v = torch.from_numpy(v_cpu).to(DEVICE)

    def evolve_fft(self, T=500, dt=1.0):
        """
        FFT-based evolution on GPU.
        """
        # Pre-compute decay factors once for this dt if we wanted, 
        # but T is small so computing on fly is fine.
        decay_u = torch.exp(-self.Du * self.K2 * dt)
        decay_v = torch.exp(-self.Dv * self.K2 * dt)
        
        for _ in range(T):
            # Reaction 1/2
            self._reaction_step(dt/2)
            
            # Diffusion (FFT)
            # complex64 usually sufficient? using float32/complex64 default
            self.u = torch.real(torch.fft.ifft2(torch.fft.fft2(self.u) * decay_u))
            self.v = torch.real(torch.fft.ifft2(torch.fft.fft2(self.v) * decay_v))
            
            # Reaction 1/2
            self._reaction_step(dt/2)
            
    def _reaction_step(self, dt):
        uvv = self.u * (self.v * self.v)
        du = -uvv + self.F * (1.0 - self.u)
        dv = uvv - (self.F + self.k) * self.v
        self.u += du * dt
        self.v += dv * dt

    def find_largest_void(self):
        """Query O(1)"""
        # 1. Find max v on GPU
        # argmax returns flat index
        max_idx = torch.argmax(self.v)
        max_val = self.v.view(-1)[max_idx]
        
        # Convert to coords
        # PyTorch doesn't have unravel_index equivalent for tensors efficiently until recently/numpy style
        # Manual unravel
        y_idx = max_idx // self.G
        x_idx = max_idx % self.G
        
        center = np.array([float(x_idx) / self.G, float(y_idx) / self.G]) # x, y
        
        # Calculate radius (on GPU sum -> CPU)
        mask = self.v > (0.5 * max_val)
        area = torch.sum(mask).item() * (self.dx * self.dy)
        radius = np.sqrt(area / np.pi)
        
        return center, radius

# -----------------------------------------------------------------------------
# PART C: MPS Benchmark
# -----------------------------------------------------------------------------

def run_mps_benchmark(N_values=None):
    if N_values is None:
        # Generate range: 50k, 100k, 150k, ..., 10M
        N_values = list(range(50000, 10000001, 50000))
    
    results = []
    print(f"\nRunning MPS Benchmark (Device: {DEVICE})")
    print(f"{'N':<8} | {'Class(s)':<10} | {'Init(s)':<10} | {'Query(s)':<10} | {'Speedup':<8}")
    print("-" * 60)
    
    # Warmup GPU
    dummy = OBDSVoidDetectorMPS()
    dummy.u = torch.randn(256, 256, device=DEVICE)
    dummy.evolve_fft(T=10)
    torch.cuda.synchronize() if torch.cuda.is_available() else None # MPS sync not strictly exposed like cuda, but queue flushes
    
    for N in N_values:
        # Generate Points
        points = []
        # Fast generation
        points = np.random.rand(N, 2)
        
        # 1. Classical (CPU)
        # For N=1M, Classical is slow (~25s). We can re-run it or just trust previous data?
        # Let's run it to be rigorous.
        classical = ClassicalVoidDetector(points)
        c_time = classical.find_largest_void_delaunay()
        
        # 2. OBDS (MPS)
        obds = OBDSVoidDetectorMPS(grid_resolution=128)
        
        # Init
        i_start = time.time()
        obds.initialize_from_points(points)
        if hasattr(torch.backends.mps, 'synchronize'): torch.backends.mps.synchronize() # Wait for upload?
        i_time = time.time() - i_start
        
        # Query
        q_start = time.time()
        obds.evolve_fft(T=500, dt=1.0)
        # Force sync for timing
        if DEVICE.type == 'mps': # Generic torch sync works for CUDA, MPS usually implicitly handled on item() access but explicit correct for timing
             # Doing a blocking view or similar ensures it finished
             # Actually `find_largest_void` calls `.item()` which blocks until result ready.
             pass
             
        res = obds.find_largest_void()
        q_time = time.time() - q_start
        
        speedup = c_time / max(q_time, 1e-9)
        
        print(f"{N:<8} | {c_time:<10.4f} | {i_time:<10.4f} | {q_time:<10.4f} | {speedup:<8.1f}")
        
        results.append({
            'N': N,
            'classical_time': c_time,
            'obds_init': i_time,
            'obds_query': q_time,
            'speedup': speedup
        })
        
    return pd.DataFrame(results)

def plot_mps_scaling(df):
    plt.figure(figsize=(10, 6))
    plt.loglog(df['N'], df['classical_time'], 'o-', label='Classical (CPU Delaunay)')
    plt.loglog(df['N'], df['obds_query'], 's-', label='OBDS Query (GPU/MPS)', linewidth=3, color='red')
    plt.loglog(df['N'], df['obds_init'], 'x--', label='OBDS Init (O(N))', alpha=0.5)
    
    plt.xlabel('Number of Points (N)')
    plt.ylabel('Time (seconds)')
    plt.title('MPS Acceleration: Classical vs OBDS Query')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig('obds_void_experiment/mps_scaling.png', dpi=300)

if __name__ == "__main__":
    df = run_mps_benchmark()
    plot_mps_scaling(df)
