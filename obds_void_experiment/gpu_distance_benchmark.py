"""
GPU-Accelerated Distance Transform Void Detection Benchmark
Compare classical Delaunay vs GPU distance transform at scale
"""

import numpy as np
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.ndimage import distance_transform_edt

class ClassicalVoidDetector:
    """Classical Delaunay-based void detection"""
    
    def __init__(self, points):
        self.points = points
    
    def find_largest_void(self):
        """Find largest void using Delaunay triangulation"""
        start = time.time()
        
        tri = Delaunay(self.points)
        
        best_radius = 0
        best_center = None
        
        for simplex in tri.simplices:
            pts = self.points[simplex]
            center, radius = self._circumcircle(pts)
            
            # Bound check
            if 0 <= center[0] <= 1 and 0 <= center[1] <= 1:
                if radius > best_radius:
                    best_radius = radius
                    best_center = center
        
        elapsed = time.time() - start
        return {'center': best_center, 'radius': best_radius, 'time': elapsed}
    
    def _circumcircle(self, pts):
        """Compute circumcircle of triangle"""
        ax, ay = pts[0]
        bx, by = pts[1]
        cx, cy = pts[2]
        
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            return np.array([0.5, 0.5]), 0.0
        
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
        
        center = np.array([ux, uy])
        radius = np.linalg.norm(pts[0] - center)
        
        return center, radius

class GPUDistanceTransformDetector:
    """GPU-accelerated distance transform void detection"""
    
    def __init__(self, grid_resolution=128, device='mps'):
        self.G = grid_resolution
        self.device = device
        self.occupancy = None
        self.distance_field = None
    
    def initialize_from_points(self, points):
        """Create occupancy grid"""
        start = time.time()
        
        # Create grid
        occupancy = np.zeros((self.G, self.G), dtype=np.float32)
        
        # Convert points to grid indices
        grid_x = (points[:, 0] * self.G).astype(int)
        grid_y = (points[:, 1] * self.G).astype(int)
        
        grid_x = np.clip(grid_x, 0, self.G - 1)
        grid_y = np.clip(grid_y, 0, self.G - 1)
        
        # Mark occupied
        for i, j in zip(grid_x, grid_y):
            occupancy[i, j] = 1.0
        
        self.occupancy = occupancy
        
        elapsed = time.time() - start
        return elapsed
    
    def compute_distance_field_gpu(self):
        """Compute distance field on GPU using iterative dilation"""
        start = time.time()
        
        # Move to GPU
        occupied = torch.from_numpy(self.occupancy).to(self.device)
        distance = torch.zeros_like(occupied)
        
        # Iterative distance computation (fast on GPU)
        kernel = torch.ones(3, 3, device=self.device) / 9.0
        
        for iteration in range(self.G // 2):  # Max possible distance
            # Dilate
            dilated = torch.nn.functional.conv2d(
                occupied.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=1
            ).squeeze()
            
            # Mark newly occupied cells
            newly_occupied = ((dilated > 0) & (occupied == 0)).float()
            distance += newly_occupied * (iteration + 1)
            
            # Update
            occupied = (dilated > 0).float()
            
            if newly_occupied.sum() == 0:
                break
        
        # Convert to numpy
        self.distance_field = distance.cpu().numpy()
        
        elapsed = time.time() - start
        return elapsed
    
    def compute_distance_field_cpu(self):
        """Compute distance field on CPU (scipy)"""
        start = time.time()
        
        free_space = (self.occupancy == 0)
        self.distance_field = distance_transform_edt(free_space)
        
        elapsed = time.time() - start
        return elapsed
    
    def find_largest_void(self):
        """Find largest void from distance field"""
        start = time.time()
        
        if self.distance_field is None:
            return None
        
        # Find maximum
        max_idx = np.unravel_index(np.argmax(self.distance_field), self.distance_field.shape)
        max_dist = self.distance_field[max_idx]
        
        # Convert to world coordinates
        center = np.array([max_idx[1], max_idx[0]]) / self.G
        radius = max_dist / self.G
        
        elapsed = time.time() - start
        return {'center': center, 'radius': radius, 'time': elapsed}

def generate_points(N, seed=42):
    """Generate random points avoiding center"""
    np.random.seed(seed)
    points = []
    
    while len(points) < N:
        p = np.random.rand(2)
        # Avoid center circle
        if np.linalg.norm(p - 0.5) > 0.2:
            points.append(p)
    
    return np.array(points[:N])

def run_benchmark():
    """Run scaling benchmark"""
    print("="*60)
    print("GPU Distance Transform Benchmark")
    print("="*60)
    
    N_values = [10000, 50000, 100000, 500000, 1000000]
    results = []
    
    print(f"\n{'N':<10} | {'Classical':<12} | {'GPU Init':<10} | {'GPU Query':<12} | {'Speedup':<8}")
    print("-"*60)
    
    for N in N_values:
        print(f"\nN = {N:,}")
        
        # Generate points
        points = generate_points(N)
        
        # Classical
        classical = ClassicalVoidDetector(points)
        c_result = classical.find_largest_void()
        c_time = c_result['time']
        
        # GPU Distance Transform
        gpu_detector = GPUDistanceTransformDetector(grid_resolution=128, device='mps')
        
        # Init
        init_time = gpu_detector.initialize_from_points(points)
        
        # Query (GPU)
        query_time_gpu = gpu_detector.compute_distance_field_gpu()
        gpu_result = gpu_detector.find_largest_void()
        query_time_find = gpu_result['time']
        
        total_query = query_time_gpu + query_time_find
        
        speedup = c_time / total_query
        
        print(f"{N:<10} | {c_time:<12.4f} | {init_time:<10.4f} | {total_query:<12.4f} | {speedup:<8.1f}×")
        
        results.append({
            'N': N,
            'classical_time': c_time,
            'gpu_init': init_time,
            'gpu_query': total_query,
            'speedup': speedup
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('obds_void_experiment/gpu_distance_benchmark.csv', index=False)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Time vs N
    ax1 = axes[0]
    ax1.plot(df['N'], df['classical_time'], 'o-', label='Classical (CPU)', linewidth=2)
    ax1.plot(df['N'], df['gpu_query'], 's-', label='GPU Distance Transform', linewidth=2)
    ax1.set_xlabel('Number of Points (N)')
    ax1.set_ylabel('Query Time (s)')
    ax1.set_title('Void Detection Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # Plot 2: Speedup vs N
    ax2 = axes[1]
    ax2.plot(df['N'], df['speedup'], 'o-', color='green', linewidth=2)
    ax2.set_xlabel('Number of Points (N)')
    ax2.set_ylabel('Speedup (×)')
    ax2.set_title('GPU Speedup over Classical')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='No speedup')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('obds_void_experiment/gpu_distance_benchmark.png', dpi=300)
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print(f"Max speedup: {df['speedup'].max():.1f}× at N={df.loc[df['speedup'].idxmax(), 'N']:,}")
    print(f"GPU query time (N=1M): {df.loc[df['N']==1000000, 'gpu_query'].values[0]:.3f}s")
    print(f"\n✓ Saved: gpu_distance_benchmark.csv")
    print(f"✓ Saved: gpu_distance_benchmark.png")
    
    return df

if __name__ == "__main__":
    df = run_benchmark()
    print("\n✓ Benchmark complete!")
