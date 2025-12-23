"""
Extreme Scale Void Detection: 10M Points
Ultimate stress test for GPU distance transform
"""

import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

class GPUDistanceTransformDetector:
    """GPU-accelerated distance transform"""
    
    def __init__(self, grid_resolution=256, device='mps'):
        self.G = grid_resolution
        self.device = device
    
    def detect_void(self, points):
        """Complete pipeline"""
        # Initialize
        init_start = time.time()
        
        occupancy = np.zeros((self.G, self.G), dtype=np.float32)
        
        # Convert to grid
        grid_x = (points[:, 0] * self.G).astype(int)
        grid_y = (points[:, 1] * self.G).astype(int)
        
        grid_x = np.clip(grid_x, 0, self.G - 1)
        grid_y = np.clip(grid_y, 0, self.G - 1)
        
        # Mark occupied (use unique cells for efficiency)
        unique_cells = np.unique(np.column_stack([grid_x, grid_y]), axis=0)
        for i, j in unique_cells:
            occupancy[i, j] = 1.0
        
        init_time = time.time() - init_start
        
        # GPU distance transform
        query_start = time.time()
        
        occupied = torch.from_numpy(occupancy).to(self.device)
        distance = torch.zeros_like(occupied)
        
        kernel = torch.ones(3, 3, device=self.device) / 9.0
        
        for iteration in range(self.G // 2):
            dilated = torch.nn.functional.conv2d(
                occupied.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=1
            ).squeeze()
            
            newly_occupied = ((dilated > 0) & (occupied == 0)).float()
            distance += newly_occupied * (iteration + 1)
            
            occupied = (dilated > 0).float()
            
            if newly_occupied.sum() == 0:
                break
        
        distance_field = distance.cpu().numpy()
        
        # Find void
        max_idx = np.unravel_index(np.argmax(distance_field), distance_field.shape)
        max_dist = distance_field[max_idx]
        
        center = np.array([max_idx[1], max_idx[0]]) / self.G
        radius = max_dist / self.G
        
        query_time = time.time() - query_start
        
        return {
            'init_time': init_time,
            'query_time': query_time,
            'total_time': init_time + query_time,
            'center': center,
            'radius': radius
        }

def generate_points(N, seed=42):
    """Generate points avoiding center"""
    np.random.seed(seed)
    points = []
    
    batch_size = 100000
    while len(points) < N:
        batch = np.random.rand(min(batch_size, N - len(points)), 2)
        # Filter: avoid center circle
        mask = np.linalg.norm(batch - 0.5, axis=1) > 0.2
        points.extend(batch[mask])
    
    return np.array(points[:N])

def run_extreme_benchmark():
    """Run 10M point benchmark"""
    print("="*70)
    print("EXTREME SCALE VOID DETECTION: 10 MILLION POINTS")
    print("="*70)
    
    N_values = [100000, 500000, 1000000, 2000000, 5000000, 10000000]
    results = []
    
    print(f"\n{'N':<12} | {'Init (s)':<10} | {'Query (s)':<10} | {'Total (s)':<10} | {'Radius'}")
    print("-"*70)
    
    for N in N_values:
        print(f"\nGenerating {N:,} points...")
        points = generate_points(N)
        
        print(f"Running GPU distance transform...")
        detector = GPUDistanceTransformDetector(grid_resolution=256, device='mps')
        result = detector.detect_void(points)
        
        print(f"{N:<12,} | {result['init_time']:<10.3f} | {result['query_time']:<10.3f} | "
              f"{result['total_time']:<10.3f} | {result['radius']:.4f}")
        
        results.append({
            'N': N,
            'init_time': result['init_time'],
            'query_time': result['query_time'],
            'total_time': result['total_time'],
            'radius': result['radius']
        })
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    N_arr = np.array([r['N'] for r in results])
    query_arr = np.array([r['query_time'] for r in results])
    total_arr = np.array([r['total_time'] for r in results])
    
    # Plot 1: Query time vs N
    ax1 = axes[0]
    ax1.plot(N_arr, query_arr, 'o-', linewidth=2, markersize=8, label='Query Time')
    ax1.axhline(y=query_arr[-1], color='r', linestyle='--', alpha=0.5, 
                label=f'10M: {query_arr[-1]:.3f}s')
    ax1.set_xlabel('Number of Points (N)')
    ax1.set_ylabel('Query Time (s)')
    ax1.set_title('GPU Distance Transform Query Time')
    ax1.set_xscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Total time breakdown
    ax2 = axes[1]
    init_arr = np.array([r['init_time'] for r in results])
    
    x = np.arange(len(N_arr))
    width = 0.35
    
    ax2.bar(x - width/2, init_arr, width, label='Init', alpha=0.8)
    ax2.bar(x + width/2, query_arr, width, label='Query', alpha=0.8)
    
    ax2.set_xlabel('Scale')
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Time Breakdown by Scale')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{n/1e6:.1f}M' for n in N_arr])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('obds_void_experiment/extreme_scale_benchmark.png', dpi=300)
    
    # Summary
    print("\n" + "="*70)
    print("EXTREME SCALE RESULTS")
    print("="*70)
    print(f"✓ Successfully processed 10,000,000 points!")
    print(f"\n10M Point Performance:")
    print(f"  Init time:  {results[-1]['init_time']:.3f}s")
    print(f"  Query time: {results[-1]['query_time']:.3f}s")
    print(f"  Total time: {results[-1]['total_time']:.3f}s")
    print(f"  Void radius: {results[-1]['radius']:.4f}")
    
    # Query time scaling
    print(f"\nQuery Time Scaling:")
    for i, r in enumerate(results):
        if i > 0:
            scale_factor = r['N'] / results[0]['N']
            time_factor = r['query_time'] / results[0]['query_time']
            print(f"  {r['N']:>10,} points: {r['query_time']:.3f}s "
                  f"({scale_factor:.0f}× points, {time_factor:.2f}× time)")
    
    print(f"\n✓ Saved: extreme_scale_benchmark.png")
    
    return results

if __name__ == "__main__":
    results = run_extreme_benchmark()
    print("\n✓ Extreme scale benchmark complete!")
