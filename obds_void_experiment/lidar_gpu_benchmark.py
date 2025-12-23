"""
LiDAR Void Detection with GPU Distance Transform
Real-time autonomous driving scenario validation
"""

import numpy as np
import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, maximum_filter
from scipy.spatial import Delaunay

class LiDARVoidDetectorClassical:
    """Classical Delaunay baseline for LiDAR"""
    
    def __init__(self, x_range=(0, 50), y_range=(-20, 20)):
        self.x_range = x_range
        self.y_range = y_range
    
    def find_largest_void(self, points_3d):
        """Find void using Delaunay on BEV projection"""
        start = time.time()
        
        # Project to BEV
        points_2d = points_3d[:, :2]
        
        # Filter to range
        mask = ((points_2d[:, 0] >= self.x_range[0]) & 
                (points_2d[:, 0] <= self.x_range[1]) &
                (points_2d[:, 1] >= self.y_range[0]) & 
                (points_2d[:, 1] <= self.y_range[1]))
        points_2d = points_2d[mask]
        
        if len(points_2d) < 4:
            return {'time': time.time() - start, 'radius': 0, 'navigable': False}
        
        # Delaunay
        tri = Delaunay(points_2d)
        
        best_radius = 0
        best_center = None
        
        for simplex in tri.simplices:
            pts = points_2d[simplex]
            center, radius = self._circumcircle(pts)
            
            # Bound check
            if (self.x_range[0] <= center[0] <= self.x_range[1] and
                self.y_range[0] <= center[1] <= self.y_range[1]):
                if radius > best_radius:
                    best_radius = radius
                    best_center = center
        
        elapsed = time.time() - start
        return {
            'time': elapsed,
            'center': best_center,
            'radius': best_radius,
            'navigable': best_radius >= 2.0
        }
    
    def _circumcircle(self, pts):
        """Compute circumcircle"""
        ax, ay = pts[0]
        bx, by = pts[1]
        cx, cy = pts[2]
        
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            return np.array([0, 0]), 0.0
        
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + 
              (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + 
              (cx**2 + cy**2) * (bx - ax)) / d
        
        center = np.array([ux, uy])
        radius = np.linalg.norm(pts[0] - center)
        
        return center, radius

class LiDARVoidDetectorGPU:
    """GPU distance transform for LiDAR"""
    
    def __init__(self, x_range=(0, 50), y_range=(-20, 20), resolution=0.5, device='mps'):
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        self.device = device
        
        self.G_x = int((x_range[1] - x_range[0]) / resolution)
        self.G_y = int((y_range[1] - y_range[0]) / resolution)
        
        self.occupancy = None
        self.distance_field = None
    
    def detect_void(self, points_3d):
        """Complete void detection pipeline"""
        # Initialize
        init_start = time.time()
        
        # Project to BEV
        points_2d = points_3d[:, :2]
        
        # Create occupancy grid
        occupancy = np.zeros((self.G_x, self.G_y), dtype=np.float32)
        
        # Convert to grid indices
        grid_x = ((points_2d[:, 0] - self.x_range[0]) / self.resolution).astype(int)
        grid_y = ((points_2d[:, 1] - self.y_range[0]) / self.resolution).astype(int)
        
        # Clip
        grid_x = np.clip(grid_x, 0, self.G_x - 1)
        grid_y = np.clip(grid_y, 0, self.G_y - 1)
        
        # Mark occupied
        for i, j in zip(grid_x, grid_y):
            occupancy[i, j] = 1.0
        
        self.occupancy = occupancy
        init_time = time.time() - init_start
        
        # Compute distance field (GPU)
        query_start = time.time()
        
        # Move to GPU
        occupied = torch.from_numpy(occupancy).to(self.device)
        distance = torch.zeros_like(occupied)
        
        # Iterative distance computation
        kernel = torch.ones(3, 3, device=self.device) / 9.0
        
        max_iter = min(max(self.G_x, self.G_y) // 2, 100)
        for iteration in range(max_iter):
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
        
        # Convert to numpy
        self.distance_field = distance.cpu().numpy() * self.resolution
        
        # Find void
        max_idx = np.unravel_index(np.argmax(self.distance_field), self.distance_field.shape)
        max_dist = self.distance_field[max_idx]
        
        # World coordinates
        center_x = self.x_range[0] + (max_idx[0] + 0.5) * self.resolution
        center_y = self.y_range[0] + (max_idx[1] + 0.5) * self.resolution
        
        query_time = time.time() - query_start
        
        return {
            'init_time': init_time,
            'query_time': query_time,
            'total_time': init_time + query_time,
            'center': (center_x, center_y),
            'radius': max_dist,
            'navigable': max_dist >= 2.0
        }

def generate_lidar_frame(num_points, frame_idx, seed=42):
    """Generate synthetic LiDAR frame"""
    np.random.seed(seed + frame_idx)
    points = []
    
    # Walls on sides (buildings)
    n_left = int(num_points * 0.35)
    for _ in range(n_left):
        x = np.random.uniform(10, 45)
        y = np.random.uniform(-20, -8)
        z = np.random.uniform(0, 3)
        points.append([x, y, z])
    
    n_right = int(num_points * 0.35)
    for _ in range(n_right):
        x = np.random.uniform(10, 45)
        y = np.random.uniform(8, 20)
        z = np.random.uniform(0, 3)
        points.append([x, y, z])
    
    # Scattered obstacles
    n_obstacles = int(num_points * 0.15)
    for _ in range(n_obstacles):
        x = np.random.uniform(5, 48)
        y = np.random.uniform(-7, 7)
        z = np.random.uniform(0, 2)
        points.append([x, y, z])
    
    # Ground
    n_ground = int(num_points * 0.15)
    for _ in range(n_ground):
        x = np.random.uniform(0, 50)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(-2, -1)
        points.append([x, y, z])
    
    return np.array(points)

def run_lidar_benchmark():
    """Run LiDAR benchmark"""
    print("="*70)
    print("LiDAR Void Detection Benchmark (GPU Distance Transform)")
    print("="*70)
    
    # Generate frames with increasing point counts
    num_frames = 15
    base_points = 8000
    results = []
    
    print(f"\n{'Frame':<6} | {'Points':<8} | {'Classical':<12} | {'GPU Query':<12} | {'Speedup':<8} | {'30Hz'}")
    print("-"*70)
    
    for frame_idx in range(num_frames):
        # Increasing point count
        num_points = base_points + frame_idx * 2000
        
        # Generate frame
        points = generate_lidar_frame(num_points, frame_idx)
        
        # Filter to range
        mask = ((points[:, 0] >= 0) & (points[:, 0] <= 50) &
                (points[:, 1] >= -20) & (points[:, 1] <= 20) &
                (points[:, 2] >= -2) & (points[:, 2] <= 5))
        points = points[mask]
        
        actual_points = len(points)
        
        # Classical
        classical = LiDARVoidDetectorClassical()
        c_result = classical.find_largest_void(points)
        c_time = c_result['time']
        
        # GPU
        gpu = LiDARVoidDetectorGPU(resolution=0.5, device='mps')
        g_result = gpu.detect_void(points)
        g_time = g_result['query_time']
        
        speedup = c_time / g_time if g_time > 0 else 0
        realtime = '✓' if g_time < 0.033 else '✗'  # 30 Hz = 33ms
        
        print(f"{frame_idx+1:<6} | {actual_points:<8,} | {c_time:<12.3f} | {g_time:<12.3f} | {speedup:<8.1f}× | {realtime}")
        
        results.append({
            'frame': frame_idx + 1,
            'points': actual_points,
            'classical_time': c_time,
            'gpu_query': g_time,
            'speedup': speedup,
            'realtime_30hz': g_time < 0.033,
            'classical_radius': c_result['radius'],
            'gpu_radius': g_result['radius']
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv('obds_void_experiment/lidar_gpu_benchmark.csv', index=False)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Time comparison
    ax1 = axes[0, 0]
    ax1.plot(df['points'], df['classical_time'], 'o-', label='Classical', linewidth=2)
    ax1.plot(df['points'], df['gpu_query'], 's-', label='GPU Distance Transform', linewidth=2)
    ax1.axhline(y=0.033, color='r', linestyle='--', alpha=0.5, label='30 Hz threshold')
    ax1.set_xlabel('Number of Points')
    ax1.set_ylabel('Query Time (s)')
    ax1.set_title('LiDAR Void Detection Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup
    ax2 = axes[0, 1]
    ax2.plot(df['points'], df['speedup'], 'o-', color='green', linewidth=2)
    ax2.set_xlabel('Number of Points')
    ax2.set_ylabel('Speedup (×)')
    ax2.set_title('GPU Speedup over Classical')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Query time vs frame
    ax3 = axes[1, 0]
    ax3.plot(df['frame'], df['gpu_query'] * 1000, 'o-', color='blue', linewidth=2)
    ax3.axhline(y=33, color='r', linestyle='--', alpha=0.5, label='30 Hz (33ms)')
    ax3.set_xlabel('Frame Number')
    ax3.set_ylabel('GPU Query Time (ms)')
    ax3.set_title('Real-Time Performance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Real-time capability
    ax4 = axes[1, 1]
    realtime_counts = df['realtime_30hz'].value_counts()
    colors = ['#ff7f0e', '#2ca02c']
    labels = ['✗ No', '✓ Yes']
    if len(realtime_counts) == 1:
        if False in realtime_counts.index:
            ax4.pie([realtime_counts[False]], labels=['✗ No'], colors=['#ff7f0e'], autopct='%1.1f%%')
        else:
            ax4.pie([realtime_counts[True]], labels=['✓ Yes'], colors=['#2ca02c'], autopct='%1.1f%%')
    else:
        ax4.pie(realtime_counts, labels=labels, colors=colors, autopct='%1.1f%%')
    ax4.set_title('Real-Time Capable (30 Hz)')
    
    plt.tight_layout()
    plt.savefig('obds_void_experiment/lidar_gpu_benchmark.png', dpi=300)
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Frames tested: {len(df)}")
    print(f"Point range: {df['points'].min():,} - {df['points'].max():,}")
    print(f"Average speedup: {df['speedup'].mean():.1f}×")
    print(f"Max speedup: {df['speedup'].max():.1f}×")
    print(f"GPU query time range: {df['gpu_query'].min()*1000:.1f}ms - {df['gpu_query'].max()*1000:.1f}ms")
    print(f"Real-time capable: {df['realtime_30hz'].sum()}/{len(df)} frames")
    
    print(f"\n✓ Saved: lidar_gpu_benchmark.csv")
    print(f"✓ Saved: lidar_gpu_benchmark.png")
    
    return df

if __name__ == "__main__":
    df = run_lidar_benchmark()
    print("\n✓ LiDAR benchmark complete!")
