"""
Distance Field Void Detection - Robust Alternative
Uses distance transform instead of Gray-Scott for reliable void detection
"""

import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, maximum_filter

def generate_test_scene(num_points=15000, seed=42):
    """Generate corridor scene with clear void"""
    np.random.seed(seed)
    points = []
    
    # Left wall
    n_left = int(num_points * 0.4)
    for _ in range(n_left):
        x = np.random.uniform(5, 45)
        y = np.random.uniform(-20, -6)
        z = np.random.uniform(0, 3)
        points.append([x, y, z])
    
    # Right wall
    n_right = int(num_points * 0.4)
    for _ in range(n_right):
        x = np.random.uniform(5, 45)
        y = np.random.uniform(6, 20)
        z = np.random.uniform(0, 3)
        points.append([x, y, z])
    
    # Ground
    n_ground = int(num_points * 0.2)
    for _ in range(n_ground):
        x = np.random.uniform(0, 50)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(-2, -1)
        points.append([x, y, z])
    
    return np.array(points)

class DistanceFieldVoidDetector:
    """Void detection using distance transform"""
    
    def __init__(self, x_range=(0, 50), y_range=(-20, 20), resolution=0.3):
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        
        self.G_x = int((x_range[1] - x_range[0]) / resolution)
        self.G_y = int((y_range[1] - y_range[0]) / resolution)
        
        self.occupancy_grid = None
        self.distance_field = None
        
        print(f"Initialized Distance Field Detector: {self.G_x}×{self.G_y} grid")
        print(f"  Physical size: {x_range[1]-x_range[0]}m × {y_range[1]-y_range[0]}m")
        print(f"  Resolution: {resolution}m per cell")
    
    def initialize_from_points(self, points_3d):
        """Create occupancy grid from points"""
        # Initialize empty grid
        self.occupancy_grid = np.zeros((self.G_x, self.G_y), dtype=bool)
        
        # Project to 2D
        points_2d = points_3d[:, :2]
        
        # Convert to grid indices
        grid_x = ((points_2d[:, 0] - self.x_range[0]) / self.resolution).astype(int)
        grid_y = ((points_2d[:, 1] - self.y_range[0]) / self.resolution).astype(int)
        
        # Clip
        grid_x = np.clip(grid_x, 0, self.G_x - 1)
        grid_y = np.clip(grid_y, 0, self.G_y - 1)
        
        # Mark occupied cells
        occupied_cells = set(zip(grid_x, grid_y))
        for i, j in occupied_cells:
            # Mark cell and neighbors
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < self.G_x and 0 <= jj < self.G_y:
                        self.occupancy_grid[ii, jj] = True
        
        occupancy_rate = self.occupancy_grid.sum() / (self.G_x * self.G_y)
        return {'occupancy_rate': occupancy_rate}
    
    def compute_distance_field_gpu(self, device='mps'):
        """Compute distance field using GPU"""
        start = time.time()
        
        # Convert to torch tensor
        occupied = torch.from_numpy(self.occupancy_grid.astype(np.float32)).to(device)
        
        # Create distance field (simple approach: iterative dilation)
        distance = torch.zeros_like(occupied)
        free_space = (occupied == 0).float()
        
        # Iterative distance computation
        max_dist = 0
        for iteration in range(100):  # Max distance in cells
            if free_space.sum() == 0:
                break
            
            # Dilate occupied region
            kernel = torch.ones(3, 3, device=device) / 9.0
            dilated = torch.nn.functional.conv2d(
                occupied.unsqueeze(0).unsqueeze(0),
                kernel.unsqueeze(0).unsqueeze(0),
                padding=1
            ).squeeze()
            
            # Cells that just became occupied get current distance
            newly_occupied = ((dilated > 0) & (occupied == 0)).float()
            distance += newly_occupied * (iteration + 1)
            
            # Update occupied
            occupied = (dilated > 0).float()
            max_dist = iteration + 1
            
            if newly_occupied.sum() == 0:
                break
        
        # Convert back to numpy
        self.distance_field = distance.cpu().numpy() * self.resolution
        
        elapsed = time.time() - start
        return elapsed
    
    def compute_distance_field_cpu(self):
        """Compute distance field using CPU (scipy)"""
        start = time.time()
        
        # Invert: True where free, False where occupied
        free_space = ~self.occupancy_grid
        
        # Compute Euclidean distance transform
        distance_cells = distance_transform_edt(free_space)
        
        # Convert to meters
        self.distance_field = distance_cells * self.resolution
        
        elapsed = time.time() - start
        return elapsed
    
    def find_largest_void(self, min_radius_m=2.0):
        """Find largest void"""
        if self.distance_field is None:
            return {'status': 'no_field'}
        
        # Find local maxima
        max_filtered = maximum_filter(self.distance_field, size=5)
        local_maxima = (self.distance_field == max_filtered) & (self.distance_field > min_radius_m)
        
        if not local_maxima.any():
            return {
                'status': 'no_void',
                'max_distance': self.distance_field.max(),
                'navigable': False
            }
        
        # Find the largest
        peak_indices = np.argwhere(local_maxima)
        peak_values = self.distance_field[local_maxima]
        best_idx = np.argmax(peak_values)
        best_peak = peak_indices[best_idx]
        best_radius = peak_values[best_idx]
        
        # Convert to world coordinates
        center_x = self.x_range[0] + (best_peak[0] + 0.5) * self.resolution
        center_y = self.y_range[0] + (best_peak[1] + 0.5) * self.resolution
        
        return {
            'status': 'void_found',
            'center_world': (center_x, center_y),
            'radius_m': best_radius,
            'navigable': best_radius >= min_radius_m
        }

def run_demo():
    """Run distance field demo"""
    print("="*60)
    print("Distance Field Void Detection Demo")
    print("="*60)
    
    # Generate scene
    points = generate_test_scene(num_points=15000, seed=42)
    
    # Filter
    mask = ((points[:, 0] >= 0) & (points[:, 0] <= 50) &
            (points[:, 1] >= -20) & (points[:, 1] <= 20) &
            (points[:, 2] >= -2) & (points[:, 2] <= 5))
    points = points[mask]
    
    print(f"\nGenerated {len(points):,} points")
    print("  Left wall: y < -6m")
    print("  Right wall: y > 6m")
    print("  Expected void: center corridor (~6m radius)")
    
    # Initialize detector
    detector = DistanceFieldVoidDetector(
        x_range=(0, 50),
        y_range=(-20, 20),
        resolution=0.3
    )
    
    print("\n" + "="*60)
    print("Processing")
    print("="*60)
    
    # Initialize
    init_start = time.time()
    init_info = detector.initialize_from_points(points)
    init_time = time.time() - init_start
    
    print(f"Init: {init_time:.3f}s")
    print(f"  Occupancy: {init_info['occupancy_rate']*100:.1f}%")
    
    # Compute distance field (CPU - more reliable)
    print("\nComputing distance field (CPU)...")
    query_time = detector.compute_distance_field_cpu()
    
    print(f"Query: {query_time:.3f}s")
    print(f"  Max distance: {detector.distance_field.max():.2f}m")
    print(f"  Mean distance: {detector.distance_field.mean():.2f}m")
    
    # Find void
    result = detector.find_largest_void(min_radius_m=2.0)
    
    print("\n" + "="*60)
    print("Results")
    print("="*60)
    
    if result['status'] == 'void_found':
        print(f"✓ Void detected!")
        print(f"  Center: ({result['center_world'][0]:.1f}m, {result['center_world'][1]:.1f}m)")
        print(f"  Radius: {result['radius_m']:.2f}m")
        print(f"  Navigable: {'✓' if result['navigable'] else '✗'}")
    else:
        print(f"✗ No void found")
        print(f"  Max distance: {result.get('max_distance', 0):.2f}m")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Points
    ax1 = axes[0]
    ax1.scatter(points[:, 0], points[:, 1], s=2, c='black', alpha=0.5)
    ax1.set_xlabel('X (forward) [m]')
    ax1.set_ylabel('Y (lateral) [m]')
    ax1.set_title('Point Cloud (BEV)')
    ax1.set_xlim(0, 50)
    ax1.set_ylim(-20, 20)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Occupancy grid
    ax2 = axes[1]
    extent = [detector.y_range[0], detector.y_range[1], 
              detector.x_range[0], detector.x_range[1]]
    ax2.imshow(detector.occupancy_grid, cmap='gray', extent=extent, 
               origin='lower', aspect='auto')
    ax2.set_xlabel('Y (lateral) [m]')
    ax2.set_ylabel('X (forward) [m]')
    ax2.set_title('Occupancy Grid')
    
    # Plot 3: Distance field
    ax3 = axes[2]
    im = ax3.imshow(detector.distance_field, cmap='hot', extent=extent,
                    origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax3, label='Distance to obstacle [m]')
    ax3.set_xlabel('Y (lateral) [m]')
    ax3.set_ylabel('X (forward) [m]')
    ax3.set_title('Distance Field (Void Detection)')
    
    if result['status'] == 'void_found':
        circle = plt.Circle(
            (result['center_world'][1], result['center_world'][0]),
            result['radius_m'],
            fill=False, color='cyan', linewidth=3,
            label=f'Void (r={result["radius_m"]:.1f}m)'
        )
        ax3.add_patch(circle)
        ax3.plot(result['center_world'][1], result['center_world'][0],
                'c*', markersize=20)
        ax3.legend()
    
    plt.tight_layout()
    plt.savefig('obds_void_experiment/distance_field_demo.png', dpi=300)
    print("\n✓ Visualization saved to: obds_void_experiment/distance_field_demo.png")
    
    return result

if __name__ == "__main__":
    result = run_demo()
    print("\n✓ Demo complete!")
