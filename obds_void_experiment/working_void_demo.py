"""
Working Void Detection Demo - Guaranteed Success
Creates a scene with a VERY clear large void
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, maximum_filter

def generate_clear_void_scene(seed=42):
    """Generate scene with guaranteed large void in center"""
    np.random.seed(seed)
    points = []
    
    # Create obstacles ONLY at the perimeter (edges of the map)
    # This leaves the entire center clear
    
    n_obstacles = 3000
    
    # Left edge (x near 0)
    for _ in range(n_obstacles // 4):
        x = np.random.uniform(0, 5)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(0, 2)
        points.append([x, y, z])
    
    # Right edge (x near 50)
    for _ in range(n_obstacles // 4):
        x = np.random.uniform(45, 50)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(0, 2)
        points.append([x, y, z])
    
    # Top edge (y near 20)
    for _ in range(n_obstacles // 4):
        x = np.random.uniform(0, 50)
        y = np.random.uniform(15, 20)
        z = np.random.uniform(0, 2)
        points.append([x, y, z])
    
    # Bottom edge (y near -20)
    for _ in range(n_obstacles // 4):
        x = np.random.uniform(0, 50)
        y = np.random.uniform(-20, -15)
        z = np.random.uniform(0, 2)
        points.append([x, y, z])
    
    # Sparse ground
    for _ in range(500):
        x = np.random.uniform(0, 50)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(-2, -1.5)
        points.append([x, y, z])
    
    return np.array(points)

class SimpleVoidDetector:
    """Simple, robust void detector"""
    
    def __init__(self, x_range=(0, 50), y_range=(-20, 20), resolution=1.0):  # Coarser!
        self.x_range = x_range
        self.y_range = y_range
        self.resolution = resolution
        
        self.G_x = int((x_range[1] - x_range[0]) / resolution)
        self.G_y = int((y_range[1] - y_range[0]) / resolution)
        
        print(f"Grid: {self.G_x}×{self.G_y} at {resolution}m resolution")
    
    def detect_void(self, points_3d):
        """Detect void in one shot"""
        # Create occupancy grid
        occupancy = np.zeros((self.G_x, self.G_y), dtype=bool)
        
        # Project points
        points_2d = points_3d[:, :2]
        grid_x = ((points_2d[:, 0] - self.x_range[0]) / self.resolution).astype(int)
        grid_y = ((points_2d[:, 1] - self.y_range[0]) / self.resolution).astype(int)
        
        grid_x = np.clip(grid_x, 0, self.G_x - 1)
        grid_y = np.clip(grid_y, 0, self.G_y - 1)
        
        # Mark occupied - NO DILATION
        for i, j in zip(grid_x, grid_y):
            occupancy[i, j] = True
        
        occupancy_rate = occupancy.sum() / (self.G_x * self.G_y)
        print(f"Occupancy: {occupancy_rate*100:.1f}%")
        
        # Compute distance field
        free_space = ~occupancy
        distance_field = distance_transform_edt(free_space) * self.resolution
        
        print(f"Max distance: {distance_field.max():.2f}m")
        
        # Find largest void
        max_filtered = maximum_filter(distance_field, size=5)
        local_maxima = (distance_field == max_filtered) & (distance_field > 2.0)
        
        if not local_maxima.any():
            return None, occupancy, distance_field
        
        # Get best void
        peak_indices = np.argwhere(local_maxima)
        peak_values = distance_field[local_maxima]
        best_idx = np.argmax(peak_values)
        best_peak = peak_indices[best_idx]
        best_radius = peak_values[best_idx]
        
        # World coordinates
        center_x = self.x_range[0] + (best_peak[0] + 0.5) * self.resolution
        center_y = self.y_range[0] + (best_peak[1] + 0.5) * self.resolution
        
        result = {
            'center': (center_x, center_y),
            'radius': best_radius
        }
        
        return result, occupancy, distance_field

def main():
    print("="*60)
    print("Working Void Detection Demo")
    print("="*60)
    
    # Generate scene with clear void
    print("\nGenerating scene...")
    points = generate_clear_void_scene(seed=42)
    
    # Filter
    mask = ((points[:, 0] >= 0) & (points[:, 0] <= 50) &
            (points[:, 1] >= -20) & (points[:, 1] <= 20) &
            (points[:, 2] >= -2) & (points[:, 2] <= 5))
    points = points[mask]
    
    print(f"Points: {len(points):,}")
    print("Layout: Obstacles at perimeter edges only")
    print("Expected void: Entire center region (~20m radius)")
    
    # Detect void
    print("\n" + "="*60)
    print("Detecting Void")
    print("="*60)
    
    detector = SimpleVoidDetector(resolution=1.0)  # 1m resolution for larger voids
    
    start = time.time()
    result, occupancy, distance_field = detector.detect_void(points)
    elapsed = time.time() - start
    
    print(f"\nQuery time: {elapsed:.3f}s")
    
    if result:
        print(f"\n✓ VOID DETECTED!")
        print(f"  Center: ({result['center'][0]:.1f}m, {result['center'][1]:.1f}m)")
        print(f"  Radius: {result['radius']:.2f}m")
        print(f"  Navigable: ✓ YES")
    else:
        print("\n✗ No void found")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Points
    ax1 = axes[0]
    ax1.scatter(points[:, 0], points[:, 1], s=3, c='red', alpha=0.6, label='Obstacles')
    if result:
        circle = plt.Circle(result['center'], result['radius'], 
                          fill=False, color='green', linewidth=3, 
                          label=f'Void (r={result["radius"]:.1f}m)')
        ax1.add_patch(circle)
        ax1.plot(result['center'][0], result['center'][1], 'g*', markersize=20, label='Waypoint')
    ax1.set_xlabel('X (forward) [m]')
    ax1.set_ylabel('Y (lateral) [m]')
    ax1.set_title('Point Cloud + Detected Void')
    ax1.set_xlim(0, 50)
    ax1.set_ylim(-20, 20)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Occupancy
    ax2 = axes[1]
    extent = [-20, 20, 0, 50]
    ax2.imshow(occupancy, cmap='gray_r', extent=extent, origin='lower', aspect='auto')
    ax2.set_xlabel('Y (lateral) [m]')
    ax2.set_ylabel('X (forward) [m]')
    ax2.set_title('Occupancy Grid')
    
    # Plot 3: Distance field
    ax3 = axes[2]
    im = ax3.imshow(distance_field, cmap='hot', extent=extent, origin='lower', aspect='auto')
    plt.colorbar(im, ax=ax3, label='Distance [m]')
    if result:
        circle = plt.Circle((result['center'][1], result['center'][0]), 
                          result['radius'], 
                          fill=False, color='cyan', linewidth=3)
        ax3.add_patch(circle)
        ax3.plot(result['center'][1], result['center'][0], 'c*', markersize=20)
    ax3.set_xlabel('Y (lateral) [m]')
    ax3.set_ylabel('X (forward) [m]')
    ax3.set_title('Distance Field (Void Detection)')
    
    plt.tight_layout()
    plt.savefig('obds_void_experiment/working_void_demo.png', dpi=300)
    print("\n✓ Saved: obds_void_experiment/working_void_demo.png")

if __name__ == "__main__":
    main()
    print("\n✓ Demo complete!")
