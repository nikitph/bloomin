"""
Final tuned void detection demo with optimized Gray-Scott parameters
"""

import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from lidar_void_detection import LiDARVoidDetectorOBDS

def generate_simple_void_scene(num_points=10000, seed=42):
    """Generate a very simple scene with clear void"""
    np.random.seed(seed)
    points = []
    
    # Ring of obstacles around perimeter
    n_ring = int(num_points * 0.8)
    for _ in range(n_ring):
        # Random angle
        theta = np.random.uniform(0, 2*np.pi)
        # Radius between 15-20m from center
        r = np.random.uniform(15, 20)
        
        x = 25 + r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.uniform(0, 2)
        
        if 0 <= x <= 50 and -20 <= y <= 20:
            points.append([x, y, z])
    
    # Ground (sparse)
    n_ground = int(num_points * 0.2)
    for _ in range(n_ground):
        x = np.random.uniform(0, 50)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(-2, -1)
        points.append([x, y, z])
    
    points = np.array(points)
    
    print(f"Generated ring scene with {len(points):,} points")
    print(f"  Ring of obstacles at radius 15-20m")
    print(f"  Expected void: center region (radius ~15m)")
    
    return points

def run_tuned_demo():
    """Run with tuned parameters"""
    print("="*60)
    print("Tuned Void Detection Demo")
    print("="*60)
    
    # Generate scene
    points = generate_simple_void_scene(num_points=10000, seed=42)
    
    # Filter
    mask = ((points[:, 0] >= 0) & (points[:, 0] <= 50) &
            (points[:, 1] >= -20) & (points[:, 1] <= 20) &
            (points[:, 2] >= -2) & (points[:, 2] <= 5))
    points = points[mask]
    
    print(f"\nFiltered to {len(points):,} points")
    
    # OBDS with tuned parameters for void growth
    print("\n" + "="*60)
    print("OBDS (Tuned for void detection)")
    print("="*60)
    
    obds = LiDARVoidDetectorOBDS(
        x_range=(0, 50), 
        y_range=(-20, 20), 
        resolution=0.4,  # Coarser for faster evolution
        Du=2e-5,
        Dv=1e-5,
        F=0.039,  # Lower F promotes void growth
        k=0.058   # Adjusted k
    )
    
    init_start = time.time()
    init_info = obds.initialize_from_lidar(points)
    init_time = time.time() - init_start
    
    print(f"Init: {init_time:.3f}s")
    print(f"  Grid: {obds.G_x}×{obds.G_y}")
    print(f"  Occupancy: {init_info['occupancy_rate']*100:.1f}%")
    
    # Evolution
    query_time = obds.evolve_gpu(T=15000, dt=1.0, device='mps')
    obds_result = obds.find_largest_void_for_navigation(min_radius_m=2.0)
    
    print(f"Query: {query_time:.3f}s")
    print(f"  Max v: {obds.v.max():.3f}")
    print(f"  Mean v: {obds.v.mean():.3f}")
    
    if obds_result['status'] == 'void_found':
        print(f"\n✓ Void detected!")
        print(f"  Center: ({obds_result['center_world'][0]:.1f}m, {obds_result['center_world'][1]:.1f}m)")
        print(f"  Radius: {obds_result['radius_m']:.2f}m")
        print(f"  Field strength: {obds_result['field_strength']:.3f}")
        print(f"  Navigable: {'✓' if obds_result['navigable'] else '✗'}")
    else:
        print("\n✗ No void found")
    
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
    
    # Plot 2: Activator u
    ax2 = axes[1]
    extent = [obds.y_range[0], obds.y_range[1], obds.x_range[0], obds.x_range[1]]
    im2 = ax2.imshow(obds.u, cmap='gray', extent=extent, origin='lower', aspect='auto')
    plt.colorbar(im2, ax=ax2, label='Activator u')
    ax2.set_xlabel('Y (lateral) [m]')
    ax2.set_ylabel('X (forward) [m]')
    ax2.set_title('Activator Field')
    
    # Plot 3: Inhibitor v (voids)
    ax3 = axes[2]
    im3 = ax3.imshow(obds.v, cmap='hot', extent=extent, origin='lower', aspect='auto')
    plt.colorbar(im3, ax=ax3, label='Inhibitor v (Void)')
    ax3.set_xlabel('Y (lateral) [m]')
    ax3.set_ylabel('X (forward) [m]')
    ax3.set_title('Void Detection Field')
    
    if obds_result['status'] == 'void_found' and obds_result['radius_m'] > 0:
        circle = plt.Circle(
            (obds_result['center_world'][1], obds_result['center_world'][0]),
            obds_result['radius_m'],
            fill=False, color='cyan', linewidth=3,
            label=f'Detected void (r={obds_result["radius_m"]:.1f}m)'
        )
        ax3.add_patch(circle)
        ax3.plot(obds_result['center_world'][1], obds_result['center_world'][0],
                'c*', markersize=20)
        ax3.legend()
    
    plt.tight_layout()
    plt.savefig('obds_void_experiment/tuned_void_demo.png', dpi=300)
    print("\n✓ Visualization saved to: obds_void_experiment/tuned_void_demo.png")
    
    return obds_result

if __name__ == "__main__":
    result = run_tuned_demo()
    print("\n✓ Demo complete!")
