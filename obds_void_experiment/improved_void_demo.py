"""
Improved Synthetic Void Detection Demo
Creates a clearer scenario with a well-defined navigable void
"""

import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from lidar_void_detection import LiDARVoidDetectorOBDS, LiDARVoidDetectorClassical

def generate_corridor_scene(num_points=15000, seed=42):
    """
    Generate a corridor/road scenario with clear void in center
    
    Layout:
    - Dense walls on left and right
    - Clear corridor in center (10m wide)
    - Some ground points
    """
    np.random.seed(seed)
    points = []
    
    # Left wall (dense)
    n_left = int(num_points * 0.4)
    for _ in range(n_left):
        x = np.random.uniform(5, 45)
        y = np.random.uniform(-20, -6)  # Leave gap at y=-6
        z = np.random.uniform(0, 3)
        points.append([x, y, z])
    
    # Right wall (dense)
    n_right = int(num_points * 0.4)
    for _ in range(n_right):
        x = np.random.uniform(5, 45)
        y = np.random.uniform(6, 20)  # Leave gap at y=6
        z = np.random.uniform(0, 3)
        points.append([x, y, z])
    
    # Ground (sparse, including center)
    n_ground = int(num_points * 0.2)
    for _ in range(n_ground):
        x = np.random.uniform(0, 50)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(-2, -1)
        points.append([x, y, z])
    
    points = np.array(points)
    
    print(f"Generated corridor scene with {len(points):,} points")
    print(f"  Left wall: y < -6m")
    print(f"  Right wall: y > 6m")
    print(f"  Clear corridor: -6m < y < 6m (12m wide)")
    print(f"  Expected void radius: ~6m")
    
    return points

def run_improved_demo():
    """Run demo with clearer void"""
    print("="*60)
    print("Improved Void Detection Demo")
    print("="*60)
    
    # Generate scene
    points = generate_corridor_scene(num_points=15000, seed=42)
    
    # Filter
    mask = ((points[:, 0] >= 0) & (points[:, 0] <= 50) &
            (points[:, 1] >= -20) & (points[:, 1] <= 20) &
            (points[:, 2] >= -2) & (points[:, 2] <= 5))
    points = points[mask]
    
    print(f"\nFiltered to {len(points):,} points")
    
    # OBDS with adjusted parameters
    print("\n" + "="*60)
    print("OBDS (GPU, 0.3m resolution, longer evolution)")
    print("="*60)
    
    obds = LiDARVoidDetectorOBDS(
        x_range=(0, 50), 
        y_range=(-20, 20), 
        resolution=0.3,  # Slightly coarser for faster evolution
        Du=2e-5,
        Dv=1e-5,
        F=0.055,
        k=0.062
    )
    
    init_start = time.time()
    init_info = obds.initialize_from_lidar(points)
    init_time = time.time() - init_start
    
    print(f"Init: {init_time:.3f}s")
    print(f"  Occupancy rate: {init_info['occupancy_rate']*100:.1f}%")
    
    # Longer evolution for clearer pattern
    query_time = obds.evolve_gpu(T=10000, dt=1.0, device='mps')
    obds_result = obds.find_largest_void_for_navigation(min_radius_m=2.0)
    
    print(f"Query: {query_time:.3f}s")
    
    if obds_result['status'] == 'void_found':
        print(f"✓ Void detected!")
        print(f"  Center: ({obds_result['center_world'][0]:.1f}m, {obds_result['center_world'][1]:.1f}m)")
        print(f"  Radius: {obds_result['radius_m']:.2f}m")
        print(f"  Field strength: {obds_result['field_strength']:.3f}")
        print(f"  Navigable: {'✓' if obds_result['navigable'] else '✗'}")
    else:
        print("✗ No void found")
        print(f"  Max field value: {obds.v.max():.3f}")
        print(f"  Mean field value: {obds.v.mean():.3f}")
    
    # Visualize field
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Points (BEV)
    ax1 = axes[0]
    ax1.scatter(points[:, 0], points[:, 1], s=1, c='black', alpha=0.5)
    ax1.set_xlabel('X (forward) [m]')
    ax1.set_ylabel('Y (lateral) [m]')
    ax1.set_title('Point Cloud (Bird\'s Eye View)')
    ax1.set_xlim(0, 50)
    ax1.set_ylim(-20, 20)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Activator field (u)
    ax2 = axes[1]
    extent = [obds.y_range[0], obds.y_range[1], obds.x_range[0], obds.x_range[1]]
    im2 = ax2.imshow(obds.u, cmap='gray', extent=extent, origin='lower', aspect='auto')
    plt.colorbar(im2, ax=ax2, label='Activator u')
    ax2.set_xlabel('Y (lateral) [m]')
    ax2.set_ylabel('X (forward) [m]')
    ax2.set_title('Activator Field (Obstacles)')
    
    # Plot 3: Inhibitor field (v) - voids
    ax3 = axes[2]
    im3 = ax3.imshow(obds.v, cmap='viridis', extent=extent, origin='lower', aspect='auto')
    plt.colorbar(im3, ax=ax3, label='Inhibitor v (Void indicator)')
    ax3.set_xlabel('Y (lateral) [m]')
    ax3.set_ylabel('X (forward) [m]')
    ax3.set_title('Inhibitor Field (Voids)')
    
    if obds_result['status'] == 'void_found' and obds_result['radius_m'] > 0:
        ax3.plot(obds_result['center_world'][1], obds_result['center_world'][0],
                'r*', markersize=20, label=f'Void (r={obds_result["radius_m"]:.1f}m)')
        ax3.legend()
    
    plt.tight_layout()
    plt.savefig('obds_void_experiment/void_fields_demo.png', dpi=300)
    print("\n✓ Visualization saved to: obds_void_experiment/void_fields_demo.png")
    
    return obds_result

if __name__ == "__main__":
    result = run_improved_demo()
    print("\n✓ Demo complete!")
