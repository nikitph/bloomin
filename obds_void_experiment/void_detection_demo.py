"""
Synthetic Void Detection Demo
Creates point clouds with known large voids to validate OBDS detection
"""

import numpy as np
import sys
import os
import time
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from lidar_void_detection import (
    LiDARVoidDetectorOBDS,
    LiDARVoidDetectorClassical
)

def generate_scene_with_voids(num_points=20000, seed=42):
    """
    Generate a driving scenario with deliberate large voids
    
    Layout:
    - Obstacles on left and right sides (buildings/walls)
    - Large navigable void in the center (road)
    - Some scattered obstacles
    """
    np.random.seed(seed)
    points = []
    
    # Left wall (x: 10-40m, y: -20 to -10m)
    n_left = int(num_points * 0.3)
    for _ in range(n_left):
        x = np.random.uniform(10, 40)
        y = np.random.uniform(-20, -10)
        z = np.random.uniform(-1, 3)
        points.append([x, y, z])
    
    # Right wall (x: 10-40m, y: 10 to 20m)
    n_right = int(num_points * 0.3)
    for _ in range(n_right):
        x = np.random.uniform(10, 40)
        y = np.random.uniform(10, 20)
        z = np.random.uniform(-1, 3)
        points.append([x, y, z])
    
    # Scattered obstacles in center (sparse)
    n_obstacles = int(num_points * 0.2)
    for _ in range(n_obstacles):
        x = np.random.uniform(5, 45)
        y = np.random.uniform(-8, 8)
        z = np.random.uniform(-1, 2)
        # Only add if not in the main void area
        if not (15 < x < 35 and -5 < y < 5):
            points.append([x, y, z])
    
    # Ground plane (sparse)
    n_ground = int(num_points * 0.2)
    for _ in range(n_ground):
        x = np.random.uniform(0, 50)
        y = np.random.uniform(-20, 20)
        z = np.random.uniform(-2, -1.5)
        points.append([x, y, z])
    
    points = np.array(points)
    
    print(f"Generated scene with {len(points):,} points")
    print(f"  Left wall: ~{n_left} points")
    print(f"  Right wall: ~{n_right} points")
    print(f"  Expected void: center region (x: 15-35m, y: -5 to 5m)")
    print(f"  Expected void radius: ~5m")
    
    return points

def visualize_scene(points, obds_result, classical_result):
    """Create bird's eye view visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Point cloud with classical void
    ax1 = axes[0]
    ax1.scatter(points[:, 0], points[:, 1], s=1, c='black', alpha=0.3, label='Obstacles')
    
    if classical_result['in_range']:
        circle = plt.Circle(
            (classical_result['center_world'][0], classical_result['center_world'][1]),
            classical_result['radius_m'],
            fill=False, color='blue', linewidth=2,
            label=f"Classical void (r={classical_result['radius_m']:.1f}m)"
        )
        ax1.add_patch(circle)
        ax1.plot(classical_result['center_world'][0], classical_result['center_world'][1],
                'b*', markersize=15)
    
    ax1.set_xlabel('X (forward) [m]')
    ax1.set_ylabel('Y (lateral) [m]')
    ax1.set_title('Classical Void Detection')
    ax1.set_xlim(0, 50)
    ax1.set_ylim(-20, 20)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Point cloud with OBDS void
    ax2 = axes[1]
    ax2.scatter(points[:, 0], points[:, 1], s=1, c='black', alpha=0.3, label='Obstacles')
    
    if obds_result['status'] == 'void_found' and obds_result['radius_m'] > 0:
        circle = plt.Circle(
            (obds_result['center_world'][0], obds_result['center_world'][1]),
            obds_result['radius_m'],
            fill=False, color='red', linewidth=2,
            label=f"OBDS void (r={obds_result['radius_m']:.1f}m)"
        )
        ax2.add_patch(circle)
        ax2.plot(obds_result['center_world'][0], obds_result['center_world'][1],
                'r*', markersize=15, label='Navigation waypoint')
    
    ax2.set_xlabel('X (forward) [m]')
    ax2.set_ylabel('Y (lateral) [m]')
    ax2.set_title('OBDS Void Detection')
    ax2.set_xlim(0, 50)
    ax2.set_ylim(-20, 20)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('obds_void_experiment/void_detection_demo.png', dpi=300)
    print("\n✓ Visualization saved to: obds_void_experiment/void_detection_demo.png")

def void_detection_demo():
    """Demonstrate successful void detection on synthetic data"""
    print("="*60)
    print("Void Detection Demo (Synthetic Data with Known Voids)")
    print("="*60)
    
    # Generate scene
    points = generate_scene_with_voids(num_points=20000, seed=42)
    
    # Filter to driving range
    mask = ((points[:, 0] >= 0) & (points[:, 0] <= 50) &
            (points[:, 1] >= -20) & (points[:, 1] <= 20) &
            (points[:, 2] >= -2) & (points[:, 2] <= 5))
    points = points[mask]
    
    print(f"\nFiltered to {len(points):,} points")
    
    # Classical method
    print("\n" + "="*60)
    print("1. Classical (Delaunay)")
    print("="*60)
    
    classical = LiDARVoidDetectorClassical(x_range=(0, 50), y_range=(-20, 20))
    classical_result = classical.find_largest_void_delaunay(points)
    
    print(f"Time: {classical_result['time']:.3f}s")
    print(f"Void center: ({classical_result['center_world'][0]:.1f}m, {classical_result['center_world'][1]:.1f}m)")
    print(f"Void radius: {classical_result['radius_m']:.2f}m")
    print(f"Navigable: {'✓' if classical_result['navigable'] else '✗'}")
    
    # OBDS method
    print("\n" + "="*60)
    print("2. OBDS (GPU, 0.2m resolution)")
    print("="*60)
    
    obds = LiDARVoidDetectorOBDS(x_range=(0, 50), y_range=(-20, 20), resolution=0.2)
    
    init_start = time.time()
    obds.initialize_from_lidar(points)
    init_time = time.time() - init_start
    
    query_time = obds.evolve_gpu(T=2000, dt=1.0, device='mps')
    obds_result = obds.find_largest_void_for_navigation(min_radius_m=2.0)
    
    print(f"Init: {init_time:.3f}s")
    print(f"Query: {query_time:.3f}s")
    
    if obds_result['status'] == 'void_found':
        print(f"Void center: ({obds_result['center_world'][0]:.1f}m, {obds_result['center_world'][1]:.1f}m)")
        print(f"Void radius: {obds_result['radius_m']:.2f}m")
        print(f"Field strength: {obds_result['field_strength']:.3f}")
        print(f"Navigable: {'✓' if obds_result['navigable'] else '✗'}")
    else:
        print("No void found")
    
    # Results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    speedup = classical_result['time'] / query_time
    print(f"Speedup: {speedup:.1f}×")
    print(f"Query time: {query_time*1000:.1f}ms")
    
    if obds_result['radius_m'] > 0:
        accuracy = (obds_result['radius_m'] / classical_result['radius_m']) * 100
        print(f"Radius accuracy: {accuracy:.1f}% of classical")
        
        center_error = np.linalg.norm(
            np.array(obds_result['center_world']) - 
            np.array(classical_result['center_world'])
        )
        print(f"Center error: {center_error:.2f}m")
    
    # Visualize
    visualize_scene(points, obds_result, classical_result)
    
    return {
        'points': points,
        'classical': classical_result,
        'obds': obds_result,
        'speedup': speedup
    }

if __name__ == "__main__":
    results = void_detection_demo()
    print("\n✓ Demo complete!")
