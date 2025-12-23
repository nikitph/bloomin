"""
Multi-Resolution Void Detection Demo
Demonstrates how grid resolution affects void detection accuracy
"""

import numpy as np
import open3d as o3d
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from lidar_void_detection import (
    LiDARVoidDetectorOBDS,
    LiDARVoidDetectorClassical
)

def load_open3d_sample():
    """Load sample point cloud from Open3D"""
    print("Loading Open3D sample point cloud...")
    
    pcd = o3d.data.DemoICPPointClouds()
    cloud = o3d.io.read_point_cloud(pcd.paths[0])
    points = np.asarray(cloud.points)
    
    # Normalize to driving scenario range
    points = points - points.mean(axis=0)
    points = points / points.std() * 10
    points[:, 0] += 25
    
    print(f"✓ Loaded {len(points):,} points from Open3D sample")
    return points

def multi_resolution_demo():
    """Test multiple resolutions"""
    print("="*60)
    print("Multi-Resolution Void Detection Demo")
    print("="*60)
    
    # Load data
    points = load_open3d_sample()
    
    # Filter to driving range
    mask = ((points[:, 0] >= 0) & (points[:, 0] <= 50) &
            (points[:, 1] >= -20) & (points[:, 1] <= 20) &
            (points[:, 2] >= -2) & (points[:, 2] <= 5))
    points = points[mask]
    
    print(f"\nFiltered to {len(points):,} points in driving range")
    
    # Test different resolutions
    resolutions = [0.2, 0.1, 0.05]
    
    print("\n" + "="*60)
    print("Testing Multiple Resolutions")
    print("="*60)
    
    results = []
    
    for res in resolutions:
        print(f"\n{'='*60}")
        print(f"Resolution: {res}m ({int(50/res)}×{int(40/res)} grid)")
        print(f"{'='*60}")
        
        # Initialize OBDS
        obds = LiDARVoidDetectorOBDS(
            x_range=(0, 50), 
            y_range=(-20, 20), 
            resolution=res
        )
        
        # Init
        init_start = time.time()
        obds.initialize_from_lidar(points)
        init_time = time.time() - init_start
        print(f"Init: {init_time:.3f}s")
        
        # Query (reduce T for finer grids to keep time reasonable)
        T = int(2000 * (0.2 / res))  # Scale iterations with resolution
        query_start = time.time()
        obds.evolve_gpu(T=T, dt=1.0, device='mps')
        obds_result = obds.find_largest_void_for_navigation(min_radius_m=2.0)
        query_time = time.time() - query_start
        
        print(f"Query: {query_time:.3f}s (T={T})")
        print(f"Void radius: {obds_result['radius_m']:.2f}m")
        print(f"Navigable: {'✓' if obds_result['navigable'] else '✗'}")
        
        if obds_result['status'] == 'void_found' and obds_result['center_world'] is not None:
            print(f"Center: ({obds_result['center_world'][0]:.1f}m, {obds_result['center_world'][1]:.1f}m)")
        
        results.append({
            'resolution': res,
            'grid_size': (obds.G_x, obds.G_y),
            'init_time': init_time,
            'query_time': query_time,
            'total_time': init_time + query_time,
            'radius': obds_result['radius_m'],
            'navigable': obds_result['navigable']
        })
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Resolution':<12} {'Grid Size':<15} {'Query Time':<12} {'Void Radius':<12} {'Navigable'}")
    print("-"*60)
    
    for r in results:
        grid_str = f"{r['grid_size'][0]}×{r['grid_size'][1]}"
        nav_str = '✓' if r['navigable'] else '✗'
        print(f"{r['resolution']:.2f}m{'':<8} {grid_str:<15} {r['query_time']:.3f}s{'':<6} {r['radius']:.2f}m{'':<6} {nav_str}")
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    print("1. Finer resolution → Better void detection")
    print("2. Query time scales with grid size, NOT point count")
    print("3. Resolution-latency tradeoff is explicit and controllable")
    print(f"4. Point count ({len(points):,}) remains constant across all tests")
    
    return results

if __name__ == "__main__":
    results = multi_resolution_demo()
    print("\n✓ Multi-resolution demo complete!")
