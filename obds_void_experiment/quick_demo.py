"""
Quick demo using Open3D sample data (no download needed)
Demonstrates void detection on real 3D point clouds
"""

import numpy as np
import open3d as o3d
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from lidar_void_detection import (
    LiDARVoidDetectorOBDS,
    LiDARVoidDetectorClassical,
    generate_report
)
import time

def load_open3d_sample():
    """Load sample point cloud from Open3D"""
    print("Loading Open3D sample point cloud...")
    
    # Try to load sample data
    try:
        pcd = o3d.data.DemoICPPointClouds()
        cloud = o3d.io.read_point_cloud(pcd.paths[0])
        points = np.asarray(cloud.points)
        
        # Normalize to driving scenario range
        points = points - points.mean(axis=0)
        points = points / points.std() * 10  # Scale to ~20m range
        points[:, 0] += 25  # Shift forward to 0-50m range
        
        print(f"✓ Loaded {len(points):,} points from Open3D sample")
        return points
        
    except Exception as e:
        print(f"Could not load Open3D sample: {e}")
        print("Falling back to synthetic data...")
        return None

def quick_demo():
    """Quick demonstration with small real data"""
    print("="*60)
    print("Quick Void Detection Demo (Open3D Sample Data)")
    print("="*60)
    
    # Try Open3D sample
    points = load_open3d_sample()
    
    # Fallback to synthetic
    if points is None:
        print("\nGenerating synthetic LiDAR data...")
        from lidar_void_detection import LiDARDataLoader
        loader = LiDARDataLoader(".", "kitti")
        points = loader.generate_synthetic_lidar(num_points=50000, seed=42)
        print(f"✓ Generated {len(points):,} synthetic points")
    
    # Filter to driving range
    mask = ((points[:, 0] >= 0) & (points[:, 0] <= 50) &
            (points[:, 1] >= -20) & (points[:, 1] <= 20) &
            (points[:, 2] >= -2) & (points[:, 2] <= 5))
    points = points[mask]
    
    print(f"\nFiltered to {len(points):,} points in driving range")
    
    # Initialize detectors
    obds = LiDARVoidDetectorOBDS(x_range=(0, 50), y_range=(-20, 20), resolution=0.2)
    classical = LiDARVoidDetectorClassical(x_range=(0, 50), y_range=(-20, 20))
    
    # Run benchmark
    print("\n" + "="*60)
    print("Running Benchmark")
    print("="*60)
    
    # Classical
    print("\n1. Classical (Delaunay)...")
    classical_result = classical.find_largest_void_delaunay(points)
    print(f"   Time: {classical_result['time']:.3f}s")
    print(f"   Void radius: {classical_result['radius_m']:.2f}m")
    print(f"   Navigable: {'✓' if classical_result['navigable'] else '✗'}")
    
    # OBDS
    print("\n2. OBDS (GPU)...")
    
    # Init
    init_start = time.time()
    obds.initialize_from_lidar(points)
    init_time = time.time() - init_start
    print(f"   Init: {init_time:.3f}s")
    
    # Query
    query_time = obds.evolve_gpu(T=2000, dt=1.0, device='mps')
    obds_result = obds.find_largest_void_for_navigation(min_radius_m=2.0)
    print(f"   Query: {query_time:.3f}s")
    print(f"   Void radius: {obds_result['radius_m']:.2f}m")
    print(f"   Navigable: {'✓' if obds_result['navigable'] else '✗'}")
    
    # Results
    speedup = classical_result['time'] / query_time
    realtime = query_time < 0.033
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Speedup: {speedup:.1f}×")
    print(f"Real-time capable (30 Hz): {'✓ YES' if realtime else '✗ NO'}")
    print(f"Query time: {query_time*1000:.1f}ms (target: <33ms)")
    
    return {
        'points': points,
        'classical': classical_result,
        'obds': obds_result,
        'speedup': speedup,
        'realtime': realtime
    }

if __name__ == "__main__":
    results = quick_demo()
    print("\n✓ Demo complete!")
