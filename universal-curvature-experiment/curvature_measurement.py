"""
Improved Curvature Measurement Module
======================================

Measures Gaussian curvature K using spherical law of cosines.
This is the corrected version that works in high dimensions.
"""

import numpy as np
from typing import Dict, Tuple


def compute_spherical_triangle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """
    Compute sides and angles of a spherical triangle.
    
    Uses spherical law of cosines to compute angles from sides.
    
    Args:
        p1, p2, p3: Three points on unit sphere (normalized)
        
    Returns:
        (a, b, c, A, B, C) where:
        - a, b, c are geodesic side lengths
        - A, B, C are angles at vertices
    """
    # Compute geodesic distances (sides)
    a = np.arccos(np.clip(p2 @ p3, -1, 1))  # Side opposite to p1
    b = np.arccos(np.clip(p1 @ p3, -1, 1))  # Side opposite to p2
    c = np.arccos(np.clip(p1 @ p2, -1, 1))  # Side opposite to p3
    
    # Check for degenerate triangle
    if a < 1e-6 or b < 1e-6 or c < 1e-6:
        return None
    
    # Check triangle inequality
    if a + b + c > 2*np.pi or a >= b + c or b >= a + c or c >= a + b:
        return None
    
    # Compute angles using spherical law of cosines
    # cos(A) = (cos(a) - cos(b)cos(c)) / (sin(b)sin(c))
    
    sin_a, sin_b, sin_c = np.sin(a), np.sin(b), np.sin(c)
    cos_a, cos_b, cos_c = np.cos(a), np.cos(b), np.cos(c)
    
    # Avoid division by zero
    if sin_b * sin_c < 1e-10 or sin_a * sin_c < 1e-10 or sin_a * sin_b < 1e-10:
        return None
    
    cos_A = (cos_a - cos_b * cos_c) / (sin_b * sin_c)
    cos_B = (cos_b - cos_a * cos_c) / (sin_a * sin_c)
    cos_C = (cos_c - cos_a * cos_b) / (sin_a * sin_b)
    
    # Clamp to valid range
    cos_A = np.clip(cos_A, -1, 1)
    cos_B = np.clip(cos_B, -1, 1)
    cos_C = np.clip(cos_C, -1, 1)
    
    A = np.arccos(cos_A)
    B = np.arccos(cos_B)
    C = np.arccos(cos_C)
    
    return (a, b, c, A, B, C)


def spherical_excess(A: float, B: float, C: float) -> float:
    """
    Compute spherical excess: E = (A + B + C) - π
    
    For a spherical triangle, the sum of angles exceeds π.
    This excess is related to the area and curvature.
    
    Args:
        A, B, C: Angles of spherical triangle (radians)
        
    Returns:
        Spherical excess E
    """
    return (A + B + C) - np.pi


def compute_triangle_curvature(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    Compute Gaussian curvature K from a spherical triangle.
    
    Uses spherical excess method: K = E / Area
    For unit sphere, K should equal 1.
    
    Args:
        p1, p2, p3: Three points on sphere (normalized)
        
    Returns:
        Gaussian curvature K, or np.nan if triangle is degenerate
    """
    # Compute triangle
    result = compute_spherical_triangle(p1, p2, p3)
    
    if result is None:
        return np.nan
    
    a, b, c, A, B, C = result
    
    # Compute spherical excess
    E = spherical_excess(A, B, C)
    
    if E < 1e-10:
        return np.nan
    
    # Compute spherical area using L'Huilier's theorem
    s = (a + b + c) / 2  # Semi-perimeter
    
    # L'Huilier's formula for spherical excess
    tan_E_4 = np.sqrt(
        np.tan(s/2) * 
        np.tan((s-a)/2) * 
        np.tan((s-b)/2) * 
        np.tan((s-c)/2)
    )
    
    if np.isnan(tan_E_4) or tan_E_4 < 0:
        return np.nan
    
    area_spherical = 4 * np.arctan(tan_E_4)
    
    if area_spherical < 1e-10:
        return np.nan
    
    # Gaussian curvature: K = E / Area
    # For unit sphere (R=1), K should be 1
    K = E / area_spherical
    
    return K


def measure_curvature(
    embeddings: np.ndarray,
    n_triangles: int = 1000,
    random_seed: int = None,
    verbose: bool = False
) -> Dict:
    """
    Measure Gaussian curvature K from embeddings.
    
    Samples random triangles and computes K using spherical excess method.
    
    Args:
        embeddings: Shape (N, D) array of embeddings
        n_triangles: Number of random triangles to sample
        random_seed: Random seed for reproducibility
        verbose: Print progress
        
    Returns:
        Dictionary with:
        - K_mean: Mean curvature
        - K_std: Standard deviation
        - K_median: Median curvature
        - K_min, K_max: Range
        - K_samples: All valid measurements
        - n_valid_triangles: Number of valid triangles
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Normalize to sphere
    emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    K_samples = []
    
    for i in range(n_triangles):
        # Sample 3 random points
        idx = np.random.choice(len(emb_norm), 3, replace=False)
        p1, p2, p3 = emb_norm[idx]
        
        # Compute curvature
        K = compute_triangle_curvature(p1, p2, p3)
        
        if not np.isnan(K) and K > 0:  # Only keep positive, valid curvatures
            K_samples.append(K)
        
        if verbose and (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_triangles} triangles, {len(K_samples)} valid")
    
    if len(K_samples) == 0:
        raise ValueError("No valid triangles found!")
    
    K_samples = np.array(K_samples)
    
    # Remove outliers (beyond 3 sigma)
    mean = np.mean(K_samples)
    std = np.std(K_samples)
    mask = np.abs(K_samples - mean) < 3*std
    K_samples_filtered = K_samples[mask]
    
    if verbose:
        print(f"  Removed {len(K_samples) - len(K_samples_filtered)} outliers")
    
    return {
        'K_mean': np.mean(K_samples_filtered),
        'K_std': np.std(K_samples_filtered),
        'K_median': np.median(K_samples_filtered),
        'K_min': np.min(K_samples_filtered),
        'K_max': np.max(K_samples_filtered),
        'K_samples': K_samples_filtered,
        'n_valid_triangles': len(K_samples_filtered),
        'n_total_triangles': n_triangles
    }
