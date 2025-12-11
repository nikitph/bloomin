"""
Geometric Metrics for REWA-Causal

Provides geodesic distances, dispersion metrics, hull overlap detection,
and weighted union operations for causal identification.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from .spherical_hull import SphericalConvexHull, spherical_convex_hull


def frechet_mean(points: np.ndarray, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """
    Compute the Fréchet mean (geodesic center) on the sphere.

    The Fréchet mean minimizes the sum of squared geodesic distances.

    Args:
        points: Array of unit vectors, shape (n, d)
        max_iter: Maximum iterations for convergence
        tol: Convergence tolerance

    Returns:
        Unit vector representing the Fréchet mean
    """
    points = np.atleast_2d(points)

    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    points = points / norms

    n = points.shape[0]
    if n == 0:
        raise ValueError("Cannot compute Fréchet mean of empty set")

    if n == 1:
        return points[0].copy()

    # Initialize with Euclidean mean
    mu = np.mean(points, axis=0)
    norm = np.linalg.norm(mu)
    if norm < 1e-10:
        mu = points[0].copy()
    else:
        mu = mu / norm

    for _ in range(max_iter):
        # Compute tangent vectors at mu pointing toward each point
        dots = np.clip(points @ mu, -1.0, 1.0)
        angles = np.arccos(dots)

        # Unit tangent directions
        tangent_dirs = points - np.outer(dots, mu)
        tangent_norms = np.linalg.norm(tangent_dirs, axis=1, keepdims=True)
        tangent_norms = np.where(tangent_norms > 1e-10, tangent_norms, 1.0)
        tangent_dirs = tangent_dirs / tangent_norms

        # Mean tangent (weighted by geodesic distance)
        mean_tangent = np.mean(angles[:, np.newaxis] * tangent_dirs, axis=0)
        step_size = np.linalg.norm(mean_tangent)

        if step_size < tol:
            break

        # Move along geodesic (exponential map)
        direction = mean_tangent / step_size
        mu = mu * np.cos(step_size) + direction * np.sin(step_size)
        mu = mu / np.linalg.norm(mu)

    return mu


def compute_dispersion(
    witnesses: np.ndarray,
    center: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute dispersion metrics for a witness set.

    Measures the geometric spread/ambiguity of the witness set.

    Args:
        witnesses: Array of unit vectors, shape (n, d)
        center: Optional precomputed center (will compute Fréchet mean if not provided)

    Returns:
        Dictionary with dispersion metrics:
        - radius: Maximum angle from center to any witness
        - mean_radius: Mean angle from center
        - variance: Angular variance
        - spread: Mean pairwise angle
    """
    witnesses = np.atleast_2d(witnesses)

    # Normalize
    norms = np.linalg.norm(witnesses, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    witnesses = witnesses / norms

    n = witnesses.shape[0]

    if n == 0:
        return {
            'radius': 0.0,
            'mean_radius': 0.0,
            'variance': 0.0,
            'spread': 0.0
        }

    if n == 1:
        return {
            'radius': 0.0,
            'mean_radius': 0.0,
            'variance': 0.0,
            'spread': 0.0
        }

    # Compute center if not provided
    if center is None:
        center = frechet_mean(witnesses)
    else:
        center = center / np.linalg.norm(center)

    # Angles from center
    dots_to_center = np.clip(witnesses @ center, -1.0, 1.0)
    angles_to_center = np.arccos(dots_to_center)

    # Pairwise angles
    pairwise_dots = np.clip(witnesses @ witnesses.T, -1.0, 1.0)
    pairwise_angles = np.arccos(pairwise_dots)
    upper_triangle = pairwise_angles[np.triu_indices(n, k=1)]

    return {
        'radius': float(np.max(angles_to_center)),
        'mean_radius': float(np.mean(angles_to_center)),
        'variance': float(np.var(angles_to_center)),
        'spread': float(np.mean(upper_triangle)) if len(upper_triangle) > 0 else 0.0
    }


def geodesic_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Compute geodesic (great circle) distance between two points on the sphere.

    Args:
        p1: First point (will be normalized)
        p2: Second point (will be normalized)

    Returns:
        Geodesic distance in radians [0, π]
    """
    p1 = p1 / np.linalg.norm(p1)
    p2 = p2 / np.linalg.norm(p2)

    dot = np.clip(np.dot(p1, p2), -1.0, 1.0)
    return float(np.arccos(dot))


def hull_overlap(
    hull1: SphericalConvexHull,
    hull2: SphericalConvexHull,
    n_samples: int = 100
) -> Dict[str, Any]:
    """
    Estimate the overlap between two spherical convex hulls.

    Uses sampling-based estimation since exact hull intersection
    on high-dimensional spheres is computationally expensive.

    Args:
        hull1: First spherical convex hull
        hull2: Second spherical convex hull
        n_samples: Number of samples for estimation

    Returns:
        Dictionary with overlap metrics:
        - has_overlap: Boolean indicating if hulls overlap
        - overlap_fraction: Estimated fraction of hull1 that overlaps with hull2
        - centroid_distance: Geodesic distance between centroids
        - sample_overlap_count: Number of hull1 samples inside hull2
    """
    if hull1.n_points == 0 or hull2.n_points == 0:
        return {
            'has_overlap': False,
            'overlap_fraction': 0.0,
            'centroid_distance': float('inf'),
            'sample_overlap_count': 0
        }

    # Centroid distance
    centroid_dist = geodesic_distance(hull1.centroid, hull2.centroid)

    # Quick check: if centroids are far apart, likely no overlap
    if centroid_dist > hull1.radius + hull2.radius + 0.1:
        return {
            'has_overlap': False,
            'overlap_fraction': 0.0,
            'centroid_distance': centroid_dist,
            'sample_overlap_count': 0
        }

    # Sample from hull1 and check containment in hull2
    samples = hull1.sample_interior(n_samples)
    in_hull2 = sum(1 for s in samples if hull2.contains_point(s))

    overlap_fraction = in_hull2 / n_samples

    return {
        'has_overlap': overlap_fraction > 0,
        'overlap_fraction': overlap_fraction,
        'centroid_distance': centroid_dist,
        'sample_overlap_count': in_hull2
    }


def weighted_union(
    hulls: List[SphericalConvexHull],
    weights: Optional[List[float]] = None
) -> SphericalConvexHull:
    """
    Compute weighted union of multiple spherical convex hulls.

    Used in do-calculus for combining interventional distributions.

    The weighted union combines points from all hulls, with optional
    weighting for the centroid computation.

    Args:
        hulls: List of SphericalConvexHull objects
        weights: Optional weights for each hull (default: uniform)

    Returns:
        Combined SphericalConvexHull
    """
    if not hulls:
        return SphericalConvexHull(points=np.zeros((0, 1)))

    # Collect all points
    all_points = []
    all_weights = []

    if weights is None:
        weights = [1.0 / len(hulls)] * len(hulls)

    for hull, w in zip(hulls, weights):
        if hull.n_points > 0:
            all_points.append(hull.points)
            # Weight each point by hull weight divided by number of points
            point_weight = w / hull.n_points if hull.n_points > 0 else 0
            all_weights.extend([point_weight] * hull.n_points)

    if not all_points:
        return SphericalConvexHull(points=np.zeros((0, 1)))

    combined_points = np.vstack(all_points)

    # Compute weighted centroid
    weights_array = np.array(all_weights)
    weights_array = weights_array / weights_array.sum()  # Normalize

    weighted_mean = (weights_array[:, np.newaxis] * combined_points).sum(axis=0)
    norm = np.linalg.norm(weighted_mean)
    if norm > 1e-10:
        weighted_centroid = weighted_mean / norm
    else:
        weighted_centroid = combined_points[0] / np.linalg.norm(combined_points[0])

    return SphericalConvexHull(points=combined_points, centroid=weighted_centroid)


def hull_union(hulls: List[SphericalConvexHull]) -> SphericalConvexHull:
    """
    Compute the union of multiple spherical convex hulls (unweighted).

    Args:
        hulls: List of SphericalConvexHull objects

    Returns:
        Combined SphericalConvexHull containing all points
    """
    return weighted_union(hulls, weights=None)


def hull_intersection_test(
    hull1: SphericalConvexHull,
    hull2: SphericalConvexHull
) -> bool:
    """
    Quick test for whether two hulls have non-empty intersection.

    Args:
        hull1: First hull
        hull2: Second hull

    Returns:
        True if hulls likely intersect
    """
    # Check if either centroid is in the other hull
    if hull1.n_points > 0 and hull2.n_points > 0:
        if hull2.contains_point(hull1.centroid):
            return True
        if hull1.contains_point(hull2.centroid):
            return True

    # Check centroid distance vs sum of radii
    if hull1.n_points > 0 and hull2.n_points > 0:
        dist = geodesic_distance(hull1.centroid, hull2.centroid)
        return dist < hull1.radius + hull2.radius

    return False


def compute_causal_shift(
    hull_before: SphericalConvexHull,
    hull_after: SphericalConvexHull
) -> Dict[str, Any]:
    """
    Compute the causal shift between two regions (before/after intervention).

    Args:
        hull_before: Region before intervention (or under X=x0)
        hull_after: Region after intervention (or under X=x1)

    Returns:
        Dictionary with causal shift metrics:
        - centroid_shift: Geodesic distance between centroids
        - direction: Unit vector direction of shift
        - radius_change: Change in radius
        - overlap: Overlap metrics
    """
    if hull_before.n_points == 0 or hull_after.n_points == 0:
        return {
            'centroid_shift': float('inf'),
            'direction': None,
            'radius_change': 0.0,
            'overlap': None
        }

    # Centroid shift
    centroid_shift = geodesic_distance(hull_before.centroid, hull_after.centroid)

    # Direction of shift (in tangent space at hull_before.centroid)
    c1 = hull_before.centroid
    c2 = hull_after.centroid

    # Project c2 onto tangent plane at c1
    tangent = c2 - (c2 @ c1) * c1
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm > 1e-10:
        direction = tangent / tangent_norm
    else:
        direction = np.zeros_like(c1)

    # Radius change
    radius_change = hull_after.radius - hull_before.radius

    # Overlap
    overlap = hull_overlap(hull_before, hull_after)

    return {
        'centroid_shift': centroid_shift,
        'direction': direction,
        'radius_change': radius_change,
        'overlap': overlap
    }
