"""
Spherical Geometry Operations

All embeddings are assumed to be L2-normalized and live on the unit sphere S^{d-1}.
Semantic similarity is measured via geodesic distance / cosine angle.
"""

import numpy as np
from typing import List, Tuple
from rewa.models import Vector


def normalize_embedding(embedding: Vector) -> Vector:
    """L2-normalize an embedding to the unit sphere."""
    norm = np.linalg.norm(embedding)
    if norm < 1e-10:
        raise ValueError("Cannot normalize zero vector")
    return embedding / norm


def cosine_similarity(a: Vector, b: Vector) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(np.dot(a, b))


def angular_distance(a: Vector, b: Vector) -> float:
    """
    Compute angular (geodesic) distance between two points on the sphere.

    Returns angle in radians [0, pi].
    """
    similarity = np.dot(a, b)
    # Clamp to handle numerical errors
    similarity = np.clip(similarity, -1.0, 1.0)
    return float(np.arccos(similarity))


def geodesic_midpoint(a: Vector, b: Vector) -> Vector:
    """
    Compute the geodesic midpoint between two points on the sphere.

    This is the point halfway along the great circle connecting a and b.
    """
    # Simple approach: normalize the sum (works for non-antipodal points)
    midpoint = a + b
    return normalize_embedding(midpoint)


def geodesic_interpolate(a: Vector, b: Vector, t: float) -> Vector:
    """
    Spherical linear interpolation (slerp) between two points.

    Args:
        a: Start point (normalized)
        b: End point (normalized)
        t: Interpolation parameter [0, 1]

    Returns:
        Point on geodesic from a to b at fraction t
    """
    dot = np.dot(a, b)
    dot = np.clip(dot, -1.0, 1.0)

    theta = np.arccos(dot)

    if theta < 1e-10:
        # Points are very close, linear interpolation is fine
        return normalize_embedding((1 - t) * a + t * b)

    sin_theta = np.sin(theta)
    wa = np.sin((1 - t) * theta) / sin_theta
    wb = np.sin(t * theta) / sin_theta

    return wa * a + wb * b


def compute_centroid(embeddings: List[Vector]) -> Vector:
    """
    Compute the Fréchet mean (centroid) of embeddings on the sphere.

    Uses iterative algorithm for accurate spherical mean.
    """
    if not embeddings:
        raise ValueError("Cannot compute centroid of empty list")

    embeddings = np.array(embeddings)

    # Start with normalized sum as initial guess
    centroid = normalize_embedding(np.mean(embeddings, axis=0))

    # Iterative refinement (Weiszfeld algorithm on sphere)
    for _ in range(10):
        # Project all points to tangent space at current centroid
        tangent_mean = np.zeros_like(centroid)

        for emb in embeddings:
            # Log map: project to tangent space
            dot = np.dot(centroid, emb)
            dot = np.clip(dot, -1.0, 1.0)
            theta = np.arccos(dot)

            if theta > 1e-10:
                # Direction in tangent space
                direction = emb - dot * centroid
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                tangent_mean += theta * direction

        tangent_mean /= len(embeddings)

        # Exp map: move centroid in direction of mean
        step_size = np.linalg.norm(tangent_mean)
        if step_size < 1e-8:
            break

        direction = tangent_mean / step_size
        centroid = np.cos(step_size) * centroid + np.sin(step_size) * direction

    return centroid


def estimate_intrinsic_dimension(
    embeddings: List[Vector],
    k: int = 10,
    method: str = "mle"
) -> float:
    """
    Estimate the intrinsic dimensionality of a set of embeddings.

    Uses Maximum Likelihood Estimation (MLE) method.

    Args:
        embeddings: List of normalized embeddings
        k: Number of nearest neighbors to consider
        method: Estimation method ('mle' or 'correlation')

    Returns:
        Estimated intrinsic dimension
    """
    if len(embeddings) < k + 1:
        return float(len(embeddings[0]) if embeddings else 0)

    embeddings = np.array(embeddings)
    n_samples = len(embeddings)

    # Compute pairwise distances
    similarities = embeddings @ embeddings.T
    distances = np.arccos(np.clip(similarities, -1.0, 1.0))

    intrinsic_dims = []

    for i in range(n_samples):
        # Get k nearest neighbors (excluding self)
        dists = distances[i].copy()
        dists[i] = np.inf
        knn_indices = np.argsort(dists)[:k]
        knn_dists = dists[knn_indices]

        # Filter out zero distances
        knn_dists = knn_dists[knn_dists > 1e-10]

        if len(knn_dists) < 2:
            continue

        if method == "mle":
            # MLE estimate: d = 1 / mean(log(r_k / r_i))
            r_k = knn_dists[-1]
            log_ratios = np.log(r_k / knn_dists[:-1])
            if np.mean(log_ratios) > 1e-10:
                intrinsic_dims.append(1.0 / np.mean(log_ratios))

    if not intrinsic_dims:
        return float(len(embeddings[0]))

    return float(np.median(intrinsic_dims))


def compute_spread(embeddings: List[Vector], centroid: Vector) -> float:
    """
    Compute the angular spread of embeddings around a centroid.

    Returns the mean angular distance from centroid.
    """
    if not embeddings:
        return 0.0

    distances = [angular_distance(centroid, emb) for emb in embeddings]
    return float(np.mean(distances))


def compute_coverage_radius(
    embeddings: List[Vector],
    centroid: Vector,
    coverage: float = 0.95
) -> float:
    """
    Compute radius that covers a fraction of embeddings.

    Args:
        embeddings: List of embeddings
        centroid: Center point
        coverage: Fraction of embeddings to cover (default 95%)

    Returns:
        Angular radius covering the specified fraction
    """
    if not embeddings:
        return 0.0

    distances = sorted([angular_distance(centroid, emb) for emb in embeddings])
    index = int(coverage * len(distances))
    index = min(index, len(distances) - 1)

    return distances[index]


def points_in_cone(
    embeddings: List[Vector],
    apex: Vector,
    direction: Vector,
    half_angle: float
) -> List[int]:
    """
    Find all embeddings within a cone.

    Args:
        embeddings: List of embeddings to check
        apex: Apex of cone (ignored for unit sphere, cone from origin)
        direction: Central direction of cone (normalized)
        half_angle: Half-angle of cone in radians

    Returns:
        Indices of embeddings within the cone
    """
    direction = normalize_embedding(direction)
    cos_threshold = np.cos(half_angle)

    indices = []
    for i, emb in enumerate(embeddings):
        if np.dot(emb, direction) >= cos_threshold:
            indices.append(i)

    return indices


def tangent_space_projection(point: Vector, base: Vector) -> Vector:
    """
    Project a point onto the tangent space at base.

    Returns the tangent vector (log map).
    """
    dot = np.dot(base, point)
    dot = np.clip(dot, -1.0, 1.0)

    if dot > 1.0 - 1e-10:
        # Points are the same
        return np.zeros_like(point)

    theta = np.arccos(dot)
    direction = point - dot * base
    norm = np.linalg.norm(direction)

    if norm < 1e-10:
        return np.zeros_like(point)

    return theta * (direction / norm)


def exp_map(base: Vector, tangent: Vector) -> Vector:
    """
    Exponential map: move from base in direction of tangent vector.

    Args:
        base: Base point on sphere (normalized)
        tangent: Tangent vector at base

    Returns:
        New point on sphere
    """
    norm = np.linalg.norm(tangent)

    if norm < 1e-10:
        return base

    direction = tangent / norm
    return np.cos(norm) * base + np.sin(norm) * direction
