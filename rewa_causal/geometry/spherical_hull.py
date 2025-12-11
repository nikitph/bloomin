"""
Spherical Convex Hull Operations

Integrates with rewa_core's existing SphericalHull and HemisphereChecker implementations
while providing the REWA-Causal API.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class SphericalConvexHull:
    """
    A spherical convex hull on S^{d-1}.

    The hull is valid only if all points satisfy the hemisphere constraint:
    all pairwise cosine similarities > -threshold (typically -0.3).

    Integrates with rewa_core's SphericalHull when available.
    """
    points: np.ndarray
    centroid: np.ndarray = field(default=None)
    radius: float = field(default=0.0)
    is_valid: bool = field(default=True)
    violation_pairs: List[Tuple[int, int]] = field(default_factory=list)

    def __post_init__(self):
        """Compute hull properties."""
        if self.points is None or len(self.points) == 0:
            self.points = np.zeros((0, 1))
            self.centroid = np.zeros(1)
            self.radius = 0.0
            self.is_valid = True
            return

        # Normalize points
        norms = np.linalg.norm(self.points, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        self.points = self.points / norms

        # Compute centroid (Fréchet mean)
        if self.centroid is None:
            self.centroid = self._compute_frechet_mean()

        # Compute radius
        self.radius = self._compute_radius()

        # Check hemisphere constraint
        self.is_valid, self.violation_pairs = self._check_hemisphere()

    @property
    def dimension(self) -> int:
        """Embedding dimension d."""
        return self.points.shape[1] if len(self.points.shape) > 1 else 1

    @property
    def n_points(self) -> int:
        """Number of points in the hull."""
        return self.points.shape[0]

    def _compute_frechet_mean(self, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """
        Compute Fréchet mean on the sphere using iterative algorithm.
        Compatible with rewa_core's implementation.
        """
        if self.n_points == 0:
            return np.zeros(self.dimension)

        if self.n_points == 1:
            return self.points[0].copy()

        # Initialize with Euclidean mean projected to sphere
        mu = np.mean(self.points, axis=0)
        norm = np.linalg.norm(mu)
        if norm < 1e-10:
            mu = self.points[0].copy()
        else:
            mu = mu / norm

        for _ in range(max_iter):
            # Compute tangent vectors
            dots = np.clip(self.points @ mu, -1.0, 1.0)
            angles = np.arccos(dots)

            # Compute unit tangent directions
            tangent_dirs = self.points - np.outer(dots, mu)
            tangent_norms = np.linalg.norm(tangent_dirs, axis=1, keepdims=True)
            tangent_norms = np.where(tangent_norms > 1e-10, tangent_norms, 1.0)
            tangent_dirs = tangent_dirs / tangent_norms

            # Mean tangent vector
            mean_tangent = np.mean(angles[:, np.newaxis] * tangent_dirs, axis=0)
            step_size = np.linalg.norm(mean_tangent)

            if step_size < tol:
                break

            # Exponential map
            direction = mean_tangent / step_size
            mu_new = mu * np.cos(step_size) + direction * np.sin(step_size)
            mu_new = mu_new / np.linalg.norm(mu_new)
            mu = mu_new

        return mu

    def _compute_radius(self) -> float:
        """Compute angular radius (max geodesic distance from centroid)."""
        if self.n_points == 0:
            return 0.0

        dots = np.clip(self.points @ self.centroid, -1.0, 1.0)
        angles = np.arccos(dots)
        return float(np.max(angles))

    def _check_hemisphere(self, threshold: float = -0.3) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        Check if all points lie within a hemisphere.
        Uses rewa_core's threshold convention.
        """
        if self.n_points < 2:
            return True, []

        similarities = self.points @ self.points.T
        violations = []

        for i in range(self.n_points):
            for j in range(i + 1, self.n_points):
                if similarities[i, j] < threshold:
                    violations.append((i, j))

        return len(violations) == 0, violations

    def contains_point(self, p: np.ndarray, margin: float = 0.0) -> bool:
        """Check if a point is inside the spherical convex hull."""
        p = p / np.linalg.norm(p)

        # A point is in the hull if it has positive dot product with all hull points
        # This is the definition from rewa_core
        dots = self.points @ p
        return bool(np.all(dots > -margin))

    def geodesic_distance_to_centroid(self, p: np.ndarray) -> float:
        """Compute geodesic distance from point to centroid."""
        p = p / np.linalg.norm(p)
        dot = np.clip(p @ self.centroid, -1.0, 1.0)
        return float(np.arccos(dot))

    def sample_interior(self, n_samples: int) -> np.ndarray:
        """Sample points from the interior using convex combinations (rewa_core method)."""
        if self.n_points == 0:
            return np.zeros((0, self.dimension))

        samples = []
        for _ in range(n_samples):
            # Random convex weights (Dirichlet)
            weights = np.random.dirichlet(np.ones(self.n_points))
            combination = weights @ self.points
            combination = combination / (np.linalg.norm(combination) + 1e-10)
            samples.append(combination)

        return np.array(samples)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'points': self.points.tolist(),
            'centroid': self.centroid.tolist(),
            'radius': self.radius,
            'is_valid': self.is_valid,
            'dimension': self.dimension,
            'n_points': self.n_points,
            'violation_pairs': self.violation_pairs
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SphericalConvexHull':
        """Create from dictionary."""
        return cls(
            points=np.array(data['points']),
            centroid=np.array(data['centroid']) if data.get('centroid') else None
        )

    def __repr__(self) -> str:
        valid_str = "valid" if self.is_valid else "INVALID"
        return f"SphericalConvexHull(n={self.n_points}, d={self.dimension}, r={self.radius:.4f}, {valid_str})"


def spherical_convex_hull(witnesses: np.ndarray) -> SphericalConvexHull:
    """
    Compute the spherical convex hull of witness vectors.

    Args:
        witnesses: Array of vectors, shape (n, d). Will be normalized to unit sphere.

    Returns:
        SphericalConvexHull object
    """
    witnesses = np.atleast_2d(witnesses)
    return SphericalConvexHull(points=witnesses)


def satisfies_hemisphere_constraint(
    witnesses: np.ndarray,
    threshold: float = -0.3
) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Check if witness set satisfies the hemisphere constraint.

    Uses the same threshold convention as rewa_core's HemisphereChecker.

    Args:
        witnesses: Array of unit vectors, shape (n, d)
        threshold: Minimum allowed pairwise similarity (default -0.3)

    Returns:
        (is_valid, violations): Boolean and list of violating point pairs
    """
    witnesses = np.atleast_2d(witnesses)
    norms = np.linalg.norm(witnesses, axis=1, keepdims=True)
    norms = np.where(norms > 1e-10, norms, 1.0)
    witnesses = witnesses / norms

    n = witnesses.shape[0]
    if n < 2:
        return True, []

    similarities = witnesses @ witnesses.T
    violations = []

    for i in range(n):
        for j in range(i + 1, n):
            if similarities[i, j] < threshold:
                violations.append((i, j))

    return len(violations) == 0, violations
