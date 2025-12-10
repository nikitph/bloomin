"""
FR-4: Hull Construction (Implicit)

Spherical convex hull computation for the admissible semantic region.
Uses geodesic convex combinations, not Euclidean shortcuts.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from .semantic_space import Witness


@dataclass
class SphericalHullResult:
    """Result of spherical hull computation."""
    witnesses: List[Witness]
    center: np.ndarray  # Geodesic center (Fréchet mean)
    angular_radius: float  # Maximum angle from center to any witness
    volume_proxy: float  # Approximation of hull "size"
    is_valid: bool
    details: Dict[str, Any]


class SphericalHull:
    """
    Computes the spherical convex hull of witnesses.

    The hull A(W) represents all possible meanings consistent with evidence W.
    Any point μ ∈ A(W) satisfies μ · w > 0 for all w ∈ W.

    We compute this implicitly via:
    1. Geodesic center (Fréchet mean on sphere)
    2. Angular spread metrics
    3. Sampling-based boundary estimation
    """

    def __init__(
        self,
        max_iterations: int = 100,
        convergence_threshold: float = 1e-6,
        num_boundary_samples: int = 100
    ):
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.num_boundary_samples = num_boundary_samples

    def compute(self, witnesses: List[Witness]) -> SphericalHullResult:
        """
        Compute the spherical convex hull of witnesses.

        Returns:
            SphericalHullResult with center, radius, and volume metrics.
        """
        if len(witnesses) == 0:
            return SphericalHullResult(
                witnesses=[],
                center=np.zeros(1),
                angular_radius=np.pi,
                volume_proxy=1.0,
                is_valid=False,
                details={"message": "No witnesses provided"}
            )

        W = np.array([w.embedding for w in witnesses])
        d = W.shape[1]

        if len(witnesses) == 1:
            return SphericalHullResult(
                witnesses=witnesses,
                center=W[0].copy(),
                angular_radius=0.0,
                volume_proxy=0.0,
                is_valid=True,
                details={"message": "Single witness - point hull"}
            )

        # Compute geodesic center (Fréchet mean)
        center = self._compute_frechet_mean(W)

        # Compute angular radius (max distance from center to any witness)
        angles = np.arccos(np.clip(W @ center, -1.0, 1.0))
        angular_radius = float(np.max(angles))

        # Compute volume proxy (based on angular spread)
        volume_proxy = self._estimate_volume(W, center, angles)

        return SphericalHullResult(
            witnesses=witnesses,
            center=center,
            angular_radius=angular_radius,
            volume_proxy=volume_proxy,
            is_valid=True,
            details={
                "num_witnesses": len(witnesses),
                "dimension": d,
                "mean_angle": float(np.mean(angles)),
                "std_angle": float(np.std(angles)),
                "angles": angles.tolist()
            }
        )

    def _compute_frechet_mean(self, W: np.ndarray) -> np.ndarray:
        """
        Compute the Fréchet mean (geodesic center) on the sphere.

        The Fréchet mean minimizes sum of squared geodesic distances.
        Uses iterative algorithm.
        """
        # Initialize with Euclidean mean, normalized
        center = np.mean(W, axis=0)
        center = center / (np.linalg.norm(center) + 1e-10)

        for iteration in range(self.max_iterations):
            # Compute tangent vectors (log map)
            # For small angles, tangent ≈ w - (w·c)c
            dots = W @ center
            tangents = W - np.outer(dots, center)

            # Average tangent direction
            mean_tangent = np.mean(tangents, axis=0)

            # Move along geodesic (exp map approximation)
            step_size = np.linalg.norm(mean_tangent)
            if step_size < self.convergence_threshold:
                break

            # Update center
            new_center = center + mean_tangent
            new_center = new_center / (np.linalg.norm(new_center) + 1e-10)

            center = new_center

        return center

    def _estimate_volume(
        self,
        W: np.ndarray,
        center: np.ndarray,
        angles: np.ndarray
    ) -> float:
        """
        Estimate the "volume" (surface area) of the spherical hull.

        Uses a combination of:
        1. Angular spread
        2. Witness dispersion
        3. Effective dimensionality
        """
        n, d = W.shape

        # Angular spread contribution
        max_angle = float(np.max(angles))
        mean_angle = float(np.mean(angles))

        # Spherical cap area ≈ 2π(1 - cos(θ)) for cap of angle θ
        cap_area = 2 * np.pi * (1 - np.cos(max_angle))

        # Dispersion: how spread out are witnesses from each other?
        if n > 1:
            pairwise_dots = W @ W.T
            np.fill_diagonal(pairwise_dots, 1.0)
            pairwise_angles = np.arccos(np.clip(pairwise_dots, -1.0, 1.0))
            dispersion = float(np.mean(pairwise_angles[np.triu_indices(n, k=1)]))
        else:
            dispersion = 0.0

        # Combine into volume proxy [0, 1]
        # Normalize by full sphere area 4π
        volume_proxy = cap_area / (4 * np.pi)

        # Adjust for dispersion
        volume_proxy *= (1 + dispersion / np.pi) / 2

        return float(np.clip(volume_proxy, 0.0, 1.0))

    def contains(
        self,
        point: np.ndarray,
        witnesses: List[Witness],
        strict: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if a point is inside the spherical convex hull.

        A point μ is in A(W) iff μ · w > 0 for all w ∈ W.

        Args:
            point: Point to check (will be normalized)
            witnesses: Witness list defining the hull
            strict: If True, require μ · w > 0. If False, allow μ · w >= 0.

        Returns:
            (is_inside, details)
        """
        if len(witnesses) == 0:
            return True, {"message": "Empty hull contains everything"}

        # Normalize point
        point = point / (np.linalg.norm(point) + 1e-10)

        W = np.array([w.embedding for w in witnesses])
        dots = W @ point

        threshold = 0.0 if strict else -1e-10
        is_inside = bool(np.all(dots > threshold))

        return is_inside, {
            "min_dot": float(np.min(dots)),
            "max_dot": float(np.max(dots)),
            "mean_dot": float(np.mean(dots)),
            "violations": [witnesses[i].id for i in range(len(witnesses))
                          if dots[i] <= threshold]
        }

    def sample_interior(
        self,
        witnesses: List[Witness],
        n_samples: int = 100,
        method: str = "rejection"
    ) -> np.ndarray:
        """
        Sample points from the interior of the spherical hull.

        Methods:
        - "rejection": Rejection sampling from sphere
        - "combination": Geodesic convex combinations of witnesses

        Returns:
            Array of shape (n_samples, d) with interior points.
        """
        if len(witnesses) == 0:
            raise ValueError("Cannot sample from empty hull")

        W = np.array([w.embedding for w in witnesses])
        d = W.shape[1]

        if method == "combination":
            return self._sample_combinations(W, n_samples)
        else:
            return self._sample_rejection(W, n_samples)

    def _sample_combinations(
        self,
        W: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        Sample via geodesic convex combinations.

        Any convex combination of witnesses, normalized, is in the hull.
        """
        n, d = W.shape
        samples = []

        for _ in range(n_samples):
            # Random convex weights
            weights = np.random.dirichlet(np.ones(n))

            # Weighted combination
            combination = weights @ W

            # Normalize to sphere
            combination = combination / (np.linalg.norm(combination) + 1e-10)
            samples.append(combination)

        return np.array(samples)

    def _sample_rejection(
        self,
        W: np.ndarray,
        n_samples: int,
        max_attempts: int = 10000
    ) -> np.ndarray:
        """
        Sample via rejection sampling.

        Sample uniformly on sphere, reject if outside hull.
        """
        n, d = W.shape
        samples = []
        attempts = 0

        while len(samples) < n_samples and attempts < max_attempts:
            # Sample uniformly on sphere
            point = np.random.randn(d)
            point = point / np.linalg.norm(point)

            # Check if in hull
            dots = W @ point
            if np.all(dots > 0):
                samples.append(point)

            attempts += 1

        if len(samples) < n_samples:
            # Fall back to combination sampling
            remaining = n_samples - len(samples)
            more_samples = self._sample_combinations(W, remaining)
            samples.extend(more_samples.tolist())

        return np.array(samples[:n_samples])

    def geodesic_interpolate(
        self,
        start: np.ndarray,
        end: np.ndarray,
        t: float
    ) -> np.ndarray:
        """
        Interpolate along geodesic between two points.

        Args:
            start: Starting point on sphere
            end: Ending point on sphere
            t: Interpolation parameter [0, 1]

        Returns:
            Point on geodesic at parameter t
        """
        # Normalize inputs
        start = start / (np.linalg.norm(start) + 1e-10)
        end = end / (np.linalg.norm(end) + 1e-10)

        # Compute angle between points
        dot = np.clip(np.dot(start, end), -1.0, 1.0)
        theta = np.arccos(dot)

        if theta < 1e-10:
            return start.copy()

        # Spherical linear interpolation (slerp)
        sin_theta = np.sin(theta)
        a = np.sin((1 - t) * theta) / sin_theta
        b = np.sin(t * theta) / sin_theta

        result = a * start + b * end
        return result / (np.linalg.norm(result) + 1e-10)
