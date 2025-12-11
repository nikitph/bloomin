"""
Geometric Layer (REWA Core)

Spherical geometry operations for causal inference:
- Spherical convex hulls
- Hemisphere constraint checking
- Fréchet mean computation
- Region dispersion metrics
- Geodesic distances and hull operations
"""

from .spherical_hull import (
    SphericalConvexHull,
    spherical_convex_hull,
    satisfies_hemisphere_constraint
)
from .metrics import (
    frechet_mean,
    compute_dispersion,
    geodesic_distance,
    hull_overlap,
    weighted_union
)

__all__ = [
    'SphericalConvexHull',
    'spherical_convex_hull',
    'satisfies_hemisphere_constraint',
    'frechet_mean',
    'compute_dispersion',
    'geodesic_distance',
    'hull_overlap',
    'weighted_union'
]
