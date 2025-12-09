"""Geometry package"""

from .fisher import (
    FisherMetric,
    FisherGeometryEstimator,
    compute_scalar_curvature,
    estimate_intrinsic_dimension
)
from .diagnostics import GeometryDiagnostics

__all__ = [
    'FisherMetric',
    'FisherGeometryEstimator',
    'compute_scalar_curvature',
    'estimate_intrinsic_dimension',
    'GeometryDiagnostics'
]
