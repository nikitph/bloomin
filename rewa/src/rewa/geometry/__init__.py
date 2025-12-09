"""
Semantic Geometry Module

Handles all spherical geometry operations:
- Chart management
- Embedding similarity calculations
- Chart selection based on query embeddings
- Overlap detection for ambiguity
"""

from rewa.geometry.spherical import (
    normalize_embedding,
    cosine_similarity,
    angular_distance,
    geodesic_midpoint,
    geodesic_interpolate,
    estimate_intrinsic_dimension,
)
from rewa.geometry.charts import (
    ChartManager,
    create_chart_from_embeddings,
    detect_chart_overlap,
)
from rewa.geometry.selector import ChartSelector

__all__ = [
    "normalize_embedding",
    "cosine_similarity",
    "angular_distance",
    "geodesic_midpoint",
    "geodesic_interpolate",
    "estimate_intrinsic_dimension",
    "ChartManager",
    "create_chart_from_embeddings",
    "detect_chart_overlap",
    "ChartSelector",
]
