"""Data package"""

from .synthetic import (
    SyntheticDocument,
    HierarchicalGaussianGenerator,
    CompositionalQAGenerator,
    GraphDistanceGenerator
)

__all__ = [
    'SyntheticDocument',
    'HierarchicalGaussianGenerator',
    'CompositionalQAGenerator',
    'GraphDistanceGenerator'
]
