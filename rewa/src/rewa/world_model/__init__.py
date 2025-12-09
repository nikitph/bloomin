"""
World Model Module

Constructs local world models from retrieved chunks:
- Entity extraction
- Fact construction
- Rule application
- World model assembly
"""

from rewa.world_model.builder import WorldModelBuilder
from rewa.world_model.merger import WorldModelMerger

__all__ = [
    "WorldModelBuilder",
    "WorldModelMerger",
]
