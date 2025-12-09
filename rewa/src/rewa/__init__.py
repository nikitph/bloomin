"""
REWA - Reasoning & Validation Layer for AI Agents

A geometric, local-world, constraint-satisfaction system that decides
whether language should even be trusted.

Two implementations are available:
- REWA: Original regex-based implementation (legacy, brittle)
- SemanticREWA: Embedding-based implementation (recommended)
"""

from rewa.models import (
    Entity,
    Fact,
    Rule,
    Chart,
    QueryIntent,
    ValueConstraint,
    TimeInterval,
    LocalWorld,
    ValidationResult,
    Contradiction,
    Impossibility,
    RewaResponse,
    RewaStatus,
)
from rewa.api import REWA
from rewa.semantic_api import SemanticREWA, SemanticRewaConfig
from rewa.embeddings import Embedder, SemanticMatcher

__version__ = "0.1.0"
__all__ = [
    # Main APIs
    "REWA",  # Legacy regex-based
    "SemanticREWA",  # Recommended embedding-based
    "SemanticRewaConfig",
    # Embeddings
    "Embedder",
    "SemanticMatcher",
    # Data models
    "Entity",
    "Fact",
    "Rule",
    "Chart",
    "QueryIntent",
    "ValueConstraint",
    "TimeInterval",
    "LocalWorld",
    "ValidationResult",
    "Contradiction",
    "Impossibility",
    "RewaResponse",
    "RewaStatus",
]
