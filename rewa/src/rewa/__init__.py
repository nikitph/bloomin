"""
REWA - Reasoning & Validation Layer for AI Agents

A geometric, local-world, constraint-satisfaction system that decides
whether language should even be trusted.
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

__version__ = "0.1.0"
__all__ = [
    "REWA",
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
