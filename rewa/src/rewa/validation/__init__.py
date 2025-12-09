"""
Validation Module

Handles contradiction detection and impossibility checking:
- Fact contradiction detection
- Temporal conflict resolution
- Logical impossibility detection
"""

from rewa.validation.contradiction import (
    ContradictionDetector,
    detect_contradictions,
)
from rewa.validation.impossibility import (
    ImpossibilityChecker,
    check_impossibility,
)

__all__ = [
    "ContradictionDetector",
    "detect_contradictions",
    "ImpossibilityChecker",
    "check_impossibility",
]
