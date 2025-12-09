"""
Rule Engine Module

Handles domain rules and constraint satisfaction:
- Rule loading and management
- Constraint evaluation
- Rule-based impossibility detection
"""

from rewa.rules.engine import RuleEngine
from rewa.rules.loader import RuleLoader, load_rules_from_yaml
from rewa.rules.solver import ConstraintSolver

__all__ = [
    "RuleEngine",
    "RuleLoader",
    "load_rules_from_yaml",
    "ConstraintSolver",
]
