"""
Rewa Core v1 - Policy-Driven Semantic Validation Engine

A middleware system that:
1. Ingests documents (evidence) and policies (biases)
2. Projects both into a shared semantic sphere
3. Computes the admissible semantic region from evidence
4. Applies explicit, logged policy bias (Mode B) when ambiguity exists
5. Generates or refuses output without hallucination

Primary Goal: Zero hallucinated approvals under ambiguous or contradictory evidence.
"""

__version__ = "1.0.0"
__author__ = "Rewa Team"

from .semantic_space import SemanticSpace, Witness
from .hemisphere import HemisphereChecker
from .hull import SphericalHull
from .entropy import EntropyEstimator
from .policy import Policy, PolicyEngine
from .mode_b import ModeBEngine
from .refusal import RefusalType, RefusalHandler
from .verbalization import VerbalizationGuard
from .audit import AuditLogger
from .core import RewaCore, RewaDecision, RewaState
