"""
Core data models for REWA.

This module defines all the data structures used throughout the system:
- Semantic geometry (Charts, embeddings)
- Knowledge representation (Entities, Facts, Rules)
- Query representation (QueryIntent, constraints)
- Validation results
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Set, Union
import numpy as np


# ============================================================================
# Basic Types
# ============================================================================

Vector = np.ndarray  # Shape: (embedding_dim,)


class ComparisonOp(Enum):
    """Comparison operators for value constraints."""
    EQ = "eq"       # equals
    NEQ = "neq"     # not equals
    GT = "gt"       # greater than
    GTE = "gte"     # greater than or equal
    LT = "lt"       # less than
    LTE = "lte"     # less than or equal
    IN = "in"       # in set
    NOT_IN = "not_in"  # not in set
    CONTAINS = "contains"  # string contains
    MATCHES = "matches"    # regex match


@dataclass
class TimeInterval:
    """Represents a time interval for temporal facts."""
    start: Optional[datetime] = None
    end: Optional[datetime] = None

    def contains(self, point: datetime) -> bool:
        """Check if a point in time falls within this interval."""
        if self.start and point < self.start:
            return False
        if self.end and point > self.end:
            return False
        return True

    def overlaps(self, other: "TimeInterval") -> bool:
        """Check if two intervals overlap."""
        if self.end and other.start and self.end < other.start:
            return False
        if self.start and other.end and self.start > other.end:
            return False
        return True

    def is_after(self, other: "TimeInterval") -> bool:
        """Check if this interval is entirely after another."""
        if self.start is None or other.end is None:
            return False
        return self.start > other.end


@dataclass
class ValueConstraint:
    """Constraint on a property value."""
    op: ComparisonOp
    value: Any

    def satisfied_by(self, actual_value: Any) -> bool:
        """Check if an actual value satisfies this constraint."""
        try:
            if self.op == ComparisonOp.EQ:
                return actual_value == self.value
            elif self.op == ComparisonOp.NEQ:
                return actual_value != self.value
            elif self.op == ComparisonOp.GT:
                return actual_value > self.value
            elif self.op == ComparisonOp.GTE:
                return actual_value >= self.value
            elif self.op == ComparisonOp.LT:
                return actual_value < self.value
            elif self.op == ComparisonOp.LTE:
                return actual_value <= self.value
            elif self.op == ComparisonOp.IN:
                return actual_value in self.value
            elif self.op == ComparisonOp.NOT_IN:
                return actual_value not in self.value
            elif self.op == ComparisonOp.CONTAINS:
                return self.value in str(actual_value)
            elif self.op == ComparisonOp.MATCHES:
                import re
                return bool(re.match(self.value, str(actual_value)))
        except (TypeError, ValueError):
            return False
        return False


# ============================================================================
# Semantic Geometry
# ============================================================================

@dataclass
class Chart:
    """
    A local semantic neighborhood on the embedding sphere.

    Charts approximate local semantic worlds. Every document belongs to
    at least one chart, and overlaps are where contradictions surface.
    """
    id: str
    witness_embedding: Vector  # L2-normalized center point
    radius: float              # Angular radius in radians
    intrinsic_dim: int         # Local intrinsic dimensionality
    domain_tags: Set[str] = field(default_factory=set)
    description: str = ""

    def contains_embedding(self, embedding: Vector, threshold: Optional[float] = None) -> bool:
        """Check if an embedding falls within this chart."""
        threshold = threshold if threshold is not None else self.radius
        similarity = np.dot(self.witness_embedding, embedding)
        # Convert cosine similarity to angular distance
        angular_dist = np.arccos(np.clip(similarity, -1.0, 1.0))
        return angular_dist <= threshold

    def similarity_to(self, embedding: Vector) -> float:
        """Compute cosine similarity to an embedding."""
        return float(np.dot(self.witness_embedding, embedding))

    def angular_distance_to(self, embedding: Vector) -> float:
        """Compute angular distance to an embedding."""
        similarity = np.dot(self.witness_embedding, embedding)
        return float(np.arccos(np.clip(similarity, -1.0, 1.0)))


# ============================================================================
# Knowledge Representation
# ============================================================================

@dataclass
class Entity:
    """
    A semantic entity extracted from text.

    Represents things like: Gun, Toy, Drug, Person, etc.
    """
    id: str
    type: str
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    source_chunk_id: Optional[str] = None
    embedding: Optional[Vector] = None

    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property value."""
        return self.properties.get(key, default)

    def has_property(self, key: str) -> bool:
        """Check if entity has a property."""
        return key in self.properties


@dataclass
class Fact:
    """
    A factual assertion about an entity.

    Represents statements like: "Gun X is dangerous" or "Drug Y cures cancer"
    """
    id: str
    subject: Entity
    predicate: str
    value: Any
    valid_time: Optional[TimeInterval] = None
    confidence: float = 1.0
    source_chunk_id: Optional[str] = None
    negated: bool = False

    def conflicts_with(self, other: "Fact") -> bool:
        """Check if this fact conflicts with another."""
        # Must be about same subject and predicate
        if self.subject.id != other.subject.id:
            return False
        if self.predicate != other.predicate:
            return False

        # Check temporal overlap
        if self.valid_time and other.valid_time:
            if not self.valid_time.overlaps(other.valid_time):
                return False

        # Check value conflict
        if self.negated != other.negated:
            if self.value == other.value:
                return True
        elif self.value != other.value:
            # Same assertion type but different values
            return True

        return False

    def supersedes(self, other: "Fact") -> bool:
        """Check if this fact supersedes another (temporal precedence)."""
        if not self.valid_time or not other.valid_time:
            return False
        return self.valid_time.is_after(other.valid_time)


@dataclass
class RuleConstraint:
    """A constraint within a rule."""
    property_path: str  # e.g., "subject.type" or "value"
    constraint: ValueConstraint


@dataclass
class Rule:
    """
    A domain rule expressing logical constraints.

    Rules can be hard (must be satisfied) or soft (preference).
    """
    id: str
    name: str
    description: str
    preconditions: List[RuleConstraint]
    postconditions: List[RuleConstraint]
    hardness: Literal["hard", "soft"] = "hard"
    domain_tags: Set[str] = field(default_factory=set)
    explanation_template: str = ""

    def applies_to(self, entity: Entity, facts: List[Fact]) -> bool:
        """Check if this rule's preconditions apply to an entity."""
        for precond in self.preconditions:
            if not self._check_constraint(precond, entity, facts):
                return False
        return True

    def is_satisfied(self, entity: Entity, facts: List[Fact]) -> bool:
        """Check if the rule is satisfied (postconditions met)."""
        for postcond in self.postconditions:
            if not self._check_constraint(postcond, entity, facts):
                return False
        return True

    def _check_constraint(
        self,
        rule_constraint: RuleConstraint,
        entity: Entity,
        facts: List[Fact]
    ) -> bool:
        """Check a single constraint against entity/facts."""
        path = rule_constraint.property_path
        constraint = rule_constraint.constraint

        # Check entity properties
        if path.startswith("entity."):
            prop_name = path[7:]
            if prop_name == "type":
                return constraint.satisfied_by(entity.type)
            return constraint.satisfied_by(entity.get_property(prop_name))

        # Check facts
        if path.startswith("fact."):
            predicate = path[5:]
            entity_facts = [f for f in facts if f.subject.id == entity.id and f.predicate == predicate]
            if not entity_facts:
                # No fact found - constraint on non-existent value
                return constraint.op == ComparisonOp.NEQ or constraint.op == ComparisonOp.NOT_IN
            return any(constraint.satisfied_by(f.value) for f in entity_facts)

        return False

    def explain_violation(self, entity: Entity, facts: List[Fact]) -> str:
        """Generate explanation for why rule is violated."""
        if self.explanation_template:
            return self.explanation_template.format(entity=entity, facts=facts)
        return f"Rule '{self.name}' violated: {self.description}"


# ============================================================================
# Query Representation
# ============================================================================

class TemporalPolicy(Enum):
    """Policy for handling temporal facts."""
    LATEST = "latest"       # Use most recent fact
    ALL = "all"             # Consider all facts
    AT_TIME = "at_time"     # Facts valid at specific time
    RANGE = "range"         # Facts valid in range


@dataclass
class QueryIntent:
    """
    Compiled representation of a user query.

    Captures the semantic intent including required types,
    property constraints, and temporal assumptions.
    """
    id: str = ""
    raw_query: str = ""
    embedding: Optional[Vector] = None
    required_types: Set[str] = field(default_factory=set)
    required_properties: Dict[str, ValueConstraint] = field(default_factory=dict)
    forbidden_properties: Dict[str, ValueConstraint] = field(default_factory=dict)
    temporal_policy: TemporalPolicy = TemporalPolicy.LATEST
    temporal_reference: Optional[datetime] = None
    temporal_range: Optional[TimeInterval] = None


# ============================================================================
# Local World Model
# ============================================================================

@dataclass
class LocalWorld:
    """
    A local world model constructed from retrieved chunks within a chart.

    Contains entities, facts, and applicable rules for reasoning.
    """
    chart: Chart
    entities: List[Entity]
    facts: List[Fact]
    rules: List[Rule]
    chunk_ids: Set[str] = field(default_factory=set)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        for e in self.entities:
            if e.id == entity_id:
                return e
        return None

    def get_facts_for_entity(self, entity_id: str) -> List[Fact]:
        """Get all facts about an entity."""
        return [f for f in self.facts if f.subject.id == entity_id]

    def get_facts_by_predicate(self, predicate: str) -> List[Fact]:
        """Get all facts with a given predicate."""
        return [f for f in self.facts if f.predicate == predicate]

    def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a given type."""
        return [e for e in self.entities if e.type == entity_type]


# ============================================================================
# Validation Results
# ============================================================================

class RewaStatus(Enum):
    """Status codes for REWA validation."""
    VALID = "VALID"
    CONFLICT = "CONFLICT"
    IMPOSSIBLE = "IMPOSSIBLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    AMBIGUOUS = "AMBIGUOUS"


@dataclass
class Contradiction:
    """Represents a detected contradiction between facts."""
    fact_a: Fact
    fact_b: Fact
    resolution: Optional[str] = None
    resolved_fact: Optional[Fact] = None
    explanation: str = ""


@dataclass
class Impossibility:
    """Represents a logically impossible query."""
    reason: str
    violated_rule: Optional[Rule] = None
    explanation: str = ""


@dataclass
class ValidationResult:
    """Result of validating a single entity against query intent."""
    entity: Entity
    satisfied: bool
    proof: List[Fact] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class RewaResponse:
    """
    The single response returned by REWA to the agent.

    This is the main integration contract - agents receive this
    structured response and decide how to proceed.
    """
    status: RewaStatus
    safe_facts: List[Fact]
    contradictions: List[Contradiction] = field(default_factory=list)
    impossibilities: List[Impossibility] = field(default_factory=list)
    ambiguous_charts: List[Chart] = field(default_factory=list)
    validation_results: List[ValidationResult] = field(default_factory=list)
    explanation: str = ""
    confidence: float = 1.0

    @property
    def is_valid(self) -> bool:
        return self.status == RewaStatus.VALID

    @property
    def has_conflicts(self) -> bool:
        return len(self.contradictions) > 0

    @property
    def is_impossible(self) -> bool:
        return self.status == RewaStatus.IMPOSSIBLE

    @property
    def is_ambiguous(self) -> bool:
        return self.status == RewaStatus.AMBIGUOUS
