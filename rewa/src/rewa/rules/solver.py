"""
Constraint Solver

Validates entities against query constraints and domain rules.
Implements the core validation logic for REWA.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from rewa.models import (
    Entity,
    Fact,
    Rule,
    QueryIntent,
    ValueConstraint,
    ComparisonOp,
    ValidationResult,
)
from rewa.rules.engine import RuleEngine, RuleEvaluation


@dataclass
class ConstraintViolation:
    """Details of a constraint violation."""
    constraint_type: str  # "required", "forbidden", "rule"
    property_name: str
    expected: Any
    actual: Any
    rule_id: Optional[str] = None


class ConstraintSolver:
    """
    Validates entities against query intent and rules.

    Implements constraint satisfaction checking:
    1. Type requirements
    2. Required property constraints
    3. Forbidden property constraints
    4. Domain rule satisfaction
    """

    def __init__(self, rule_engine: Optional[RuleEngine] = None):
        """
        Initialize the constraint solver.

        Args:
            rule_engine: Optional rule engine for rule-based validation
        """
        self.rule_engine = rule_engine or RuleEngine()

    def validate_entity(
        self,
        entity: Entity,
        facts: List[Fact],
        intent: QueryIntent,
        domains: Optional[Set[str]] = None
    ) -> ValidationResult:
        """
        Validate an entity against query intent.

        Args:
            entity: Entity to validate
            facts: Facts about the entity
            intent: Query intent with requirements
            domains: Optional domains for rule filtering

        Returns:
            ValidationResult with satisfaction status
        """
        violations: List[str] = []
        proof_facts: List[Fact] = []
        confidence = entity.confidence

        # Check type requirements
        if intent.required_types:
            if entity.type not in intent.required_types:
                violations.append(
                    f"Type mismatch: entity is {entity.type}, "
                    f"required one of {intent.required_types}"
                )

        # Check required properties
        for prop_name, constraint in intent.required_properties.items():
            satisfied, fact = self._check_property_constraint(
                entity, facts, prop_name, constraint, required=True
            )
            if not satisfied:
                violations.append(
                    f"Required property '{prop_name}' not satisfied: "
                    f"expected {constraint.op.value} {constraint.value}"
                )
            elif fact:
                proof_facts.append(fact)

        # Check forbidden properties
        for prop_name, constraint in intent.forbidden_properties.items():
            violated, fact = self._check_property_constraint(
                entity, facts, prop_name, constraint, required=False
            )
            if violated:
                violations.append(
                    f"Forbidden property '{prop_name}' is present: "
                    f"has {constraint.op.value} {constraint.value}"
                )

        # Check domain rules
        rule_evaluations = self.rule_engine.evaluate_entity(
            entity, facts, domains
        )
        for eval_result in rule_evaluations:
            if not eval_result.is_satisfied:
                violations.append(
                    f"Rule '{eval_result.rule.name}' violated: "
                    f"{eval_result.violation_reason}"
                )
                # Reduce confidence for rule violations
                if eval_result.rule.hardness == "hard":
                    confidence *= 0.5
                else:
                    confidence *= 0.8

        # Calculate final confidence
        if violations:
            confidence *= (0.5 ** len(violations))

        return ValidationResult(
            entity=entity,
            satisfied=len(violations) == 0,
            proof=proof_facts,
            violations=violations,
            confidence=confidence,
        )

    def validate_entities(
        self,
        entities: List[Entity],
        facts: List[Fact],
        intent: QueryIntent,
        domains: Optional[Set[str]] = None
    ) -> List[ValidationResult]:
        """
        Validate multiple entities against query intent.

        Args:
            entities: Entities to validate
            facts: All facts
            intent: Query intent
            domains: Optional domains for rule filtering

        Returns:
            List of ValidationResults
        """
        results = []

        for entity in entities:
            entity_facts = [f for f in facts if f.subject.id == entity.id]
            result = self.validate_entity(entity, entity_facts, intent, domains)
            results.append(result)

        # Sort by satisfaction and confidence
        results.sort(
            key=lambda r: (r.satisfied, r.confidence),
            reverse=True
        )

        return results

    def find_satisfying_entities(
        self,
        entities: List[Entity],
        facts: List[Fact],
        intent: QueryIntent,
        domains: Optional[Set[str]] = None,
        min_confidence: float = 0.5
    ) -> List[ValidationResult]:
        """
        Find all entities that satisfy the query intent.

        Args:
            entities: Entities to check
            facts: All facts
            intent: Query intent
            domains: Optional domains
            min_confidence: Minimum confidence threshold

        Returns:
            List of satisfying ValidationResults
        """
        all_results = self.validate_entities(entities, facts, intent, domains)

        return [
            r for r in all_results
            if r.satisfied and r.confidence >= min_confidence
        ]

    def _check_property_constraint(
        self,
        entity: Entity,
        facts: List[Fact],
        prop_name: str,
        constraint: ValueConstraint,
        required: bool
    ) -> Tuple[bool, Optional[Fact]]:
        """
        Check if a property constraint is satisfied.

        Args:
            entity: Entity to check
            facts: Facts about the entity
            prop_name: Property name
            constraint: Value constraint
            required: If True, check if satisfied; if False, check if violated

        Returns:
            (satisfied_or_violated, supporting_fact)
        """
        # First check entity properties
        if entity.has_property(prop_name):
            value = entity.get_property(prop_name)
            satisfied = constraint.satisfied_by(value)
            return satisfied if required else not satisfied, None

        # Then check facts
        matching_facts = [
            f for f in facts
            if f.subject.id == entity.id and f.predicate == prop_name
        ]

        if not matching_facts:
            # No fact found
            if required:
                # Required property missing
                return False, None
            else:
                # Forbidden property not present - good
                return False, None

        # Check facts
        for fact in matching_facts:
            # Handle negated facts
            effective_value = not fact.value if fact.negated else fact.value
            satisfied = constraint.satisfied_by(effective_value)

            if required and satisfied:
                return True, fact
            elif not required and satisfied:
                return True, fact  # This is a violation of forbidden

        if required:
            return False, None
        else:
            return False, None


class IntentValidator:
    """
    Validates query intents for internal consistency.

    Checks for contradictions within the intent itself.
    """

    def __init__(self):
        self.contradiction_patterns: List[Tuple[str, str, str]] = [
            # (prop1, prop2, reason)
            ("dangerous", "safe", "Cannot require both dangerous and safe"),
            ("is_toy", "is_real", "Cannot require both toy and real"),
            ("cures", "causes", "Cannot require both curing and causing same condition"),
        ]

    def validate(self, intent: QueryIntent) -> List[str]:
        """
        Validate an intent for internal consistency.

        Returns:
            List of contradiction reasons (empty if consistent)
        """
        issues = []

        # Check required vs forbidden conflicts
        for prop in intent.required_properties:
            if prop in intent.forbidden_properties:
                req = intent.required_properties[prop]
                forb = intent.forbidden_properties[prop]
                if self._constraints_conflict(req, forb):
                    issues.append(
                        f"Property '{prop}' is both required ({req.value}) "
                        f"and forbidden ({forb.value})"
                    )

        # Check known contradiction patterns
        for prop1, prop2, reason in self.contradiction_patterns:
            if prop1 in intent.required_properties and prop2 in intent.required_properties:
                c1 = intent.required_properties[prop1]
                c2 = intent.required_properties[prop2]
                if c1.value == True and c2.value == True:
                    issues.append(reason)

        return issues

    def _constraints_conflict(
        self,
        c1: ValueConstraint,
        c2: ValueConstraint
    ) -> bool:
        """Check if two constraints conflict."""
        if c1.op == ComparisonOp.EQ and c2.op == ComparisonOp.EQ:
            return c1.value == c2.value

        if c1.op == ComparisonOp.EQ and c2.op == ComparisonOp.NEQ:
            return c1.value != c2.value

        return False
