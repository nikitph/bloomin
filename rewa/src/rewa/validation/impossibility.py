"""
Impossibility Checker

Checks for logically impossible queries and entity configurations.
Implements hard constraint checking against domain rules.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from rewa.models import (
    QueryIntent,
    Entity,
    Fact,
    Rule,
    Impossibility,
    ValueConstraint,
    ComparisonOp,
)
from rewa.rules.engine import RuleEngine


@dataclass
class ImpossibilityPattern:
    """Pattern for detecting impossible configurations."""
    name: str
    description: str
    check_function: Callable[[QueryIntent], bool]
    explanation: str


class ImpossibilityChecker:
    """
    Checks for logical impossibilities.

    Implements the impossibility detection from PRD:
    - Pattern-based detection (known impossibilities)
    - Rule-based detection (domain constraints)
    - Structural detection (self-contradictions)
    """

    def __init__(self, rule_engine: Optional[RuleEngine] = None):
        """
        Initialize the checker.

        Args:
            rule_engine: Optional rule engine for rule-based checking
        """
        self.rule_engine = rule_engine or RuleEngine()
        self.patterns: List[ImpossibilityPattern] = []
        self._setup_default_patterns()

    def _setup_default_patterns(self):
        """Setup default impossibility patterns."""
        # Medical impossibilities
        self.add_pattern(ImpossibilityPattern(
            name="cancer_cure_no_side_effects",
            description="Cancer cure without side effects",
            check_function=self._check_cancer_cure_no_side_effects,
            explanation="Any effective cancer treatment will have side effects. "
                       "This is a fundamental reality of oncology.",
        ))

        # Physical impossibilities
        self.add_pattern(ImpossibilityPattern(
            name="perpetual_motion",
            description="Perpetual motion machine",
            check_function=self._check_perpetual_motion,
            explanation="Perpetual motion violates the laws of thermodynamics. "
                       "No machine can create energy from nothing.",
        ))

        self.add_pattern(ImpossibilityPattern(
            name="infinite_energy",
            description="Infinite energy source",
            check_function=self._check_infinite_energy,
            explanation="Infinite energy violates conservation of energy. "
                       "All energy sources are finite.",
        ))

        # Logical impossibilities
        self.add_pattern(ImpossibilityPattern(
            name="square_circle",
            description="Square circle",
            check_function=self._check_square_circle,
            explanation="A square circle is a logical contradiction. "
                       "An object cannot be both square and circular.",
        ))

        # Safety impossibilities
        self.add_pattern(ImpossibilityPattern(
            name="safe_and_dangerous",
            description="Both safe and dangerous",
            check_function=self._check_safe_and_dangerous,
            explanation="An item cannot be both safe and dangerous. "
                       "These properties are mutually exclusive.",
        ))

        self.add_pattern(ImpossibilityPattern(
            name="toy_weapon",
            description="Real weapon that is a toy",
            check_function=self._check_toy_weapon,
            explanation="A real weapon cannot be classified as a toy. "
                       "Toys must be safe by definition.",
        ))

    def add_pattern(self, pattern: ImpossibilityPattern) -> None:
        """Add an impossibility pattern."""
        self.patterns.append(pattern)

    def check(
        self,
        intent: QueryIntent,
        domains: Optional[Set[str]] = None
    ) -> List[Impossibility]:
        """
        Check if a query intent is impossible.

        Args:
            intent: Query intent to check
            domains: Optional domains for rule filtering

        Returns:
            List of impossibilities found
        """
        impossibilities = []

        # Pattern-based checking
        for pattern in self.patterns:
            if pattern.check_function(intent):
                impossibilities.append(Impossibility(
                    reason=pattern.description,
                    violated_rule=None,
                    explanation=pattern.explanation,
                ))

        # Structural checking (self-contradictions)
        structural = self._check_structural(intent)
        impossibilities.extend(structural)

        # Rule-based checking
        if domains:
            rule_based = self.rule_engine.check_impossibility(intent, domains)
            impossibilities.extend(rule_based)

        return impossibilities

    def check_entity(
        self,
        entity: Entity,
        facts: List[Fact],
        domains: Optional[Set[str]] = None
    ) -> List[Impossibility]:
        """
        Check if an entity configuration is impossible.

        Args:
            entity: Entity to check
            facts: Facts about the entity
            domains: Optional domains

        Returns:
            List of impossibilities
        """
        impossibilities = []

        # Check for contradictory facts
        contradictory = self._check_contradictory_facts(facts)
        impossibilities.extend(contradictory)

        # Rule-based entity checking
        if domains:
            evaluations = self.rule_engine.evaluate_entity(entity, facts, domains)
            for eval_result in evaluations:
                if not eval_result.is_satisfied and eval_result.rule.hardness == "hard":
                    impossibilities.append(Impossibility(
                        reason=f"Violates rule: {eval_result.rule.name}",
                        violated_rule=eval_result.rule,
                        explanation=eval_result.violation_reason or "",
                    ))

        return impossibilities

    def _check_structural(self, intent: QueryIntent) -> List[Impossibility]:
        """Check for structural impossibilities in intent."""
        impossibilities = []

        # Check for direct contradictions
        for req_prop, req_constraint in intent.required_properties.items():
            if req_prop in intent.forbidden_properties:
                forb_constraint = intent.forbidden_properties[req_prop]
                if self._constraints_contradict(req_constraint, forb_constraint):
                    impossibilities.append(Impossibility(
                        reason=f"Property '{req_prop}' is both required and forbidden",
                        explanation=f"Cannot require {req_prop}={req_constraint.value} "
                                   f"while forbidding {req_prop}={forb_constraint.value}",
                    ))

        return impossibilities

    def _check_contradictory_facts(self, facts: List[Fact]) -> List[Impossibility]:
        """Check for logically contradictory fact combinations."""
        impossibilities = []

        # Group by predicate
        by_predicate: Dict[str, List[Fact]] = {}
        for fact in facts:
            key = f"{fact.subject.id}:{fact.predicate}"
            if key not in by_predicate:
                by_predicate[key] = []
            by_predicate[key].append(fact)

        # Check each group
        for key, group in by_predicate.items():
            if len(group) < 2:
                continue

            for i, f1 in enumerate(group):
                for f2 in group[i + 1:]:
                    if self._facts_contradict(f1, f2):
                        impossibilities.append(Impossibility(
                            reason=f"Contradictory facts about {f1.predicate}",
                            explanation=f"Fact '{f1.predicate}={f1.value}' "
                                       f"contradicts '{f2.predicate}={f2.value}'",
                        ))

        return impossibilities

    def _facts_contradict(self, f1: Fact, f2: Fact) -> bool:
        """Check if two facts directly contradict."""
        if f1.negated != f2.negated:
            # One negated, one not - contradiction if same value
            return f1.value == f2.value

        # Both same polarity - contradiction if different boolean values
        if isinstance(f1.value, bool) and isinstance(f2.value, bool):
            return f1.value != f2.value

        return False

    def _constraints_contradict(
        self,
        c1: ValueConstraint,
        c2: ValueConstraint
    ) -> bool:
        """Check if two constraints contradict."""
        # EQ vs EQ with same value
        if c1.op == ComparisonOp.EQ and c2.op == ComparisonOp.EQ:
            return c1.value == c2.value

        # EQ vs NEQ
        if c1.op == ComparisonOp.EQ and c2.op == ComparisonOp.NEQ:
            return c1.value != c2.value

        return False

    # ==================== Pattern Check Functions ====================

    def _check_cancer_cure_no_side_effects(self, intent: QueryIntent) -> bool:
        """Check for cancer cure without side effects impossibility."""
        raw_query = intent.raw_query.lower()

        # Check if query mentions cancer (cure/treatment context)
        cures_cancer = False

        # Check raw query for cancer-related cure/treatment
        if "cancer" in raw_query:
            if any(word in raw_query for word in ["cure", "cures", "treat", "treats", "heal", "heals", "treatment"]):
                cures_cancer = True

        # Check property-based detection
        if "cures" in intent.required_properties:
            constraint = intent.required_properties["cures"]
            if "cancer" in str(constraint.value).lower():
                cures_cancer = True

        if not cures_cancer:
            return False

        # Must require no side effects
        no_side_effects = False

        # Check raw query
        if any(phrase in raw_query for phrase in ["no side effect", "zero side effect", "without side effect"]):
            no_side_effects = True

        # Check properties
        if "has_side_effects" in intent.required_properties:
            constraint = intent.required_properties["has_side_effects"]
            if constraint.value == False:
                no_side_effects = True

        if "has_side_effects" in intent.forbidden_properties:
            no_side_effects = True

        return cures_cancer and no_side_effects

    def _check_perpetual_motion(self, intent: QueryIntent) -> bool:
        """Check for perpetual motion impossibility."""
        perpetual = False

        # Check raw query
        if "perpetual motion" in intent.raw_query.lower():
            perpetual = True

        # Check required properties
        if "perpetual_motion" in intent.required_properties:
            constraint = intent.required_properties["perpetual_motion"]
            if constraint.value == True:
                perpetual = True

        return perpetual

    def _check_infinite_energy(self, intent: QueryIntent) -> bool:
        """Check for infinite energy impossibility."""
        infinite = False

        if "infinite energy" in intent.raw_query.lower():
            infinite = True

        if "creates_energy_from_nothing" in intent.required_properties:
            constraint = intent.required_properties["creates_energy_from_nothing"]
            if constraint.value == True:
                infinite = True

        return infinite

    def _check_square_circle(self, intent: QueryIntent) -> bool:
        """Check for square circle impossibility."""
        return "square circle" in intent.raw_query.lower()

    def _check_safe_and_dangerous(self, intent: QueryIntent) -> bool:
        """Check for safe AND dangerous impossibility."""
        requires_safe = False
        requires_dangerous = False

        if "safe" in intent.required_properties:
            if intent.required_properties["safe"].value == True:
                requires_safe = True

        if "dangerous" in intent.required_properties:
            if intent.required_properties["dangerous"].value == True:
                requires_dangerous = True

        return requires_safe and requires_dangerous

    def _check_toy_weapon(self, intent: QueryIntent) -> bool:
        """Check for real weapon toy impossibility."""
        requires_real = "is_real" in intent.required_properties
        requires_weapon = "Weapon" in intent.required_types
        requires_toy = "is_toy" in intent.required_properties

        if requires_toy and intent.required_properties.get("is_toy"):
            if intent.required_properties["is_toy"].value == True:
                if requires_real and requires_weapon:
                    return True

        return False


def check_impossibility(
    intent: QueryIntent,
    domains: Optional[Set[str]] = None,
    rule_engine: Optional[RuleEngine] = None
) -> List[Impossibility]:
    """
    Convenience function to check impossibility.

    Args:
        intent: Query intent to check
        domains: Optional domains for rule filtering
        rule_engine: Optional rule engine

    Returns:
        List of impossibilities
    """
    checker = ImpossibilityChecker(rule_engine)
    return checker.check(intent, domains)
