"""
Rule Engine

Evaluates domain rules against entities and facts.
Supports hard constraints (must be satisfied) and soft constraints (preferences).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from rewa.models import (
    Entity,
    Fact,
    Rule,
    RuleConstraint,
    ValueConstraint,
    ComparisonOp,
    QueryIntent,
    Impossibility,
)


@dataclass
class RuleEvaluation:
    """Result of evaluating a rule against an entity."""
    rule: Rule
    entity: Entity
    preconditions_met: bool
    postconditions_met: bool
    is_satisfied: bool
    violation_reason: Optional[str] = None


class RuleEngine:
    """
    Evaluates domain rules against entities and facts.

    The engine checks:
    1. Whether rules apply to given entities (preconditions)
    2. Whether postconditions are satisfied
    3. Hard vs soft constraint handling
    """

    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.domain_rules: Dict[str, Set[str]] = {}  # domain -> rule_ids

    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the engine."""
        self.rules[rule.id] = rule

        for domain in rule.domain_tags:
            if domain not in self.domain_rules:
                self.domain_rules[domain] = set()
            self.domain_rules[domain].add(rule.id)

    def add_rules(self, rules: List[Rule]) -> None:
        """Add multiple rules."""
        for rule in rules:
            self.add_rule(rule)

    def get_rules_for_domain(self, domain: str) -> List[Rule]:
        """Get all rules applicable to a domain."""
        rule_ids = self.domain_rules.get(domain, set())
        return [self.rules[rid] for rid in rule_ids if rid in self.rules]

    def get_rules_for_domains(self, domains: Set[str]) -> List[Rule]:
        """Get all rules applicable to any of the given domains."""
        rule_ids: Set[str] = set()
        for domain in domains:
            rule_ids.update(self.domain_rules.get(domain, set()))

        return [self.rules[rid] for rid in rule_ids if rid in self.rules]

    def evaluate_rule(
        self,
        rule: Rule,
        entity: Entity,
        facts: List[Fact]
    ) -> RuleEvaluation:
        """
        Evaluate a single rule against an entity.

        Args:
            rule: Rule to evaluate
            entity: Entity to check
            facts: Facts about the entity

        Returns:
            RuleEvaluation with results
        """
        # Check preconditions
        preconditions_met = self._check_preconditions(rule, entity, facts)

        if not preconditions_met:
            # Rule doesn't apply to this entity
            return RuleEvaluation(
                rule=rule,
                entity=entity,
                preconditions_met=False,
                postconditions_met=True,  # Vacuously true
                is_satisfied=True,
            )

        # Check postconditions
        postconditions_met, violation = self._check_postconditions(
            rule, entity, facts
        )

        is_satisfied = postconditions_met or rule.hardness == "soft"

        return RuleEvaluation(
            rule=rule,
            entity=entity,
            preconditions_met=True,
            postconditions_met=postconditions_met,
            is_satisfied=is_satisfied,
            violation_reason=violation,
        )

    def evaluate_entity(
        self,
        entity: Entity,
        facts: List[Fact],
        domains: Optional[Set[str]] = None
    ) -> List[RuleEvaluation]:
        """
        Evaluate all applicable rules against an entity.

        Args:
            entity: Entity to check
            facts: Facts about the entity
            domains: Optional domains to filter rules

        Returns:
            List of RuleEvaluation results
        """
        if domains:
            rules = self.get_rules_for_domains(domains)
        else:
            rules = list(self.rules.values())

        evaluations = []
        for rule in rules:
            eval_result = self.evaluate_rule(rule, entity, facts)
            evaluations.append(eval_result)

        return evaluations

    def check_impossibility(
        self,
        intent: QueryIntent,
        domains: Optional[Set[str]] = None
    ) -> List[Impossibility]:
        """
        Check if a query intent is impossible according to rules.

        This checks rules without requiring actual entities/facts,
        looking for structural impossibilities.

        Args:
            intent: Query intent to check
            domains: Optional domains to filter rules

        Returns:
            List of impossibilities found
        """
        impossibilities = []

        if domains:
            rules = self.get_rules_for_domains(domains)
        else:
            rules = list(self.rules.values())

        for rule in rules:
            if rule.hardness != "hard":
                continue

            # Check if intent contradicts rule
            contradiction = self._check_intent_vs_rule(intent, rule)
            if contradiction:
                impossibilities.append(Impossibility(
                    reason=contradiction,
                    violated_rule=rule,
                    explanation=rule.explain_violation(None, []),
                ))

        return impossibilities

    def _check_preconditions(
        self,
        rule: Rule,
        entity: Entity,
        facts: List[Fact]
    ) -> bool:
        """Check if all preconditions are met."""
        for precond in rule.preconditions:
            if not self._evaluate_constraint(precond, entity, facts):
                return False
        return True

    def _check_postconditions(
        self,
        rule: Rule,
        entity: Entity,
        facts: List[Fact]
    ) -> Tuple[bool, Optional[str]]:
        """Check if all postconditions are met."""
        for postcond in rule.postconditions:
            if not self._evaluate_constraint(postcond, entity, facts):
                violation = f"Postcondition failed: {postcond.property_path} " \
                           f"{postcond.constraint.op.value} {postcond.constraint.value}"
                return False, violation
        return True, None

    def _evaluate_constraint(
        self,
        rule_constraint: RuleConstraint,
        entity: Entity,
        facts: List[Fact]
    ) -> bool:
        """Evaluate a single constraint."""
        path = rule_constraint.property_path
        constraint = rule_constraint.constraint

        # Entity property constraints
        if path.startswith("entity."):
            prop_name = path[7:]
            if prop_name == "type":
                return constraint.satisfied_by(entity.type)
            value = entity.get_property(prop_name)
            if value is None:
                # Property doesn't exist
                return constraint.op in (ComparisonOp.NEQ, ComparisonOp.NOT_IN)
            return constraint.satisfied_by(value)

        # Fact constraints
        if path.startswith("fact."):
            predicate = path[5:]
            entity_facts = [
                f for f in facts
                if f.subject.id == entity.id and f.predicate == predicate
            ]

            if not entity_facts:
                # No fact exists
                return constraint.op in (ComparisonOp.NEQ, ComparisonOp.NOT_IN)

            # Check if any fact satisfies the constraint
            return any(
                constraint.satisfied_by(f.value if not f.negated else not f.value)
                for f in entity_facts
            )

        return False

    def _check_intent_vs_rule(
        self,
        intent: QueryIntent,
        rule: Rule
    ) -> Optional[str]:
        """
        Check if a query intent contradicts a rule.

        Returns contradiction reason if found, None otherwise.
        """
        # Check if intent requires something the rule forbids
        for precond in rule.preconditions:
            for postcond in rule.postconditions:
                # If precondition matches intent requirements
                # and postcondition conflicts with intent requirements
                if self._intent_matches_condition(intent, precond):
                    contradiction = self._intent_contradicts_condition(intent, postcond)
                    if contradiction:
                        return f"Intent requires {contradiction} but rule '{rule.name}' forbids it"

        return None

    def _intent_matches_condition(
        self,
        intent: QueryIntent,
        condition: RuleConstraint
    ) -> bool:
        """Check if intent matches a rule condition."""
        path = condition.property_path
        constraint = condition.constraint

        if path.startswith("entity.type"):
            # Check required types
            if constraint.op == ComparisonOp.EQ:
                return constraint.value in intent.required_types
            elif constraint.op == ComparisonOp.IN:
                return any(t in constraint.value for t in intent.required_types)

        if path.startswith("fact."):
            predicate = path[5:]
            # Check if intent has requirement for this predicate
            if predicate in intent.required_properties:
                return True

        return False

    def _intent_contradicts_condition(
        self,
        intent: QueryIntent,
        condition: RuleConstraint
    ) -> Optional[str]:
        """Check if intent contradicts a condition."""
        path = condition.property_path
        constraint = condition.constraint

        if path.startswith("fact."):
            predicate = path[5:]

            # Check required properties
            if predicate in intent.required_properties:
                req_constraint = intent.required_properties[predicate]
                if self._constraints_conflict(req_constraint, constraint):
                    return f"{predicate} = {req_constraint.value}"

            # Check forbidden properties
            if predicate in intent.forbidden_properties:
                forb_constraint = intent.forbidden_properties[predicate]
                if not self._constraints_conflict(forb_constraint, constraint):
                    return f"{predicate} = {forb_constraint.value}"

        return None

    def _constraints_conflict(
        self,
        c1: ValueConstraint,
        c2: ValueConstraint
    ) -> bool:
        """Check if two constraints conflict (cannot both be satisfied)."""
        # EQ conflicts with different value
        if c1.op == ComparisonOp.EQ and c2.op == ComparisonOp.EQ:
            return c1.value != c2.value

        # EQ true conflicts with EQ false
        if c1.op == ComparisonOp.EQ and c2.op == ComparisonOp.NEQ:
            return c1.value == c2.value

        # Range conflicts
        if c1.op == ComparisonOp.GT and c2.op == ComparisonOp.LT:
            return c1.value >= c2.value

        if c1.op == ComparisonOp.LT and c2.op == ComparisonOp.GT:
            return c1.value <= c2.value

        return False
