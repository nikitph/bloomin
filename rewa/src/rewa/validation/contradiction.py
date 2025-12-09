"""
Contradiction Detector

Detects contradictions between facts across:
- Same entity, different values
- Temporal conflicts
- Negation conflicts
- Cross-chart conflicts
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rewa.models import (
    Fact,
    Contradiction,
    TimeInterval,
)


@dataclass
class ConflictResolution:
    """Strategy for resolving conflicts."""
    strategy: str  # "temporal", "confidence", "source", "manual"
    winner: Optional[Fact] = None
    reason: str = ""


class ContradictionDetector:
    """
    Detects contradictions between facts.

    Implements the contradiction detection algorithm from PRD:
    - Group facts by subject and predicate
    - Check for conflicting values
    - Apply resolution strategies
    """

    def __init__(
        self,
        temporal_precedence: bool = True,
        confidence_threshold: float = 0.1
    ):
        """
        Initialize the detector.

        Args:
            temporal_precedence: If True, later facts override earlier
            confidence_threshold: Minimum difference to prefer one fact
        """
        self.temporal_precedence = temporal_precedence
        self.confidence_threshold = confidence_threshold

    def detect(self, facts: List[Fact]) -> List[Contradiction]:
        """
        Detect all contradictions in a set of facts.

        Args:
            facts: List of facts to check

        Returns:
            List of detected contradictions
        """
        contradictions = []

        # Group facts by subject and predicate
        groups = self._group_facts(facts)

        for (subject_id, predicate), group_facts in groups.items():
            if len(group_facts) < 2:
                continue

            # Check for conflicts within each group
            group_contradictions = self._check_group(group_facts)
            contradictions.extend(group_contradictions)

        return contradictions

    def detect_and_resolve(
        self,
        facts: List[Fact]
    ) -> Tuple[List[Fact], List[Contradiction]]:
        """
        Detect contradictions and resolve them.

        Args:
            facts: List of facts to check

        Returns:
            (resolved_facts, contradictions)
        """
        contradictions = self.detect(facts)

        if not contradictions:
            return facts, []

        # Remove losing facts
        facts_to_remove = set()
        for contradiction in contradictions:
            if contradiction.resolved_fact:
                # Remove the non-winning fact
                loser = (
                    contradiction.fact_b
                    if contradiction.resolved_fact.id == contradiction.fact_a.id
                    else contradiction.fact_a
                )
                facts_to_remove.add(loser.id)

        resolved_facts = [f for f in facts if f.id not in facts_to_remove]

        return resolved_facts, contradictions

    def _group_facts(
        self,
        facts: List[Fact]
    ) -> Dict[Tuple[str, str], List[Fact]]:
        """Group facts by subject ID and predicate."""
        groups: Dict[Tuple[str, str], List[Fact]] = defaultdict(list)

        for fact in facts:
            key = (fact.subject.id, fact.predicate)
            groups[key].append(fact)

        return groups

    def _check_group(self, facts: List[Fact]) -> List[Contradiction]:
        """Check a group of facts for contradictions."""
        contradictions = []

        for i, fact_a in enumerate(facts):
            for fact_b in facts[i + 1:]:
                if self._facts_conflict(fact_a, fact_b):
                    resolution = self._resolve_conflict(fact_a, fact_b)
                    contradiction = Contradiction(
                        fact_a=fact_a,
                        fact_b=fact_b,
                        resolution=resolution.strategy,
                        resolved_fact=resolution.winner,
                        explanation=resolution.reason,
                    )
                    contradictions.append(contradiction)

        return contradictions

    def _facts_conflict(self, fact_a: Fact, fact_b: Fact) -> bool:
        """Check if two facts conflict."""
        # Different subjects or predicates - no conflict
        if fact_a.subject.id != fact_b.subject.id:
            return False
        if fact_a.predicate != fact_b.predicate:
            return False

        # Check temporal overlap
        if fact_a.valid_time and fact_b.valid_time:
            if not fact_a.valid_time.overlaps(fact_b.valid_time):
                return False

        # Check value conflict
        if fact_a.negated != fact_b.negated:
            # One is negated, one is not
            if self._values_equal(fact_a.value, fact_b.value):
                return True
        else:
            # Same polarity - conflict if different values
            if not self._values_equal(fact_a.value, fact_b.value):
                return True

        return False

    def _values_equal(self, v1: Any, v2: Any) -> bool:
        """Check if two values are equal (handling type variations)."""
        # Direct equality
        if v1 == v2:
            return True

        # Boolean normalization
        bool_true = {True, "true", "True", "yes", "Yes", 1, "1"}
        bool_false = {False, "false", "False", "no", "No", 0, "0"}

        if v1 in bool_true and v2 in bool_true:
            return True
        if v1 in bool_false and v2 in bool_false:
            return True

        # String normalization
        if isinstance(v1, str) and isinstance(v2, str):
            return v1.lower().strip() == v2.lower().strip()

        return False

    def _resolve_conflict(
        self,
        fact_a: Fact,
        fact_b: Fact
    ) -> ConflictResolution:
        """Resolve a conflict between two facts."""
        # Try temporal resolution
        if self.temporal_precedence:
            temporal_winner = self._temporal_resolution(fact_a, fact_b)
            if temporal_winner:
                return ConflictResolution(
                    strategy="temporal",
                    winner=temporal_winner,
                    reason="More recent fact takes precedence",
                )

        # Try confidence resolution
        conf_diff = abs(fact_a.confidence - fact_b.confidence)
        if conf_diff >= self.confidence_threshold:
            winner = fact_a if fact_a.confidence > fact_b.confidence else fact_b
            return ConflictResolution(
                strategy="confidence",
                winner=winner,
                reason=f"Higher confidence ({winner.confidence:.2f})",
            )

        # Cannot resolve automatically
        return ConflictResolution(
            strategy="unresolved",
            winner=None,
            reason="Cannot automatically resolve conflict",
        )

    def _temporal_resolution(
        self,
        fact_a: Fact,
        fact_b: Fact
    ) -> Optional[Fact]:
        """Try to resolve conflict using temporal information."""
        if not fact_a.valid_time or not fact_a.valid_time.start:
            return None
        if not fact_b.valid_time or not fact_b.valid_time.start:
            return None

        if fact_a.valid_time.start > fact_b.valid_time.start:
            return fact_a
        elif fact_b.valid_time.start > fact_a.valid_time.start:
            return fact_b

        return None


def detect_contradictions(facts: List[Fact]) -> List[Contradiction]:
    """
    Convenience function to detect contradictions.

    Args:
        facts: List of facts to check

    Returns:
        List of contradictions
    """
    detector = ContradictionDetector()
    return detector.detect(facts)


class CrossChartContradictionDetector:
    """
    Detects contradictions between facts from different charts.

    This is important for ambiguity detection - when the same
    entity has conflicting facts in different semantic regions.
    """

    def __init__(self):
        self.base_detector = ContradictionDetector()

    def detect_cross_chart(
        self,
        facts_by_chart: Dict[str, List[Fact]]
    ) -> List[Tuple[str, str, Contradiction]]:
        """
        Detect contradictions across charts.

        Args:
            facts_by_chart: Map of chart_id -> facts

        Returns:
            List of (chart_a_id, chart_b_id, contradiction) tuples
        """
        cross_contradictions = []

        chart_ids = list(facts_by_chart.keys())

        for i, chart_a_id in enumerate(chart_ids):
            for chart_b_id in chart_ids[i + 1:]:
                facts_a = facts_by_chart[chart_a_id]
                facts_b = facts_by_chart[chart_b_id]

                # Only check facts about same entities
                entities_a = {f.subject.id for f in facts_a}
                entities_b = {f.subject.id for f in facts_b}
                common_entities = entities_a & entities_b

                if not common_entities:
                    continue

                # Filter to common entities
                relevant_facts = [
                    f for f in facts_a + facts_b
                    if f.subject.id in common_entities
                ]

                contradictions = self.base_detector.detect(relevant_facts)

                for c in contradictions:
                    # Check if facts are from different charts
                    fact_a_chart = self._get_fact_chart(c.fact_a, facts_by_chart)
                    fact_b_chart = self._get_fact_chart(c.fact_b, facts_by_chart)

                    if fact_a_chart != fact_b_chart:
                        cross_contradictions.append(
                            (fact_a_chart, fact_b_chart, c)
                        )

        return cross_contradictions

    def _get_fact_chart(
        self,
        fact: Fact,
        facts_by_chart: Dict[str, List[Fact]]
    ) -> Optional[str]:
        """Get which chart a fact belongs to."""
        for chart_id, facts in facts_by_chart.items():
            for f in facts:
                if f.id == fact.id:
                    return chart_id
        return None
