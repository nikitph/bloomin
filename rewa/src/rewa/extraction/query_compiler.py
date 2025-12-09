"""
Query Compiler

Compiles natural language queries into structured QueryIntent objects.
Extracts required types, properties, constraints, and temporal assumptions.
"""

import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from rewa.models import (
    QueryIntent,
    ValueConstraint,
    ComparisonOp,
    TemporalPolicy,
    TimeInterval,
    Vector,
)


@dataclass
class TypePattern:
    """Pattern for detecting required types."""
    pattern: str
    entity_type: str
    confidence: float = 0.8


@dataclass
class PropertyPattern:
    """Pattern for detecting property constraints."""
    pattern: str
    property_name: str
    value_extractor: str  # Regex group or literal value
    comparison: ComparisonOp = ComparisonOp.EQ
    forbidden: bool = False


class QueryCompiler:
    """
    Compiles natural language queries into QueryIntent.

    Analyzes query text to extract:
    - Required entity types
    - Required property constraints
    - Forbidden property constraints
    - Temporal policies
    """

    def __init__(self):
        self.type_patterns: List[TypePattern] = []
        self.property_patterns: List[PropertyPattern] = []
        self.temporal_keywords: Dict[str, TemporalPolicy] = {}
        self._setup_default_patterns()

    def _setup_default_patterns(self):
        """Setup default patterns for query compilation."""
        # Type patterns
        self.type_patterns = [
            TypePattern(r'\b(gun|weapon|firearm|pistol|rifle)\b', "Weapon", 0.9),
            TypePattern(r'\b(toy|plaything)\b', "Toy", 0.9),
            TypePattern(r'\b(drug|medication|medicine|pharmaceutical)\b', "Drug", 0.9),
            TypePattern(r'\b(device|charger|appliance|equipment)\b', "Device", 0.8),
            TypePattern(r'\b(treatment|therapy|cure)\b', "Treatment", 0.8),
            TypePattern(r'\b(person|patient|individual)\b', "Person", 0.7),
            TypePattern(r'\b(cancer|disease|condition|illness)\b', "MedicalCondition", 0.8),
        ]

        # Property patterns (required)
        self.property_patterns = [
            # Danger/Safety
            PropertyPattern(
                r'\b(dangerous|lethal|deadly)\b',
                "dangerous", "true", ComparisonOp.EQ, forbidden=False
            ),
            PropertyPattern(
                r'\b(safe|harmless|non-lethal)\b',
                "dangerous", "false", ComparisonOp.EQ, forbidden=False
            ),

            # Self-defense
            PropertyPattern(
                r'\b(self[- ]?defense|protection)\b',
                "usable_for_self_defense", "true", ComparisonOp.EQ, forbidden=False
            ),

            # Medical effectiveness
            PropertyPattern(
                r'\b(cures?|treats?|heals?)\s+(\w+)',
                "cures", "2", ComparisonOp.EQ, forbidden=False
            ),
            PropertyPattern(
                r'\b(no|zero|without)\s+side\s+effects?\b',
                "has_side_effects", "false", ComparisonOp.EQ, forbidden=False
            ),
            PropertyPattern(
                r'\bwith\s+side\s+effects?\b',
                "has_side_effects", "true", ComparisonOp.EQ, forbidden=False
            ),

            # Forbidden patterns
            PropertyPattern(
                r'\bnot?\s+(?:a\s+)?toy\b',
                "is_toy", "true", ComparisonOp.EQ, forbidden=True
            ),
            PropertyPattern(
                r'\bnot?\s+fake\b',
                "is_fake", "true", ComparisonOp.EQ, forbidden=True
            ),
            PropertyPattern(
                r'\breal\b(?!\s+estate)',
                "is_real", "true", ComparisonOp.EQ, forbidden=False
            ),
        ]

        # Temporal keywords
        self.temporal_keywords = {
            "latest": TemporalPolicy.LATEST,
            "current": TemporalPolicy.LATEST,
            "now": TemporalPolicy.LATEST,
            "recent": TemporalPolicy.LATEST,
            "all": TemporalPolicy.ALL,
            "history": TemporalPolicy.ALL,
            "historical": TemporalPolicy.ALL,
        }

    def compile(
        self,
        query: str,
        embedding: Optional[Vector] = None
    ) -> QueryIntent:
        """
        Compile a natural language query into QueryIntent.

        Args:
            query: Natural language query string
            embedding: Optional pre-computed embedding

        Returns:
            Compiled QueryIntent
        """
        query_lower = query.lower()

        # Extract required types
        required_types = self._extract_types(query_lower)

        # Extract property constraints
        required_props, forbidden_props = self._extract_properties(query_lower)

        # Determine temporal policy
        temporal_policy, temporal_ref = self._extract_temporal(query_lower)

        return QueryIntent(
            id=str(uuid.uuid4()),
            raw_query=query,
            embedding=embedding,
            required_types=required_types,
            required_properties=required_props,
            forbidden_properties=forbidden_props,
            temporal_policy=temporal_policy,
            temporal_reference=temporal_ref,
        )

    def _extract_types(self, query: str) -> Set[str]:
        """Extract required entity types from query."""
        types: Set[str] = set()

        for pattern in self.type_patterns:
            if re.search(pattern.pattern, query, re.IGNORECASE):
                types.add(pattern.entity_type)

        return types

    def _extract_properties(
        self,
        query: str
    ) -> Tuple[Dict[str, ValueConstraint], Dict[str, ValueConstraint]]:
        """Extract required and forbidden property constraints."""
        required: Dict[str, ValueConstraint] = {}
        forbidden: Dict[str, ValueConstraint] = {}

        for pattern in self.property_patterns:
            match = re.search(pattern.pattern, query, re.IGNORECASE)
            if match:
                # Extract value
                if pattern.value_extractor.isdigit():
                    # Reference to regex group
                    group_num = int(pattern.value_extractor)
                    if group_num <= len(match.groups()):
                        value = match.group(group_num)
                    else:
                        value = True
                elif pattern.value_extractor == "true":
                    value = True
                elif pattern.value_extractor == "false":
                    value = False
                else:
                    value = pattern.value_extractor

                constraint = ValueConstraint(
                    op=pattern.comparison,
                    value=value
                )

                if pattern.forbidden:
                    forbidden[pattern.property_name] = constraint
                else:
                    required[pattern.property_name] = constraint

        return required, forbidden

    def _extract_temporal(
        self,
        query: str
    ) -> Tuple[TemporalPolicy, Optional[datetime]]:
        """Extract temporal policy and reference from query."""
        policy = TemporalPolicy.LATEST  # Default
        reference = None

        # Check for temporal keywords
        for keyword, temp_policy in self.temporal_keywords.items():
            if keyword in query:
                policy = temp_policy
                break

        # Check for specific date references
        date_patterns = [
            (r'as of (\d{4}-\d{2}-\d{2})', '%Y-%m-%d'),
            (r'on (\d{2}/\d{2}/\d{4})', '%m/%d/%Y'),
            (r'in (\d{4})', '%Y'),
        ]

        for pattern, fmt in date_patterns:
            match = re.search(pattern, query)
            if match:
                try:
                    reference = datetime.strptime(match.group(1), fmt)
                    policy = TemporalPolicy.AT_TIME
                    break
                except ValueError:
                    continue

        return policy, reference


def compile_query(
    query: str,
    embedding: Optional[Vector] = None
) -> QueryIntent:
    """
    Convenience function to compile a query.

    Args:
        query: Natural language query
        embedding: Optional embedding

    Returns:
        Compiled QueryIntent
    """
    compiler = QueryCompiler()
    return compiler.compile(query, embedding)


class ImpossibilityPatternDetector:
    """
    Detects logically impossible patterns in queries.

    Catches contradictions like "drug with zero side effects that cures cancer"
    before they even reach the constraint solver.
    """

    def __init__(self):
        self.impossibility_patterns: List[Tuple[str, str]] = []
        self._setup_patterns()

    def _setup_patterns(self):
        """Setup impossibility detection patterns."""
        self.impossibility_patterns = [
            # Medical impossibilities - various orderings
            (
                r'(?:cure|cures|treat|treats|heal|heals).*cancer.*(?:no|zero|without)\s+side\s+effects?',
                "Medical treatments that cure cancer inherently have side effects"
            ),
            (
                r'(?:no|zero|without)\s+side\s+effects?.*(?:cure|cures|treat|treats|heal|heals).*cancer',
                "Medical treatments that cure cancer inherently have side effects"
            ),
            (
                r'cancer.*(?:cure|cures|treatment).*(?:no|zero|without)\s+side\s+effects?',
                "Medical treatments that cure cancer inherently have side effects"
            ),
            (
                r'cancer.*(?:no|zero|without)\s+side\s+effects?',
                "Medical treatments that cure cancer inherently have side effects"
            ),

            # Physical impossibilities
            (
                r'\bperpetual\s+motion\b',
                "Perpetual motion violates thermodynamics"
            ),
            (
                r'\binfinite\s+energy\b',
                "Infinite energy violates conservation laws"
            ),
            (
                r'\b(?:faster|quicker)\s+than\s+light\b',
                "Faster than light travel violates relativity"
            ),

            # Logical impossibilities
            (
                r'\bsquare\s+circle\b',
                "A square circle is a logical contradiction"
            ),
            (
                r'\bmarried\s+bachelor\b',
                "A married bachelor is a logical contradiction"
            ),

            # Contradictory properties
            (
                r'\b(?:safe|harmless).*(?:dangerous|lethal|deadly)\b',
                "Cannot be both safe and dangerous"
            ),
            (
                r'\b(?:dangerous|lethal|deadly).*(?:safe|harmless)\b',
                "Cannot be both dangerous and safe"
            ),
        ]

    def detect(self, query: str) -> List[str]:
        """
        Detect impossibilities in query.

        Args:
            query: Query text to check

        Returns:
            List of impossibility reasons (empty if none found)
        """
        reasons = []
        query_lower = query.lower()

        for pattern, reason in self.impossibility_patterns:
            if re.search(pattern, query_lower):
                reasons.append(reason)

        return reasons

    def is_possible(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a query is logically possible.

        Returns:
            (is_possible, reason_if_impossible)
        """
        reasons = self.detect(query)

        if reasons:
            return False, reasons[0]

        return True, None
