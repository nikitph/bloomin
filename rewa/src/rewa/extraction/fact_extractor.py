"""
Fact Extractor

Extracts facts (subject-predicate-value triples) from text.
Facts represent assertions about entities.
"""

import re
import uuid
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from rewa.models import Entity, Fact, TimeInterval


@dataclass
class FactPattern:
    """Pattern for extracting facts."""
    name: str
    predicate: str
    subject_pattern: str  # Regex to match subject
    value_pattern: str    # Regex to extract value
    negation_pattern: Optional[str] = None  # Pattern indicating negation
    temporal_pattern: Optional[str] = None  # Pattern for time extraction
    confidence: float = 0.8


class FactExtractor:
    """
    Extracts facts from text given entities.

    Facts are subject-predicate-value triples that capture
    assertions about entities.
    """

    def __init__(self):
        self.patterns: List[FactPattern] = []
        self.predicate_keywords: Dict[str, List[str]] = {}
        self.negation_indicators: List[str] = [
            "not", "no", "never", "none", "cannot", "can't",
            "doesn't", "don't", "isn't", "aren't", "won't",
            "without", "lack", "lacking", "absence", "absent"
        ]
        self._setup_default_patterns()

    def _setup_default_patterns(self):
        """Setup default fact extraction patterns."""
        # Danger/Safety facts
        self.add_pattern(FactPattern(
            name="is_dangerous",
            predicate="dangerous",
            subject_pattern=r'(.+?)\s+(?:is|are)\s+(?:very\s+)?dangerous',
            value_pattern=r'dangerous',
            negation_pattern=r'not\s+dangerous|safe|harmless',
            confidence=0.85
        ))

        # Medical treatment facts
        self.add_pattern(FactPattern(
            name="cures",
            predicate="cures",
            subject_pattern=r'(.+?)\s+(?:cures?|treats?|heals?)',
            value_pattern=r'(?:cures?|treats?|heals?)\s+(.+?)(?:\.|,|$)',
            confidence=0.8
        ))

        # Side effects facts
        self.add_pattern(FactPattern(
            name="has_side_effects",
            predicate="has_side_effects",
            subject_pattern=r'(.+?)\s+(?:has|have|causes?|produces?)\s+(?:\w+\s+)?side\s+effects?',
            value_pattern=r'side\s+effects?\s*(?:include|such\s+as|like)?\s*:?\s*(.+?)(?:\.|$)',
            negation_pattern=r'no\s+side\s+effects?|without\s+side\s+effects?|zero\s+side\s+effects?',
            confidence=0.85
        ))

        # Usability facts
        self.add_pattern(FactPattern(
            name="usable_for",
            predicate="usable_for",
            subject_pattern=r'(.+?)\s+(?:can\s+be\s+used|is\s+used|useful)\s+for',
            value_pattern=r'(?:used|useful)\s+for\s+(.+?)(?:\.|,|$)',
            confidence=0.75
        ))

        # Toy/Real distinction
        self.add_pattern(FactPattern(
            name="is_toy",
            predicate="is_toy",
            subject_pattern=r'(.+?)\s+is\s+(?:a\s+)?toy',
            value_pattern=r'toy',
            confidence=0.9
        ))

        self.add_pattern(FactPattern(
            name="is_real",
            predicate="is_real",
            subject_pattern=r'(.+?)\s+is\s+(?:a\s+)?real',
            value_pattern=r'real|genuine|authentic',
            confidence=0.9
        ))

        # Setup predicate keywords
        self.predicate_keywords = {
            "dangerous": ["dangerous", "lethal", "deadly", "harmful", "hazardous"],
            "safe": ["safe", "harmless", "benign", "non-toxic"],
            "cures": ["cures", "treats", "heals", "remedies"],
            "causes": ["causes", "induces", "triggers", "produces"],
            "contains": ["contains", "includes", "has", "comprises"],
            "requires": ["requires", "needs", "demands", "necessitates"],
        }

    def add_pattern(self, pattern: FactPattern) -> None:
        """Add a fact extraction pattern."""
        self.patterns.append(pattern)

    def extract(
        self,
        text: str,
        entities: List[Entity],
        chunk_id: Optional[str] = None
    ) -> List[Fact]:
        """
        Extract facts from text given entities.

        Args:
            text: Text to extract from
            entities: Entities found in the text
            chunk_id: Optional chunk identifier

        Returns:
            List of extracted facts
        """
        facts: List[Fact] = []
        text_lower = text.lower()

        # Pattern-based extraction
        for pattern in self.patterns:
            extracted = self._extract_with_pattern(
                pattern, text_lower, entities, chunk_id
            )
            facts.extend(extracted)

        # Keyword-based extraction for each entity
        for entity in entities:
            keyword_facts = self._extract_keyword_facts(
                entity, text_lower, chunk_id
            )
            facts.extend(keyword_facts)

        # Deduplicate
        facts = self._deduplicate_facts(facts)

        return facts

    def _extract_with_pattern(
        self,
        pattern: FactPattern,
        text: str,
        entities: List[Entity],
        chunk_id: Optional[str]
    ) -> List[Fact]:
        """Extract facts using a specific pattern."""
        facts = []

        # Find subject matches
        subject_matches = re.finditer(
            pattern.subject_pattern, text, re.IGNORECASE
        )

        for match in subject_matches:
            subject_text = match.group(1) if match.groups() else match.group(0)

            # Find matching entity
            entity = self._find_matching_entity(subject_text, entities)
            if not entity:
                continue

            # Check for negation
            negated = False
            if pattern.negation_pattern:
                context = self._get_context(text, match.start(), match.end())
                if re.search(pattern.negation_pattern, context, re.IGNORECASE):
                    negated = True

            # Also check general negation indicators
            if not negated:
                negated = self._check_negation(text, match.start(), match.end())

            # Extract value
            value_match = re.search(
                pattern.value_pattern, text[match.start():], re.IGNORECASE
            )
            value = value_match.group(1) if value_match and value_match.groups() else True

            # Extract temporal info
            valid_time = None
            if pattern.temporal_pattern:
                time_match = re.search(
                    pattern.temporal_pattern, text, re.IGNORECASE
                )
                if time_match:
                    valid_time = self._parse_temporal(time_match.group(0))

            # Create fact
            fact_id = self._generate_fact_id(entity.id, pattern.predicate, value)
            fact = Fact(
                id=fact_id,
                subject=entity,
                predicate=pattern.predicate,
                value=value,
                valid_time=valid_time,
                confidence=pattern.confidence * (0.9 if negated else 1.0),
                source_chunk_id=chunk_id,
                negated=negated,
            )
            facts.append(fact)

        return facts

    def _extract_keyword_facts(
        self,
        entity: Entity,
        text: str,
        chunk_id: Optional[str]
    ) -> List[Fact]:
        """Extract facts based on keyword proximity to entity."""
        facts = []

        # Find entity mentions in text
        entity_pattern = re.escape(entity.name.lower())
        entity_matches = list(re.finditer(entity_pattern, text, re.IGNORECASE))

        if not entity_matches:
            return facts

        for predicate, keywords in self.predicate_keywords.items():
            for keyword in keywords:
                keyword_matches = list(re.finditer(
                    r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE
                ))

                for km in keyword_matches:
                    # Check proximity to any entity mention
                    for em in entity_matches:
                        distance = abs(km.start() - em.start())
                        if distance < 100:  # Within 100 chars
                            # Check negation
                            negated = self._check_negation(
                                text, min(km.start(), em.start()),
                                max(km.end(), em.end())
                            )

                            fact_id = self._generate_fact_id(
                                entity.id, predicate, keyword
                            )
                            confidence = 0.7 * (1 - distance / 200)  # Decay with distance

                            fact = Fact(
                                id=fact_id,
                                subject=entity,
                                predicate=predicate,
                                value=True,
                                confidence=confidence,
                                source_chunk_id=chunk_id,
                                negated=negated,
                            )
                            facts.append(fact)
                            break  # One fact per keyword match

        return facts

    def _find_matching_entity(
        self,
        subject_text: str,
        entities: List[Entity]
    ) -> Optional[Entity]:
        """Find an entity that matches the subject text."""
        subject_lower = subject_text.lower().strip()

        for entity in entities:
            entity_name_lower = entity.name.lower()
            if entity_name_lower in subject_lower or subject_lower in entity_name_lower:
                return entity

        return None

    def _check_negation(self, text: str, start: int, end: int) -> bool:
        """Check if there's negation in the context around a span."""
        # Get context window
        context_start = max(0, start - 30)
        context_end = min(len(text), end + 10)
        context = text[context_start:context_end].lower()

        for indicator in self.negation_indicators:
            if indicator in context:
                return True

        return False

    def _get_context(self, text: str, start: int, end: int, window: int = 50) -> str:
        """Get context window around a span."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]

    def _parse_temporal(self, time_text: str) -> Optional[TimeInterval]:
        """Parse temporal expression into TimeInterval."""
        # Simple temporal parsing
        time_text = time_text.lower()

        # Common patterns
        date_patterns = [
            (r'(\d{4})-(\d{2})-(\d{2})', '%Y-%m-%d'),
            (r'(\d{2})/(\d{2})/(\d{4})', '%m/%d/%Y'),
        ]

        for pattern, fmt in date_patterns:
            match = re.search(pattern, time_text)
            if match:
                try:
                    date = datetime.strptime(match.group(0), fmt)
                    return TimeInterval(start=date)
                except ValueError:
                    continue

        # Relative time expressions
        if "now" in time_text or "current" in time_text:
            return TimeInterval(start=datetime.now())

        return None

    def _generate_fact_id(
        self,
        entity_id: str,
        predicate: str,
        value: Any
    ) -> str:
        """Generate deterministic fact ID."""
        id_input = f"{entity_id}:{predicate}:{value}"
        return hashlib.md5(id_input.encode()).hexdigest()[:12]

    def _deduplicate_facts(self, facts: List[Fact]) -> List[Fact]:
        """Deduplicate facts, keeping highest confidence."""
        seen: Dict[str, Fact] = {}

        for fact in facts:
            key = f"{fact.subject.id}:{fact.predicate}:{fact.negated}"

            if key in seen:
                if fact.confidence > seen[key].confidence:
                    seen[key] = fact
            else:
                seen[key] = fact

        return list(seen.values())


class TemporalFactResolver:
    """
    Resolves temporal conflicts between facts.

    Implements temporal precedence rules.
    """

    def __init__(self, policy: str = "latest"):
        """
        Initialize resolver.

        Args:
            policy: Resolution policy - "latest", "earliest", or "all"
        """
        self.policy = policy

    def resolve(self, facts: List[Fact]) -> List[Fact]:
        """
        Resolve temporal conflicts in facts.

        Args:
            facts: List of potentially conflicting facts

        Returns:
            Resolved list of facts
        """
        if self.policy == "all":
            return facts

        # Group facts by subject and predicate
        groups: Dict[str, List[Fact]] = {}

        for fact in facts:
            key = f"{fact.subject.id}:{fact.predicate}"
            if key not in groups:
                groups[key] = []
            groups[key].append(fact)

        # Resolve each group
        resolved = []
        for key, group_facts in groups.items():
            if len(group_facts) == 1:
                resolved.append(group_facts[0])
            else:
                winner = self._resolve_group(group_facts)
                resolved.append(winner)

        return resolved

    def _resolve_group(self, facts: List[Fact]) -> Fact:
        """Resolve a group of conflicting facts."""
        # Separate by temporal info
        with_time = [f for f in facts if f.valid_time and f.valid_time.start]
        without_time = [f for f in facts if not f.valid_time or not f.valid_time.start]

        if with_time:
            if self.policy == "latest":
                return max(with_time, key=lambda f: f.valid_time.start)
            else:  # earliest
                return min(with_time, key=lambda f: f.valid_time.start)

        # No temporal info - use confidence
        return max(facts, key=lambda f: f.confidence)
