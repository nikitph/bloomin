"""
Entity Extractor

Extracts semantic entities from text chunks.
Uses pattern matching and configurable extractors.
"""

import re
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import hashlib

from rewa.models import Entity, Vector


@dataclass
class ExtractionPattern:
    """Pattern for extracting entities."""
    name: str
    entity_type: str
    pattern: str  # Regex pattern
    property_extractors: Dict[str, str] = None  # property_name -> regex group name
    confidence: float = 0.8


@dataclass
class Chunk:
    """A text chunk from retrieval."""
    id: str
    text: str
    embedding: Optional[Vector] = None
    metadata: Dict[str, Any] = None


class EntityExtractor:
    """
    Extracts entities from text chunks.

    Supports multiple extraction strategies:
    - Pattern-based extraction
    - Keyword-based extraction
    - Custom extractors
    """

    def __init__(self):
        self.patterns: Dict[str, List[ExtractionPattern]] = {}
        self.keyword_types: Dict[str, Set[str]] = {}  # keyword -> entity types
        self.custom_extractors: List[Callable[[str], List[Entity]]] = []
        self._setup_default_patterns()

    def _setup_default_patterns(self):
        """Setup default extraction patterns."""
        # Weapon patterns
        self.add_pattern(ExtractionPattern(
            name="firearm",
            entity_type="Weapon",
            pattern=r'\b(gun|rifle|pistol|firearm|handgun|shotgun|revolver)\b',
            property_extractors={"is_firearm": "true"},
            confidence=0.9
        ))

        # Toy patterns
        self.add_pattern(ExtractionPattern(
            name="toy_gun",
            entity_type="Toy",
            pattern=r'\b(toy\s+gun|nerf|water\s+gun|plastic\s+gun|fake\s+gun)\b',
            property_extractors={"is_toy": "true"},
            confidence=0.9
        ))

        # Drug/Medicine patterns
        self.add_pattern(ExtractionPattern(
            name="medication",
            entity_type="Drug",
            pattern=r'\b(medication|medicine|drug|pharmaceutical|treatment|therapy)\b',
            confidence=0.7
        ))

        # Medical condition patterns
        self.add_pattern(ExtractionPattern(
            name="condition",
            entity_type="MedicalCondition",
            pattern=r'\b(cancer|disease|syndrome|disorder|illness|infection)\b',
            confidence=0.7
        ))

        # Person patterns
        self.add_pattern(ExtractionPattern(
            name="person",
            entity_type="Person",
            pattern=r'\b(patient|person|individual|user|customer)\b',
            confidence=0.6
        ))

        # Device patterns
        self.add_pattern(ExtractionPattern(
            name="device",
            entity_type="Device",
            pattern=r'\b(charger|device|machine|equipment|appliance)\b',
            confidence=0.6
        ))

        # Add keyword types for common concepts
        self._setup_keyword_types()

    def _setup_keyword_types(self):
        """Setup keyword to type mappings."""
        self.keyword_types = {
            # Danger indicators
            "dangerous": {"Weapon", "Hazard"},
            "lethal": {"Weapon", "Hazard"},
            "harmful": {"Hazard"},
            "safe": {"SafeItem"},

            # Medical
            "cures": {"Drug", "Treatment"},
            "treats": {"Drug", "Treatment"},
            "heals": {"Drug", "Treatment"},
            "side_effects": {"Drug"},

            # Safety
            "self_defense": {"Weapon", "SafetyDevice"},
            "protection": {"SafetyDevice"},

            # Physics
            "perpetual": {"ImpossibleDevice"},
            "infinite_energy": {"ImpossibleDevice"},
        }

    def add_pattern(self, pattern: ExtractionPattern) -> None:
        """Add an extraction pattern."""
        entity_type = pattern.entity_type
        if entity_type not in self.patterns:
            self.patterns[entity_type] = []
        self.patterns[entity_type].append(pattern)

    def add_custom_extractor(
        self,
        extractor: Callable[[str], List[Entity]]
    ) -> None:
        """Add a custom entity extractor function."""
        self.custom_extractors.append(extractor)

    def extract(self, chunk: Chunk) -> List[Entity]:
        """
        Extract all entities from a chunk.

        Args:
            chunk: Text chunk to extract from

        Returns:
            List of extracted entities
        """
        entities: List[Entity] = []
        text = chunk.text.lower()

        # Pattern-based extraction
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern.pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = self._create_entity_from_match(
                        match, pattern, chunk
                    )
                    entities.append(entity)

        # Custom extractors
        for extractor in self.custom_extractors:
            custom_entities = extractor(chunk.text)
            for e in custom_entities:
                e.source_chunk_id = chunk.id
            entities.extend(custom_entities)

        # Deduplicate and merge
        entities = self._deduplicate_entities(entities)

        return entities

    def extract_from_chunks(self, chunks: List[Chunk]) -> List[Entity]:
        """Extract entities from multiple chunks."""
        all_entities: List[Entity] = []

        for chunk in chunks:
            entities = self.extract(chunk)
            all_entities.extend(entities)

        return self._deduplicate_entities(all_entities)

    def _create_entity_from_match(
        self,
        match: re.Match,
        pattern: ExtractionPattern,
        chunk: Chunk
    ) -> Entity:
        """Create an entity from a regex match."""
        matched_text = match.group(0)

        # Generate deterministic ID from content
        id_input = f"{pattern.entity_type}:{matched_text}:{chunk.id}"
        entity_id = hashlib.md5(id_input.encode()).hexdigest()[:12]

        # Extract properties
        properties: Dict[str, Any] = {}

        if pattern.property_extractors:
            for prop_name, prop_value in pattern.property_extractors.items():
                if prop_value == "true":
                    properties[prop_name] = True
                elif prop_value == "false":
                    properties[prop_name] = False
                else:
                    properties[prop_name] = prop_value

        # Add context from surrounding text
        context = self._extract_context(chunk.text, match.start(), match.end())
        properties["_context"] = context
        properties["_matched_text"] = matched_text

        return Entity(
            id=entity_id,
            type=pattern.entity_type,
            name=matched_text,
            properties=properties,
            confidence=pattern.confidence,
            source_chunk_id=chunk.id,
            embedding=chunk.embedding,
        )

    def _extract_context(
        self,
        text: str,
        start: int,
        end: int,
        window: int = 50
    ) -> str:
        """Extract context window around a match."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate entities, merging properties."""
        seen: Dict[str, Entity] = {}

        for entity in entities:
            key = f"{entity.type}:{entity.name.lower()}"

            if key in seen:
                # Merge properties
                existing = seen[key]
                for prop, value in entity.properties.items():
                    if prop not in existing.properties:
                        existing.properties[prop] = value
                # Update confidence (max)
                existing.confidence = max(existing.confidence, entity.confidence)
            else:
                seen[key] = entity

        return list(seen.values())


class PropertyExtractor:
    """
    Extracts properties for entities from text.

    Works with the EntityExtractor to enrich entities with properties.
    """

    def __init__(self):
        self.property_patterns: Dict[str, List[Tuple[str, str]]] = {}
        self._setup_default_properties()

    def _setup_default_properties(self):
        """Setup default property extraction patterns."""
        # Weapon properties
        self.property_patterns["Weapon"] = [
            ("dangerous", r'\b(dangerous|lethal|deadly|harmful)\b'),
            ("caliber", r'(\d+)\s*(?:mm|caliber|cal)'),
            ("automatic", r'\b(automatic|semi-automatic|full-auto)\b'),
        ]

        # Drug properties
        self.property_patterns["Drug"] = [
            ("side_effects", r'\b(side\s+effects?|adverse\s+effects?)\b'),
            ("dosage", r'(\d+)\s*(?:mg|ml|g)\b'),
            ("fda_approved", r'\b(fda\s+approved|approved\s+by\s+fda)\b'),
        ]

        # Device properties
        self.property_patterns["Device"] = [
            ("voltage", r'(\d+)\s*(?:v|volt|volts)\b'),
            ("wattage", r'(\d+)\s*(?:w|watt|watts)\b'),
            ("safety_certified", r'\b(ul\s+listed|ce\s+certified|safety\s+certified)\b'),
        ]

    def extract_properties(
        self,
        entity: Entity,
        text: str
    ) -> Dict[str, Any]:
        """
        Extract additional properties for an entity from text.

        Args:
            entity: Entity to extract properties for
            text: Text to extract from

        Returns:
            Dictionary of extracted properties
        """
        properties: Dict[str, Any] = {}

        patterns = self.property_patterns.get(entity.type, [])

        for prop_name, pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Try to extract value from match
                if match.groups():
                    properties[prop_name] = match.group(1)
                else:
                    properties[prop_name] = True

        return properties

    def enrich_entities(
        self,
        entities: List[Entity],
        text: str
    ) -> List[Entity]:
        """Enrich all entities with additional properties."""
        for entity in entities:
            additional_props = self.extract_properties(entity, text)
            entity.properties.update(additional_props)

        return entities
