"""
World Model Builder

Builds local world models from retrieved chunks within a chart.
Orchestrates entity extraction, fact extraction, and rule loading.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from rewa.models import (
    Chart,
    Entity,
    Fact,
    Rule,
    LocalWorld,
    Vector,
)
from rewa.extraction.entity_extractor import EntityExtractor, Chunk, PropertyExtractor
from rewa.extraction.fact_extractor import FactExtractor, TemporalFactResolver
from rewa.rules.loader import RuleLoader, get_default_rules


@dataclass
class RetrievedChunk:
    """A chunk retrieved from a vector database or other source."""
    id: str
    text: str
    embedding: Optional[Vector] = None
    metadata: Optional[Dict[str, Any]] = None
    score: float = 1.0  # Retrieval score/similarity


class WorldModelBuilder:
    """
    Builds local world models from retrieved chunks.

    Implements the local world construction from PRD:
    1. Extract entities from chunks
    2. Extract facts about entities
    3. Load domain rules
    4. Assemble into LocalWorld
    """

    def __init__(
        self,
        entity_extractor: Optional[EntityExtractor] = None,
        fact_extractor: Optional[FactExtractor] = None,
        rule_loader: Optional[RuleLoader] = None,
        use_default_rules: bool = True,
    ):
        """
        Initialize the builder.

        Args:
            entity_extractor: Custom entity extractor
            fact_extractor: Custom fact extractor
            rule_loader: Custom rule loader
            use_default_rules: If True, include built-in rules
        """
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.fact_extractor = fact_extractor or FactExtractor()
        self.property_extractor = PropertyExtractor()
        self.temporal_resolver = TemporalFactResolver()
        self.rule_loader = rule_loader
        self.use_default_rules = use_default_rules

        # Cache for rules
        self._default_rules: Optional[List[Rule]] = None

    def build(
        self,
        chart: Chart,
        chunks: List[RetrievedChunk],
        additional_rules: Optional[List[Rule]] = None,
    ) -> LocalWorld:
        """
        Build a local world model for a chart.

        Args:
            chart: The chart defining the semantic region
            chunks: Retrieved text chunks
            additional_rules: Extra rules to include

        Returns:
            LocalWorld containing entities, facts, and rules
        """
        # Convert to internal Chunk format
        internal_chunks = [
            Chunk(
                id=c.id,
                text=c.text,
                embedding=c.embedding,
                metadata=c.metadata or {},
            )
            for c in chunks
        ]

        # Extract entities
        entities = self.entity_extractor.extract_from_chunks(internal_chunks)

        # Enrich entities with properties
        for chunk in internal_chunks:
            entities = self.property_extractor.enrich_entities(
                entities, chunk.text
            )

        # Extract facts
        all_facts: List[Fact] = []
        for chunk in internal_chunks:
            chunk_facts = self.fact_extractor.extract(
                chunk.text, entities, chunk.id
            )
            all_facts.extend(chunk_facts)

        # Resolve temporal conflicts
        resolved_facts = self.temporal_resolver.resolve(all_facts)

        # Load rules for this chart's domains
        rules = self._load_rules(chart.domain_tags)

        # Add additional rules if provided
        if additional_rules:
            rules.extend(additional_rules)

        # Track chunk IDs
        chunk_ids = {c.id for c in chunks}

        return LocalWorld(
            chart=chart,
            entities=entities,
            facts=resolved_facts,
            rules=rules,
            chunk_ids=chunk_ids,
        )

    def build_for_query(
        self,
        chart: Chart,
        chunks: List[RetrievedChunk],
        query_domains: Optional[Set[str]] = None,
    ) -> LocalWorld:
        """
        Build a world model optimized for query processing.

        Args:
            chart: The chart
            chunks: Retrieved chunks
            query_domains: Additional domains from query intent

        Returns:
            LocalWorld
        """
        # Combine chart domains with query domains
        all_domains = chart.domain_tags.copy()
        if query_domains:
            all_domains.update(query_domains)

        # Create a temporary chart with combined domains for rule loading
        extended_chart = Chart(
            id=chart.id,
            witness_embedding=chart.witness_embedding,
            radius=chart.radius,
            intrinsic_dim=chart.intrinsic_dim,
            domain_tags=all_domains,
            description=chart.description,
        )

        return self.build(extended_chart, chunks)

    def _load_rules(self, domains: Set[str]) -> List[Rule]:
        """Load rules for the given domains."""
        rules: List[Rule] = []

        # Load default rules if enabled
        if self.use_default_rules:
            if self._default_rules is None:
                self._default_rules = get_default_rules()

            # Filter default rules by domain
            for rule in self._default_rules:
                if not rule.domain_tags or rule.domain_tags & domains:
                    rules.append(rule)

        # Load from rule loader if available
        if self.rule_loader:
            loaded_rules = self.rule_loader.load_for_domains(domains)
            rules.extend(loaded_rules)

        return rules


class IncrementalWorldBuilder:
    """
    Builds world models incrementally as chunks arrive.

    Useful for streaming retrieval scenarios.
    """

    def __init__(self, base_builder: Optional[WorldModelBuilder] = None):
        """
        Initialize the incremental builder.

        Args:
            base_builder: Base builder to use
        """
        self.builder = base_builder or WorldModelBuilder()
        self._current_world: Optional[LocalWorld] = None
        self._processed_chunks: Set[str] = set()

    def initialize(self, chart: Chart) -> LocalWorld:
        """Initialize with an empty world."""
        self._current_world = LocalWorld(
            chart=chart,
            entities=[],
            facts=[],
            rules=self.builder._load_rules(chart.domain_tags),
            chunk_ids=set(),
        )
        self._processed_chunks = set()
        return self._current_world

    def add_chunk(self, chunk: RetrievedChunk) -> LocalWorld:
        """
        Add a chunk to the world model.

        Args:
            chunk: New chunk to process

        Returns:
            Updated LocalWorld
        """
        if self._current_world is None:
            raise ValueError("Must call initialize() first")

        if chunk.id in self._processed_chunks:
            return self._current_world

        # Build mini-world from single chunk
        mini_world = self.builder.build(
            self._current_world.chart,
            [chunk],
        )

        # Merge into current world
        self._merge_world(mini_world)
        self._processed_chunks.add(chunk.id)

        return self._current_world

    def add_chunks(self, chunks: List[RetrievedChunk]) -> LocalWorld:
        """Add multiple chunks."""
        for chunk in chunks:
            self.add_chunk(chunk)
        return self._current_world

    def get_world(self) -> Optional[LocalWorld]:
        """Get the current world model."""
        return self._current_world

    def _merge_world(self, mini_world: LocalWorld) -> None:
        """Merge a mini-world into the current world."""
        if self._current_world is None:
            return

        # Merge entities (deduplicate by ID)
        existing_ids = {e.id for e in self._current_world.entities}
        for entity in mini_world.entities:
            if entity.id not in existing_ids:
                self._current_world.entities.append(entity)
                existing_ids.add(entity.id)

        # Merge facts (deduplicate by ID)
        existing_fact_ids = {f.id for f in self._current_world.facts}
        for fact in mini_world.facts:
            if fact.id not in existing_fact_ids:
                self._current_world.facts.append(fact)
                existing_fact_ids.add(fact.id)

        # Merge chunk IDs
        self._current_world.chunk_ids.update(mini_world.chunk_ids)
