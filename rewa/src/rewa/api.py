"""
REWA Main API

The single entry point for AI agents to interact with REWA.
Exposes one call: verify()
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

from rewa.models import (
    Chart,
    Entity,
    Fact,
    Rule,
    QueryIntent,
    LocalWorld,
    RewaResponse,
    RewaStatus,
    Contradiction,
    Impossibility,
    ValidationResult,
    Vector,
)
from rewa.geometry.charts import ChartManager
from rewa.geometry.selector import ChartSelector, ChartSelection
from rewa.extraction.entity_extractor import EntityExtractor
from rewa.extraction.fact_extractor import FactExtractor
from rewa.extraction.query_compiler import QueryCompiler, ImpossibilityPatternDetector
from rewa.rules.engine import RuleEngine
from rewa.rules.loader import RuleLoader, get_default_rules
from rewa.rules.solver import ConstraintSolver, IntentValidator
from rewa.validation.contradiction import ContradictionDetector
from rewa.validation.impossibility import ImpossibilityChecker
from rewa.world_model.builder import WorldModelBuilder, RetrievedChunk
from rewa.world_model.merger import WorldModelMerger


@dataclass
class RewaConfig:
    """Configuration for REWA."""
    # Chart selection
    similarity_threshold: float = 0.5
    secondary_threshold: float = 0.3
    max_charts: int = 5

    # Validation
    min_confidence: float = 0.5
    detect_contradictions: bool = True
    resolve_contradictions: bool = True

    # Rules
    use_default_rules: bool = True
    rule_directories: Optional[List[str]] = None

    # Ambiguity
    ambiguity_threshold: float = 0.5  # Above this, flag as ambiguous


class REWA:
    """
    REWA - Reasoning & Validation Layer for AI Agents.

    A geometric, local-world, constraint-satisfaction system that decides
    whether language should even be trusted.

    Usage:
        rewa = REWA()
        response = rewa.verify(query, retrieved_chunks)
        if response.status == RewaStatus.VALID:
            # Use response.safe_facts
        elif response.status == RewaStatus.IMPOSSIBLE:
            # Query is logically impossible
    """

    def __init__(
        self,
        config: Optional[RewaConfig] = None,
        embedder: Optional[Callable[[str], Vector]] = None,
    ):
        """
        Initialize REWA.

        Args:
            config: Optional configuration
            embedder: Optional function to compute embeddings
        """
        self.config = config or RewaConfig()
        self.embedder = embedder

        # Initialize components
        self.chart_manager = ChartManager()
        self.chart_selector = ChartSelector(
            self.chart_manager,
            similarity_threshold=self.config.similarity_threshold,
            secondary_threshold=self.config.secondary_threshold,
            max_charts=self.config.max_charts,
        )

        self.query_compiler = QueryCompiler()
        self.impossibility_pattern_detector = ImpossibilityPatternDetector()

        # Rule engine setup
        self.rule_engine = RuleEngine()
        if self.config.use_default_rules:
            self.rule_engine.add_rules(get_default_rules())

        if self.config.rule_directories:
            rule_loader = RuleLoader(self.config.rule_directories)
            for rule in rule_loader.load_all():
                self.rule_engine.add_rule(rule)

        self.constraint_solver = ConstraintSolver(self.rule_engine)
        self.intent_validator = IntentValidator()
        self.impossibility_checker = ImpossibilityChecker(self.rule_engine)
        self.contradiction_detector = ContradictionDetector()
        self.world_builder = WorldModelBuilder(
            use_default_rules=self.config.use_default_rules,
        )
        self.world_merger = WorldModelMerger(
            detect_contradictions=self.config.detect_contradictions,
            resolve_contradictions=self.config.resolve_contradictions,
        )

    def verify(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        query_embedding: Optional[Vector] = None,
    ) -> RewaResponse:
        """
        Verify retrieved information against a query.

        This is the main API entry point.

        Args:
            query: Natural language query string
            retrieved_chunks: List of chunks from retrieval, each containing:
                - "id": str
                - "text": str
                - "embedding": Optional[Vector]
                - "metadata": Optional[Dict]
                - "score": Optional[float]
            query_embedding: Optional pre-computed query embedding

        Returns:
            RewaResponse with validation results
        """
        # Convert chunks to internal format
        chunks = self._convert_chunks(retrieved_chunks)

        # Compile query intent
        if query_embedding is None and self.embedder:
            query_embedding = self.embedder(query)

        intent = self.query_compiler.compile(query, query_embedding)

        # Check for pattern-based impossibilities first
        pattern_impossibilities = self.impossibility_pattern_detector.detect(query)
        if pattern_impossibilities:
            return RewaResponse(
                status=RewaStatus.IMPOSSIBLE,
                safe_facts=[],
                impossibilities=[
                    Impossibility(reason=r) for r in pattern_impossibilities
                ],
                explanation=f"Query is logically impossible: {pattern_impossibilities[0]}",
                confidence=1.0,
            )

        # Validate intent for internal consistency
        intent_issues = self.intent_validator.validate(intent)
        if intent_issues:
            return RewaResponse(
                status=RewaStatus.IMPOSSIBLE,
                safe_facts=[],
                impossibilities=[
                    Impossibility(reason=issue) for issue in intent_issues
                ],
                explanation=f"Query contains contradictions: {intent_issues[0]}",
                confidence=1.0,
            )

        # Select charts based on query
        chart_selection = self._select_charts(intent, query_embedding, chunks)

        # Check for ambiguity
        if chart_selection.ambiguity_score > self.config.ambiguity_threshold:
            if len(chart_selection.primary_charts) > 1:
                return self._handle_ambiguity(intent, chart_selection, chunks)

        # Build local world models
        worlds = self._build_worlds(chart_selection, chunks, intent)

        if not worlds:
            return RewaResponse(
                status=RewaStatus.INSUFFICIENT_EVIDENCE,
                safe_facts=[],
                explanation="No relevant information found in retrieved content",
                confidence=0.0,
            )

        # Merge worlds if multiple
        if len(worlds) > 1:
            merge_result = self.world_merger.merge(worlds)
            merged_world = merge_result.merged_world
            contradictions = merge_result.contradictions

            # Check for unresolved contradictions
            unresolved = [c for c in contradictions if not c.resolved_fact]
            if unresolved:
                return RewaResponse(
                    status=RewaStatus.CONFLICT,
                    safe_facts=self._get_safe_facts(merged_world, intent),
                    contradictions=unresolved,
                    explanation=f"Found {len(unresolved)} unresolved contradictions",
                    confidence=0.5,
                )
        else:
            merged_world = worlds[0]
            contradictions = self.contradiction_detector.detect(merged_world.facts)

        # Check for impossibilities based on rules
        domains = self._get_domains(intent, chart_selection)
        impossibilities = self.impossibility_checker.check(intent, domains)
        if impossibilities:
            return RewaResponse(
                status=RewaStatus.IMPOSSIBLE,
                safe_facts=[],
                impossibilities=impossibilities,
                explanation=impossibilities[0].explanation or impossibilities[0].reason,
                confidence=1.0,
            )

        # Validate entities against intent
        validation_results = self.constraint_solver.validate_entities(
            merged_world.entities,
            merged_world.facts,
            intent,
            domains,
        )

        # Find satisfying entities
        satisfying = [r for r in validation_results if r.satisfied]

        if not satisfying:
            # No entities satisfy the query
            return RewaResponse(
                status=RewaStatus.INSUFFICIENT_EVIDENCE,
                safe_facts=[],
                validation_results=validation_results,
                contradictions=contradictions,
                explanation="No entities satisfy all query requirements",
                confidence=0.3,
            )

        # Collect safe facts from satisfying entities
        safe_facts = []
        for result in satisfying:
            entity_facts = merged_world.get_facts_for_entity(result.entity.id)
            safe_facts.extend(entity_facts)

        # Compute overall confidence
        avg_confidence = sum(r.confidence for r in satisfying) / len(satisfying)

        return RewaResponse(
            status=RewaStatus.VALID,
            safe_facts=safe_facts,
            contradictions=contradictions,
            validation_results=satisfying,
            explanation=f"Found {len(satisfying)} entities satisfying query requirements",
            confidence=avg_confidence,
        )

    def add_chart(self, chart: Chart) -> None:
        """Add a chart to the semantic space."""
        self.chart_manager.add_chart(chart)

    def add_rule(self, rule: Rule) -> None:
        """Add a domain rule."""
        self.rule_engine.add_rule(rule)

    def add_rules(self, rules: List[Rule]) -> None:
        """Add multiple domain rules."""
        self.rule_engine.add_rules(rules)

    def _convert_chunks(
        self,
        chunks: List[Dict[str, Any]]
    ) -> List[RetrievedChunk]:
        """Convert external chunk format to internal."""
        return [
            RetrievedChunk(
                id=c.get("id", str(i)),
                text=c.get("text", ""),
                embedding=c.get("embedding"),
                metadata=c.get("metadata"),
                score=c.get("score", 1.0),
            )
            for i, c in enumerate(chunks)
        ]

    def _select_charts(
        self,
        intent: QueryIntent,
        query_embedding: Optional[Vector],
        chunks: List[RetrievedChunk],
    ) -> ChartSelection:
        """Select relevant charts for the query."""
        if len(self.chart_manager) == 0:
            # No charts defined - create default chart from chunks
            self._create_default_chart(chunks, intent)

        if query_embedding is not None:
            return self.chart_selector.select_charts(
                query_embedding,
                list(intent.required_types) if intent.required_types else None,
            )

        # Fallback: use first chunk embedding or select all charts
        for chunk in chunks:
            if chunk.embedding is not None:
                return self.chart_selector.select_charts(chunk.embedding)

        # No embeddings available - return all charts
        return ChartSelection(
            primary_charts=self.chart_manager.get_all_charts(),
            secondary_charts=[],
            ambiguity_score=0.0,
            overlap_regions=[],
        )

    def _create_default_chart(
        self,
        chunks: List[RetrievedChunk],
        intent: QueryIntent,
    ) -> None:
        """Create a default chart from chunks when none exist."""
        import numpy as np
        from rewa.geometry.charts import create_chart_from_embeddings

        # Collect embeddings
        embeddings = [c.embedding for c in chunks if c.embedding is not None]

        if not embeddings:
            # Create a random embedding for default chart
            # In practice, you'd use the embedder
            embeddings = [np.random.randn(768) for _ in range(3)]
            for i, emb in enumerate(embeddings):
                embeddings[i] = emb / np.linalg.norm(emb)

        # Create chart with domains from intent
        domains = intent.required_types or {"general"}

        chart = create_chart_from_embeddings(
            embeddings,
            domain_tags=domains,
            chart_id="default",
            description="Auto-generated default chart",
        )

        self.chart_manager.add_chart(chart)

    def _build_worlds(
        self,
        selection: ChartSelection,
        chunks: List[RetrievedChunk],
        intent: QueryIntent,
    ) -> List[LocalWorld]:
        """Build local worlds for selected charts."""
        worlds = []

        all_charts = selection.primary_charts + selection.secondary_charts

        for chart in all_charts:
            # Filter chunks to those in this chart
            chart_chunks = self._filter_chunks_for_chart(chunks, chart)

            if chart_chunks:
                world = self.world_builder.build_for_query(
                    chart, chart_chunks, intent.required_types
                )
                worlds.append(world)

        return worlds

    def _filter_chunks_for_chart(
        self,
        chunks: List[RetrievedChunk],
        chart: Chart,
    ) -> List[RetrievedChunk]:
        """Filter chunks to those that belong to a chart."""
        if not chunks:
            return []

        # If chunks don't have embeddings, return all
        if all(c.embedding is None for c in chunks):
            return chunks

        # Filter by chart containment
        filtered = []
        for chunk in chunks:
            if chunk.embedding is not None:
                if chart.contains_embedding(chunk.embedding, chart.radius * 1.5):
                    filtered.append(chunk)
            else:
                # Include chunks without embeddings
                filtered.append(chunk)

        return filtered if filtered else chunks

    def _handle_ambiguity(
        self,
        intent: QueryIntent,
        selection: ChartSelection,
        chunks: List[RetrievedChunk],
    ) -> RewaResponse:
        """Handle ambiguous query that spans multiple charts."""
        # Build worlds for all charts
        worlds = self._build_worlds(selection, chunks, intent)

        # Collect all facts but mark as ambiguous
        all_facts = []
        for world in worlds:
            all_facts.extend(world.facts)

        return RewaResponse(
            status=RewaStatus.AMBIGUOUS,
            safe_facts=all_facts,  # Include but mark status as ambiguous
            ambiguous_charts=selection.primary_charts + selection.secondary_charts,
            explanation=f"Query is ambiguous across {len(selection.primary_charts)} semantic regions. "
                       f"Consider clarifying the domain.",
            confidence=1.0 - selection.ambiguity_score,
        )

    def _get_safe_facts(
        self,
        world: LocalWorld,
        intent: QueryIntent,
    ) -> List[Fact]:
        """Get facts that are safe to use (high confidence, no contradictions)."""
        safe = []

        for fact in world.facts:
            if fact.confidence >= self.config.min_confidence:
                safe.append(fact)

        return safe

    def _get_domains(
        self,
        intent: QueryIntent,
        selection: ChartSelection,
    ) -> Set[str]:
        """Get all relevant domains."""
        domains: Set[str] = set()

        # From intent
        if intent.required_types:
            domains.update(intent.required_types)

        # From charts
        for chart in selection.primary_charts + selection.secondary_charts:
            domains.update(chart.domain_tags)

        return domains


# Convenience functions for simple usage

def create_rewa(
    config: Optional[RewaConfig] = None,
    embedder: Optional[Callable[[str], Vector]] = None,
) -> REWA:
    """Create a REWA instance with default configuration."""
    return REWA(config, embedder)


def verify(
    query: str,
    chunks: List[Dict[str, Any]],
    embedder: Optional[Callable[[str], Vector]] = None,
) -> RewaResponse:
    """Quick verification without creating REWA instance."""
    rewa = REWA(embedder=embedder)
    return rewa.verify(query, chunks)
