"""
World Model Merger

Merges multiple local world models, handling:
- Entity deduplication
- Fact merging and conflict resolution
- Cross-world consistency checking
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from rewa.models import (
    Chart,
    Entity,
    Fact,
    Rule,
    LocalWorld,
    Contradiction,
)
from rewa.validation.contradiction import (
    ContradictionDetector,
    CrossChartContradictionDetector,
)


@dataclass
class MergeResult:
    """Result of merging world models."""
    merged_world: LocalWorld
    source_worlds: List[LocalWorld]
    contradictions: List[Contradiction]
    cross_chart_conflicts: List[Tuple[str, str, Contradiction]]
    entities_merged: int
    facts_merged: int


class WorldModelMerger:
    """
    Merges multiple local world models.

    Used when a query spans multiple charts and we need
    to reason over combined information.
    """

    def __init__(
        self,
        detect_contradictions: bool = True,
        resolve_contradictions: bool = True,
    ):
        """
        Initialize the merger.

        Args:
            detect_contradictions: Whether to detect contradictions
            resolve_contradictions: Whether to attempt resolution
        """
        self.detect_contradictions = detect_contradictions
        self.resolve_contradictions = resolve_contradictions
        self.contradiction_detector = ContradictionDetector()
        self.cross_chart_detector = CrossChartContradictionDetector()

    def merge(
        self,
        worlds: List[LocalWorld],
        primary_chart: Optional[Chart] = None,
    ) -> MergeResult:
        """
        Merge multiple world models.

        Args:
            worlds: List of LocalWorld instances to merge
            primary_chart: Optional primary chart for the merged world

        Returns:
            MergeResult with merged world and diagnostics
        """
        if not worlds:
            raise ValueError("Cannot merge empty list of worlds")

        if len(worlds) == 1:
            return MergeResult(
                merged_world=worlds[0],
                source_worlds=worlds,
                contradictions=[],
                cross_chart_conflicts=[],
                entities_merged=0,
                facts_merged=0,
            )

        # Merge entities
        merged_entities, entity_merge_count = self._merge_entities(worlds)

        # Merge facts
        all_facts: List[Fact] = []
        for world in worlds:
            all_facts.extend(world.facts)

        # Update fact subject references to merged entities
        entity_map = {e.id: e for e in merged_entities}
        updated_facts = []
        for fact in all_facts:
            if fact.subject.id in entity_map:
                # Create fact with merged entity
                updated_fact = Fact(
                    id=fact.id,
                    subject=entity_map[fact.subject.id],
                    predicate=fact.predicate,
                    value=fact.value,
                    valid_time=fact.valid_time,
                    confidence=fact.confidence,
                    source_chunk_id=fact.source_chunk_id,
                    negated=fact.negated,
                )
                updated_facts.append(updated_fact)
            else:
                updated_facts.append(fact)

        # Deduplicate facts
        merged_facts, fact_merge_count = self._merge_facts(updated_facts)

        # Detect contradictions
        contradictions: List[Contradiction] = []
        if self.detect_contradictions:
            if self.resolve_contradictions:
                merged_facts, contradictions = self.contradiction_detector.detect_and_resolve(
                    merged_facts
                )
            else:
                contradictions = self.contradiction_detector.detect(merged_facts)

        # Detect cross-chart conflicts
        cross_chart_conflicts = []
        if self.detect_contradictions and len(worlds) > 1:
            facts_by_chart = {
                world.chart.id: world.facts
                for world in worlds
            }
            cross_chart_conflicts = self.cross_chart_detector.detect_cross_chart(
                facts_by_chart
            )

        # Merge rules (union, deduplicated)
        merged_rules = self._merge_rules(worlds)

        # Determine chart for merged world
        if primary_chart:
            chart = primary_chart
        else:
            chart = self._create_merged_chart(worlds)

        # Merge chunk IDs
        all_chunk_ids: Set[str] = set()
        for world in worlds:
            all_chunk_ids.update(world.chunk_ids)

        merged_world = LocalWorld(
            chart=chart,
            entities=merged_entities,
            facts=merged_facts,
            rules=merged_rules,
            chunk_ids=all_chunk_ids,
        )

        return MergeResult(
            merged_world=merged_world,
            source_worlds=worlds,
            contradictions=contradictions,
            cross_chart_conflicts=cross_chart_conflicts,
            entities_merged=entity_merge_count,
            facts_merged=fact_merge_count,
        )

    def _merge_entities(
        self,
        worlds: List[LocalWorld]
    ) -> Tuple[List[Entity], int]:
        """Merge entities from multiple worlds."""
        merged: Dict[str, Entity] = {}
        merge_count = 0

        for world in worlds:
            for entity in world.entities:
                if entity.id in merged:
                    # Merge properties and update confidence
                    existing = merged[entity.id]
                    for prop, value in entity.properties.items():
                        if prop not in existing.properties:
                            existing.properties[prop] = value
                    existing.confidence = max(existing.confidence, entity.confidence)
                    merge_count += 1
                else:
                    merged[entity.id] = entity

        return list(merged.values()), merge_count

    def _merge_facts(
        self,
        facts: List[Fact]
    ) -> Tuple[List[Fact], int]:
        """Merge and deduplicate facts."""
        merged: Dict[str, Fact] = {}
        merge_count = 0

        for fact in facts:
            if fact.id in merged:
                # Keep higher confidence
                if fact.confidence > merged[fact.id].confidence:
                    merged[fact.id] = fact
                merge_count += 1
            else:
                merged[fact.id] = fact

        return list(merged.values()), merge_count

    def _merge_rules(self, worlds: List[LocalWorld]) -> List[Rule]:
        """Merge rules from multiple worlds."""
        merged: Dict[str, Rule] = {}

        for world in worlds:
            for rule in world.rules:
                if rule.id not in merged:
                    merged[rule.id] = rule

        return list(merged.values())

    def _create_merged_chart(self, worlds: List[LocalWorld]) -> Chart:
        """Create a chart representing the merged world."""
        import numpy as np
        from rewa.geometry.spherical import compute_centroid, compute_coverage_radius

        # Combine all witness embeddings
        embeddings = [w.chart.witness_embedding for w in worlds]

        if len(embeddings) == 1:
            return worlds[0].chart

        # Compute centroid
        centroid = compute_centroid(embeddings)

        # Compute radius to cover all original charts
        max_dist = 0.0
        for world in worlds:
            from rewa.geometry.spherical import angular_distance
            dist = angular_distance(centroid, world.chart.witness_embedding)
            max_dist = max(max_dist, dist + world.chart.radius)

        # Union of domain tags
        all_domains: Set[str] = set()
        for world in worlds:
            all_domains.update(world.chart.domain_tags)

        # Average intrinsic dimension
        avg_dim = int(np.mean([w.chart.intrinsic_dim for w in worlds]))

        return Chart(
            id=f"merged_{'_'.join(w.chart.id[:8] for w in worlds)}",
            witness_embedding=centroid,
            radius=max_dist,
            intrinsic_dim=avg_dim,
            domain_tags=all_domains,
            description="Merged chart from multiple sources",
        )


class ConflictAwareWorldBuilder:
    """
    Builds merged world models with conflict awareness.

    Tracks conflicts and provides conflict resolution strategies.
    """

    def __init__(self, merger: Optional[WorldModelMerger] = None):
        """
        Initialize builder.

        Args:
            merger: Optional world model merger
        """
        self.merger = merger or WorldModelMerger()

    def build_with_conflict_analysis(
        self,
        worlds: List[LocalWorld]
    ) -> Tuple[LocalWorld, Dict[str, any]]:
        """
        Build merged world with detailed conflict analysis.

        Returns:
            (merged_world, conflict_analysis)
        """
        result = self.merger.merge(worlds)

        conflict_analysis = {
            "total_contradictions": len(result.contradictions),
            "cross_chart_conflicts": len(result.cross_chart_conflicts),
            "resolved_contradictions": sum(
                1 for c in result.contradictions if c.resolved_fact
            ),
            "unresolved_contradictions": sum(
                1 for c in result.contradictions if not c.resolved_fact
            ),
            "conflicting_chart_pairs": list(set(
                (c[0], c[1]) for c in result.cross_chart_conflicts
            )),
            "ambiguity_score": self._compute_ambiguity_score(result),
        }

        return result.merged_world, conflict_analysis

    def _compute_ambiguity_score(self, result: MergeResult) -> float:
        """Compute ambiguity score based on conflicts."""
        if not result.source_worlds:
            return 0.0

        factors = []

        # Factor: cross-chart conflicts
        if result.cross_chart_conflicts:
            factors.append(min(1.0, len(result.cross_chart_conflicts) * 0.2))

        # Factor: unresolved contradictions
        unresolved = sum(1 for c in result.contradictions if not c.resolved_fact)
        if unresolved:
            factors.append(min(1.0, unresolved * 0.3))

        # Factor: multiple source worlds
        if len(result.source_worlds) > 1:
            factors.append(min(0.5, (len(result.source_worlds) - 1) * 0.1))

        if not factors:
            return 0.0

        return min(1.0, sum(factors))
