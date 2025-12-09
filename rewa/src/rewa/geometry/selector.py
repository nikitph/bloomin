"""
Chart Selector

Selects relevant charts for a given query embedding.
Multiple charts may be selected to preserve ambiguity.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

from rewa.models import Chart, Vector, QueryIntent
from rewa.geometry.charts import ChartManager
from rewa.geometry.spherical import (
    normalize_embedding,
    cosine_similarity,
    angular_distance,
)


@dataclass
class ChartSelection:
    """Result of chart selection."""
    primary_charts: List[Chart]  # Charts that definitely contain query
    secondary_charts: List[Chart]  # Charts near query (for ambiguity)
    ambiguity_score: float  # How ambiguous the selection is (0-1)
    overlap_regions: List[Tuple[Chart, Chart, float]]  # Overlapping chart pairs


class ChartSelector:
    """
    Selects charts for query processing.

    Implements the chart selection algorithm from the PRD:
    - Multiple charts are kept
    - Ambiguity is preserved
    """

    def __init__(
        self,
        chart_manager: ChartManager,
        similarity_threshold: float = 0.5,
        secondary_threshold: float = 0.3,
        max_charts: int = 5,
    ):
        """
        Initialize the chart selector.

        Args:
            chart_manager: Manager containing all charts
            similarity_threshold: Minimum similarity for primary selection
            secondary_threshold: Minimum similarity for secondary selection
            max_charts: Maximum total charts to return
        """
        self.chart_manager = chart_manager
        self.similarity_threshold = similarity_threshold
        self.secondary_threshold = secondary_threshold
        self.max_charts = max_charts

    def select_charts(
        self,
        query_embedding: Vector,
        domain_filter: Optional[List[str]] = None,
    ) -> ChartSelection:
        """
        Select charts for a query embedding.

        Args:
            query_embedding: The query embedding (will be normalized)
            domain_filter: Optional list of domain tags to filter by

        Returns:
            ChartSelection with primary and secondary charts
        """
        query_embedding = normalize_embedding(query_embedding)

        # Get all candidate charts
        if domain_filter:
            candidates = []
            for tag in domain_filter:
                candidates.extend(self.chart_manager.get_charts_by_domain(tag))
            # Remove duplicates while preserving order
            seen = set()
            unique_candidates = []
            for c in candidates:
                if c.id not in seen:
                    seen.add(c.id)
                    unique_candidates.append(c)
            candidates = unique_candidates
        else:
            candidates = self.chart_manager.get_all_charts()

        if not candidates:
            return ChartSelection(
                primary_charts=[],
                secondary_charts=[],
                ambiguity_score=0.0,
                overlap_regions=[],
            )

        # Score all candidates
        scored_charts: List[Tuple[Chart, float]] = []
        for chart in candidates:
            similarity = chart.similarity_to(query_embedding)
            # Adjust threshold based on chart radius
            # Larger charts should be easier to match
            effective_threshold = self._effective_threshold(chart)
            if similarity >= self.secondary_threshold:
                scored_charts.append((chart, similarity))

        # Sort by similarity
        scored_charts.sort(key=lambda x: x[1], reverse=True)

        # Classify into primary and secondary
        primary_charts = []
        secondary_charts = []

        for chart, similarity in scored_charts:
            effective_threshold = self._effective_threshold(chart)

            if similarity >= effective_threshold:
                primary_charts.append(chart)
            elif similarity >= self.secondary_threshold:
                secondary_charts.append(chart)

        # Apply max_charts limit
        total_charts = len(primary_charts) + len(secondary_charts)
        if total_charts > self.max_charts:
            # Prioritize primary charts
            if len(primary_charts) >= self.max_charts:
                primary_charts = primary_charts[:self.max_charts]
                secondary_charts = []
            else:
                remaining = self.max_charts - len(primary_charts)
                secondary_charts = secondary_charts[:remaining]

        # Detect overlaps
        all_selected = primary_charts + secondary_charts
        overlap_regions = self._find_overlaps(all_selected)

        # Compute ambiguity score
        ambiguity_score = self._compute_ambiguity(
            primary_charts, secondary_charts, overlap_regions
        )

        return ChartSelection(
            primary_charts=primary_charts,
            secondary_charts=secondary_charts,
            ambiguity_score=ambiguity_score,
            overlap_regions=overlap_regions,
        )

    def select_for_intent(self, intent: QueryIntent) -> ChartSelection:
        """
        Select charts based on a QueryIntent.

        Uses the intent's embedding and can filter by required types.
        """
        if intent.embedding is None:
            raise ValueError("QueryIntent must have an embedding for chart selection")

        # Derive domain filter from required types if present
        domain_filter = None
        if intent.required_types:
            # Map types to domains (this could be more sophisticated)
            domain_filter = list(intent.required_types)

        return self.select_charts(intent.embedding, domain_filter)

    def _effective_threshold(self, chart: Chart) -> float:
        """
        Compute effective similarity threshold for a chart.

        Larger charts (larger radius) get lower thresholds.
        """
        # Base threshold adjusted by chart size
        # Radius in radians: 0 (point) to pi (hemisphere)
        # Normalize to [0, 1] and adjust threshold
        normalized_radius = chart.radius / np.pi
        threshold_reduction = normalized_radius * 0.2  # Up to 20% reduction

        return max(0.2, self.similarity_threshold - threshold_reduction)

    def _find_overlaps(
        self,
        charts: List[Chart]
    ) -> List[Tuple[Chart, Chart, float]]:
        """Find all overlapping pairs among selected charts."""
        overlaps = []

        for i, chart_a in enumerate(charts):
            for chart_b in charts[i + 1:]:
                # Compute distance between centers
                center_dist = angular_distance(
                    chart_a.witness_embedding,
                    chart_b.witness_embedding
                )

                # Check if they overlap
                max_dist = chart_a.radius + chart_b.radius
                if center_dist < max_dist:
                    overlap_degree = 1.0 - (center_dist / max_dist)
                    overlaps.append((chart_a, chart_b, overlap_degree))

        return overlaps

    def _compute_ambiguity(
        self,
        primary_charts: List[Chart],
        secondary_charts: List[Chart],
        overlap_regions: List[Tuple[Chart, Chart, float]],
    ) -> float:
        """
        Compute ambiguity score for the selection.

        High ambiguity when:
        - Multiple primary charts exist
        - Primary and secondary charts have different domains
        - Significant overlaps exist
        """
        if not primary_charts:
            return 0.0 if not secondary_charts else 1.0

        factors = []

        # Factor 1: Multiple primary charts
        if len(primary_charts) > 1:
            factors.append(min(1.0, (len(primary_charts) - 1) * 0.3))

        # Factor 2: Domain diversity
        all_domains: set = set()
        primary_domains: set = set()
        for c in primary_charts:
            all_domains.update(c.domain_tags)
            primary_domains.update(c.domain_tags)
        for c in secondary_charts:
            all_domains.update(c.domain_tags)

        if len(all_domains) > len(primary_domains):
            domain_diversity = 1.0 - (len(primary_domains) / len(all_domains))
            factors.append(domain_diversity * 0.5)

        # Factor 3: Overlap intensity
        if overlap_regions:
            avg_overlap = np.mean([o[2] for o in overlap_regions])
            factors.append(avg_overlap * 0.4)

        if not factors:
            return 0.0

        return min(1.0, sum(factors))


def adaptive_chart_threshold(chart: Chart, base_threshold: float = 0.5) -> float:
    """
    Compute adaptive similarity threshold for a chart.

    Takes into account:
    - Chart radius (larger = lower threshold)
    - Intrinsic dimension (higher = stricter matching)
    """
    # Radius factor: larger charts are easier to match
    radius_factor = 1.0 - (chart.radius / np.pi) * 0.3

    # Dimension factor: high dimension = stricter
    # Assume embedding dim of ~768, intrinsic dim typically much lower
    dim_factor = 1.0 + (chart.intrinsic_dim / 100) * 0.1

    return base_threshold * radius_factor * dim_factor
