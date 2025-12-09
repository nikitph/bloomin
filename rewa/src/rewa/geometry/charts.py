"""
Chart Management

Charts are local semantic neighborhoods on the embedding sphere.
They approximate local semantic worlds and enable local reasoning.
"""

import uuid
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from rewa.models import Chart, Vector
from rewa.geometry.spherical import (
    normalize_embedding,
    compute_centroid,
    compute_coverage_radius,
    estimate_intrinsic_dimension,
    angular_distance,
    cosine_similarity,
)


def create_chart_from_embeddings(
    embeddings: List[Vector],
    domain_tags: Optional[Set[str]] = None,
    coverage: float = 0.95,
    chart_id: Optional[str] = None,
    description: str = ""
) -> Chart:
    """
    Create a chart from a set of embeddings.

    Args:
        embeddings: List of normalized embeddings in this semantic region
        domain_tags: Tags describing the domain (e.g., {"medical", "drugs"})
        coverage: Fraction of embeddings the radius should cover
        chart_id: Optional explicit ID
        description: Human-readable description

    Returns:
        A Chart centered on the embeddings
    """
    if not embeddings:
        raise ValueError("Cannot create chart from empty embeddings")

    # Normalize all embeddings
    normalized = [normalize_embedding(e) for e in embeddings]

    # Compute centroid (witness embedding)
    witness = compute_centroid(normalized)

    # Compute radius to cover specified fraction
    radius = compute_coverage_radius(normalized, witness, coverage)

    # Estimate intrinsic dimensionality
    intrinsic_dim = int(estimate_intrinsic_dimension(normalized))

    return Chart(
        id=chart_id or str(uuid.uuid4()),
        witness_embedding=witness,
        radius=radius,
        intrinsic_dim=intrinsic_dim,
        domain_tags=domain_tags or set(),
        description=description,
    )


def detect_chart_overlap(chart_a: Chart, chart_b: Chart) -> Tuple[bool, float]:
    """
    Detect if two charts overlap and compute overlap degree.

    Returns:
        (overlaps, overlap_degree) where overlap_degree is in [0, 1]
    """
    # Angular distance between centers
    center_dist = angular_distance(
        chart_a.witness_embedding,
        chart_b.witness_embedding
    )

    # Charts overlap if distance < sum of radii
    max_dist = chart_a.radius + chart_b.radius
    overlaps = center_dist < max_dist

    if not overlaps:
        return False, 0.0

    # Compute overlap degree (higher = more overlap)
    # 1.0 when centers are same, 0.0 when just touching
    overlap_degree = 1.0 - (center_dist / max_dist)

    return True, overlap_degree


class ChartManager:
    """
    Manages a collection of charts for semantic space coverage.

    Handles chart creation, lookup, and overlap detection.
    """

    def __init__(self):
        self.charts: Dict[str, Chart] = {}
        self._domain_index: Dict[str, Set[str]] = {}  # domain_tag -> chart_ids

    def add_chart(self, chart: Chart) -> None:
        """Add a chart to the manager."""
        self.charts[chart.id] = chart

        # Update domain index
        for tag in chart.domain_tags:
            if tag not in self._domain_index:
                self._domain_index[tag] = set()
            self._domain_index[tag].add(chart.id)

    def remove_chart(self, chart_id: str) -> None:
        """Remove a chart from the manager."""
        if chart_id not in self.charts:
            return

        chart = self.charts[chart_id]

        # Update domain index
        for tag in chart.domain_tags:
            if tag in self._domain_index:
                self._domain_index[tag].discard(chart_id)

        del self.charts[chart_id]

    def get_chart(self, chart_id: str) -> Optional[Chart]:
        """Get a chart by ID."""
        return self.charts.get(chart_id)

    def get_charts_by_domain(self, domain_tag: str) -> List[Chart]:
        """Get all charts with a specific domain tag."""
        chart_ids = self._domain_index.get(domain_tag, set())
        return [self.charts[cid] for cid in chart_ids if cid in self.charts]

    def find_containing_charts(
        self,
        embedding: Vector,
        threshold_multiplier: float = 1.0
    ) -> List[Tuple[Chart, float]]:
        """
        Find all charts that contain a given embedding.

        Args:
            embedding: Query embedding (normalized)
            threshold_multiplier: Multiply chart radius by this factor

        Returns:
            List of (chart, similarity) tuples, sorted by similarity descending
        """
        embedding = normalize_embedding(embedding)
        results = []

        for chart in self.charts.values():
            threshold = chart.radius * threshold_multiplier
            if chart.contains_embedding(embedding, threshold):
                sim = chart.similarity_to(embedding)
                results.append((chart, sim))

        # Sort by similarity (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def find_overlapping_charts(self, chart_id: str) -> List[Tuple[Chart, float]]:
        """
        Find all charts that overlap with a given chart.

        Returns:
            List of (chart, overlap_degree) tuples
        """
        if chart_id not in self.charts:
            return []

        source_chart = self.charts[chart_id]
        results = []

        for other_id, other_chart in self.charts.items():
            if other_id == chart_id:
                continue

            overlaps, degree = detect_chart_overlap(source_chart, other_chart)
            if overlaps:
                results.append((other_chart, degree))

        # Sort by overlap degree (highest first)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def find_nearest_charts(
        self,
        embedding: Vector,
        k: int = 5
    ) -> List[Tuple[Chart, float]]:
        """
        Find the k nearest charts to an embedding.

        Returns:
            List of (chart, similarity) tuples, sorted by similarity descending
        """
        embedding = normalize_embedding(embedding)
        results = []

        for chart in self.charts.values():
            sim = chart.similarity_to(embedding)
            results.append((chart, sim))

        # Sort by similarity and take top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def create_chart_hierarchy(self) -> Dict[str, List[str]]:
        """
        Build a hierarchy of charts based on containment.

        Returns:
            Dict mapping chart_id -> list of child chart_ids
        """
        hierarchy: Dict[str, List[str]] = {cid: [] for cid in self.charts}

        for chart_id, chart in self.charts.items():
            for other_id, other_chart in self.charts.items():
                if chart_id == other_id:
                    continue

                # Check if other_chart's center is within this chart
                if chart.contains_embedding(other_chart.witness_embedding):
                    # Check if other is smaller (potential child)
                    if other_chart.radius < chart.radius * 0.8:
                        hierarchy[chart_id].append(other_id)

        return hierarchy

    def merge_charts(
        self,
        chart_ids: List[str],
        new_domain_tags: Optional[Set[str]] = None
    ) -> Chart:
        """
        Merge multiple charts into a single larger chart.

        Args:
            chart_ids: IDs of charts to merge
            new_domain_tags: Tags for the merged chart

        Returns:
            New merged chart
        """
        charts_to_merge = [
            self.charts[cid] for cid in chart_ids
            if cid in self.charts
        ]

        if not charts_to_merge:
            raise ValueError("No valid charts to merge")

        # Combine witness embeddings
        embeddings = [c.witness_embedding for c in charts_to_merge]

        # Union of domain tags if not specified
        if new_domain_tags is None:
            new_domain_tags = set()
            for c in charts_to_merge:
                new_domain_tags.update(c.domain_tags)

        # Create new chart
        new_witness = compute_centroid(embeddings)

        # New radius should cover all original charts
        max_dist = 0.0
        for c in charts_to_merge:
            dist_to_center = angular_distance(new_witness, c.witness_embedding)
            coverage_dist = dist_to_center + c.radius
            max_dist = max(max_dist, coverage_dist)

        # Average intrinsic dimension
        avg_dim = int(np.mean([c.intrinsic_dim for c in charts_to_merge]))

        return Chart(
            id=str(uuid.uuid4()),
            witness_embedding=new_witness,
            radius=max_dist,
            intrinsic_dim=avg_dim,
            domain_tags=new_domain_tags,
            description=f"Merged from {len(charts_to_merge)} charts",
        )

    def get_all_charts(self) -> List[Chart]:
        """Get all charts."""
        return list(self.charts.values())

    def __len__(self) -> int:
        return len(self.charts)

    def __iter__(self):
        return iter(self.charts.values())
