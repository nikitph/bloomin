"""
Experiment 1: Impossible Query Detection

Tests REWA's ability to detect logically impossible queries.

Dataset: Manually curated paradoxes
- "Drug that cures cancer with zero side effects"
- "Perpetual motion machine"
- "Square circle"

Success Criteria:
- REWA > 90% detection rate
- Baseline (RAG-only) < 20%
"""

from typing import List

from rewa.models import RewaStatus
from rewa.experiments.base import BaseExperiment, TestCase


class ImpossibleQueryExperiment(BaseExperiment):
    """Tests impossible query detection capability."""

    @property
    def name(self) -> str:
        return "impossible_query_detection"

    @property
    def description(self) -> str:
        return "Tests REWA's ability to detect logically impossible queries"

    def get_test_cases(self) -> List[TestCase]:
        """Generate impossible query test cases."""
        cases = []

        # Medical impossibilities
        cases.append(TestCase(
            id="med_001",
            query="Find a drug that cures cancer with zero side effects",
            chunks=[
                {
                    "id": "c1",
                    "text": "CancerCure-X is a revolutionary treatment that cures "
                           "all forms of cancer. Clinical trials show 95% remission "
                           "rates with no reported side effects whatsoever.",
                },
                {
                    "id": "c2",
                    "text": "Cancer treatment typically involves chemotherapy or "
                           "radiation, both of which have significant side effects.",
                },
            ],
            expected_status=RewaStatus.IMPOSSIBLE,
            description="Cancer cure without side effects is medically impossible",
            tags=["medical", "cancer", "impossibility"],
        ))

        cases.append(TestCase(
            id="med_002",
            query="Drug with no side effects that completely eliminates tumors",
            chunks=[
                {
                    "id": "c1",
                    "text": "MiraclePill eliminates tumors completely with its "
                           "natural formula. Users report feeling great with "
                           "absolutely no negative effects.",
                },
            ],
            expected_status=RewaStatus.IMPOSSIBLE,
            description="Tumor elimination without side effects is impossible",
            tags=["medical", "impossibility"],
        ))

        # Physical impossibilities
        cases.append(TestCase(
            id="phys_001",
            query="Find a perpetual motion machine design",
            chunks=[
                {
                    "id": "c1",
                    "text": "The Smith Perpetual Motion Engine uses magnetic "
                           "fields to generate infinite energy without any fuel "
                           "input. It runs forever.",
                },
            ],
            expected_status=RewaStatus.IMPOSSIBLE,
            description="Perpetual motion violates thermodynamics",
            tags=["physics", "impossibility"],
        ))

        cases.append(TestCase(
            id="phys_002",
            query="Device that creates infinite energy from nothing",
            chunks=[
                {
                    "id": "c1",
                    "text": "The Zero-Point Generator extracts unlimited energy "
                           "from vacuum fluctuations, providing infinite power "
                           "with no input required.",
                },
            ],
            expected_status=RewaStatus.IMPOSSIBLE,
            description="Infinite energy violates conservation laws",
            tags=["physics", "impossibility"],
        ))

        cases.append(TestCase(
            id="phys_003",
            query="A machine that travels faster than the speed of light",
            chunks=[
                {
                    "id": "c1",
                    "text": "The WarpDrive X can achieve speeds of 10x the "
                           "speed of light using quantum tunneling principles.",
                },
            ],
            expected_status=RewaStatus.IMPOSSIBLE,
            description="FTL travel violates relativity",
            tags=["physics", "impossibility"],
        ))

        # Logical impossibilities
        cases.append(TestCase(
            id="logic_001",
            query="Show me a square circle",
            chunks=[
                {
                    "id": "c1",
                    "text": "The revolutionary SquareCircle shape combines the "
                           "properties of both squares and circles, having four "
                           "equal sides while maintaining a constant radius.",
                },
            ],
            expected_status=RewaStatus.IMPOSSIBLE,
            description="Square circle is a logical contradiction",
            tags=["logic", "impossibility"],
        ))

        cases.append(TestCase(
            id="logic_002",
            query="Find a married bachelor",
            chunks=[
                {
                    "id": "c1",
                    "text": "John is a married bachelor who lives alone. He "
                           "enjoys his single life while also being in a "
                           "committed marriage.",
                },
            ],
            expected_status=RewaStatus.IMPOSSIBLE,
            description="Married bachelor is a contradiction",
            tags=["logic", "impossibility"],
        ))

        # Valid queries (negative cases - should NOT be flagged as impossible)
        cases.append(TestCase(
            id="valid_001",
            query="Find cancer treatment options",
            chunks=[
                {
                    "id": "c1",
                    "text": "Chemotherapy is a common cancer treatment that uses "
                           "drugs to kill cancer cells. Side effects include "
                           "nausea, hair loss, and fatigue.",
                },
                {
                    "id": "c2",
                    "text": "Immunotherapy helps the immune system fight cancer. "
                           "Side effects are generally milder than chemotherapy "
                           "but can include fatigue and skin reactions.",
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Valid cancer treatment query",
            tags=["medical", "valid"],
        ))

        cases.append(TestCase(
            id="valid_002",
            query="Find a renewable energy source",
            chunks=[
                {
                    "id": "c1",
                    "text": "Solar panels convert sunlight into electricity. "
                           "They are a sustainable energy source with no "
                           "direct emissions during operation.",
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Valid energy query",
            tags=["physics", "valid"],
        ))

        cases.append(TestCase(
            id="valid_003",
            query="Find a circular shape",
            chunks=[
                {
                    "id": "c1",
                    "text": "A circle is a shape where all points are equidistant "
                           "from a center point. It has no corners or edges.",
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Valid geometry query",
            tags=["logic", "valid"],
        ))

        return cases


def get_impossible_queries() -> List[str]:
    """Get list of impossible queries for testing."""
    return [
        "Drug that cures cancer with zero side effects",
        "Perpetual motion machine",
        "Square circle",
        "Married bachelor",
        "Device that creates infinite energy from nothing",
        "Faster than light travel device",
        "Object that is both completely red and completely blue",
        "Triangle with four sides",
        "Liquid that is colder than absolute zero",
        "Weight loss pill with no diet changes and guaranteed results",
    ]


def get_valid_queries() -> List[str]:
    """Get list of valid queries that should NOT be flagged."""
    return [
        "Cancer treatment options",
        "Renewable energy sources",
        "Circular shapes",
        "Single person living alone",
        "Battery-powered device",
        "Normal speed vehicle",
        "Red colored object",
        "Three-sided shape",
        "Cold liquid",
        "Weight loss methods",
    ]
