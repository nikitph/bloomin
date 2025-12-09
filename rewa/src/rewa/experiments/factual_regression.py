"""
Experiment 5: Factual Regression Safety

Ensures REWA never:
- Removes correct information
- Invents new facts

Metric: Δ hallucination rate vs baseline

This is a safety check to ensure REWA improves rather than
degrades factual accuracy.
"""

from typing import List, Dict, Any, Set, Tuple

from rewa.models import RewaStatus, Fact
from rewa.experiments.base import BaseExperiment, TestCase, ExperimentResult, ExperimentMetrics


class FactualRegressionExperiment(BaseExperiment):
    """Tests that REWA doesn't introduce factual regression."""

    @property
    def name(self) -> str:
        return "factual_regression_safety"

    @property
    def description(self) -> str:
        return "Ensures REWA preserves correct facts and doesn't invent new ones"

    def get_test_cases(self) -> List[TestCase]:
        """Generate factual regression test cases."""
        cases = []

        # Case 1: Correct facts should be preserved
        cases.append(TestCase(
            id="preserve_001",
            query="What is the capital of France?",
            chunks=[
                {
                    "id": "c1",
                    "text": "Paris is the capital of France. It has been the "
                           "capital since the 10th century. Paris is home to "
                           "the Eiffel Tower and Louvre Museum.",
                },
            ],
            expected_status=RewaStatus.VALID,
            expected_properties={"safe_facts": True},  # Should have facts
            description="Correct geographical facts should be preserved",
            tags=["factual", "preserve"],
        ))

        cases.append(TestCase(
            id="preserve_002",
            query="What are the side effects of aspirin?",
            chunks=[
                {
                    "id": "c1",
                    "text": "Aspirin (acetylsalicylic acid) can cause stomach "
                           "irritation, bleeding, and allergic reactions. It "
                           "should not be given to children due to Reye's syndrome risk.",
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Correct medical facts should be preserved",
            tags=["factual", "preserve", "medical"],
        ))

        # Case 2: Facts should not be invented
        cases.append(TestCase(
            id="no_invent_001",
            query="What is the population of Atlantis?",
            chunks=[
                {
                    "id": "c1",
                    "text": "Atlantis is a legendary island first mentioned by "
                           "Plato. It is generally considered a myth or allegory "
                           "rather than a real place.",
                },
            ],
            expected_status=RewaStatus.INSUFFICIENT_EVIDENCE,
            description="Should not invent population for mythical place",
            tags=["factual", "no_invent"],
        ))

        cases.append(TestCase(
            id="no_invent_002",
            query="What is the boiling point of phlogiston?",
            chunks=[
                {
                    "id": "c1",
                    "text": "Phlogiston was a hypothetical substance believed to "
                           "be released during combustion. The phlogiston theory "
                           "was replaced by oxygen theory in the 18th century.",
                },
            ],
            expected_status=RewaStatus.INSUFFICIENT_EVIDENCE,
            description="Should not invent properties for disproven concept",
            tags=["factual", "no_invent"],
        ))

        # Case 3: Contradictory chunks - should flag, not pick randomly
        cases.append(TestCase(
            id="conflict_001",
            query="What year was the company founded?",
            chunks=[
                {
                    "id": "c1",
                    "text": "TechCorp was founded in 1998 by John Smith. The "
                           "company started in a garage in Silicon Valley.",
                },
                {
                    "id": "c2",
                    "text": "TechCorp, established in 2001 by Sarah Johnson, "
                           "began as a consulting firm before pivoting to software.",
                },
            ],
            expected_status=RewaStatus.CONFLICT,
            description="Should flag contradictory founding dates",
            tags=["factual", "conflict"],
        ))

        # Case 4: Partial information - should not fabricate missing parts
        cases.append(TestCase(
            id="partial_001",
            query="What is the complete address of the office?",
            chunks=[
                {
                    "id": "c1",
                    "text": "Our office is located on Main Street in downtown. "
                           "The building has 20 floors.",
                },
            ],
            expected_status=RewaStatus.INSUFFICIENT_EVIDENCE,
            description="Should not invent street number or city",
            tags=["factual", "partial"],
        ))

        # Case 5: Hedged/uncertain information should be flagged
        cases.append(TestCase(
            id="uncertain_001",
            query="What causes the disease?",
            chunks=[
                {
                    "id": "c1",
                    "text": "The exact cause of XYZ disease is unknown. "
                           "Researchers believe it may be genetic, but this "
                           "has not been confirmed. Some studies suggest "
                           "environmental factors might play a role.",
                },
            ],
            expected_status=RewaStatus.INSUFFICIENT_EVIDENCE,
            description="Uncertain/hedged facts should be treated cautiously",
            tags=["factual", "uncertain"],
        ))

        # Case 6: Verified facts with high confidence
        cases.append(TestCase(
            id="verified_001",
            query="What is the speed of light?",
            chunks=[
                {
                    "id": "c1",
                    "text": "The speed of light in vacuum is exactly "
                           "299,792,458 meters per second. This is a "
                           "fundamental physical constant denoted c.",
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Well-established physical constants should be valid",
            tags=["factual", "verified"],
        ))

        # Case 7: Historical facts
        cases.append(TestCase(
            id="history_001",
            query="When did World War II end?",
            chunks=[
                {
                    "id": "c1",
                    "text": "World War II ended in 1945. Germany surrendered "
                           "on May 8, 1945 (V-E Day), and Japan surrendered on "
                           "September 2, 1945 (V-J Day).",
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Historical facts should be preserved",
            tags=["factual", "history"],
        ))

        # Case 8: Source attribution - don't strip
        cases.append(TestCase(
            id="source_001",
            query="What does the study say about coffee?",
            chunks=[
                {
                    "id": "c1",
                    "text": "According to a 2023 Harvard study, moderate coffee "
                           "consumption (3-4 cups/day) is associated with lower "
                           "risk of cardiovascular disease. The study analyzed "
                           "data from 200,000 participants.",
                },
            ],
            expected_status=RewaStatus.VALID,
            description="Source attribution should be preserved",
            tags=["factual", "source"],
        ))

        return cases

    def _evaluate_response(
        self,
        case: TestCase,
        response
    ) -> bool:
        """Custom evaluation for factual regression."""
        # Basic status check
        if response.status != case.expected_status:
            return False

        # For VALID responses, ensure facts exist
        if case.expected_status == RewaStatus.VALID:
            if not response.safe_facts and "safe_facts" in case.expected_properties:
                return False

        # For INSUFFICIENT_EVIDENCE, ensure no facts were invented
        if case.expected_status == RewaStatus.INSUFFICIENT_EVIDENCE:
            # Facts should be empty or very low confidence
            if response.safe_facts:
                for fact in response.safe_facts:
                    if fact.confidence > 0.8:
                        return False  # Shouldn't have high-confidence invented facts

        return True

    def _compute_metrics(self) -> ExperimentMetrics:
        """Compute metrics specific to factual regression."""
        base_metrics = super()._compute_metrics()

        # Additional metrics
        facts_preserved = 0
        facts_invented = 0
        facts_removed = 0

        for result in self.results:
            details = result.details
            if "preserve" in details.get("case_id", ""):
                if result.success:
                    facts_preserved += 1
                else:
                    facts_removed += 1
            if "no_invent" in details.get("case_id", ""):
                if not result.success:
                    facts_invented += 1

        base_metrics.details.update({
            "facts_preserved": facts_preserved,
            "facts_incorrectly_removed": facts_removed,
            "facts_incorrectly_invented": facts_invented,
            "regression_score": (facts_removed + facts_invented) / len(self.results) if self.results else 0,
        })

        return base_metrics


class FactInventionDetector:
    """
    Detects if facts were invented that weren't in the source.

    Compares REWA output facts against source chunks to ensure
    all facts are grounded.
    """

    def __init__(self):
        self.source_facts: Set[str] = set()

    def extract_source_facts(self, chunks: List[Dict[str, Any]]) -> Set[str]:
        """Extract key facts from source chunks."""
        facts = set()

        for chunk in chunks:
            text = chunk.get("text", "").lower()

            # Simple fact extraction (in real system, use NER + relation extraction)
            # Extract numbers
            import re
            numbers = re.findall(r'\b\d+(?:,\d+)*(?:\.\d+)?\b', text)
            facts.update(numbers)

            # Extract proper nouns (capitalized words not at start of sentence)
            words = text.split()
            for i, word in enumerate(words):
                if i > 0 and word[0].isupper():
                    facts.add(word.lower())

        return facts

    def check_for_inventions(
        self,
        response_facts: List[Fact],
        source_facts: Set[str]
    ) -> List[Fact]:
        """
        Check if any response facts were invented.

        Returns list of potentially invented facts.
        """
        invented = []

        for fact in response_facts:
            # Check if fact value appears in source
            value_str = str(fact.value).lower()

            if value_str not in source_facts:
                # Check if any part of the value is in source
                value_parts = value_str.split()
                found = any(part in source_facts for part in value_parts)

                if not found:
                    invented.append(fact)

        return invented


def compute_hallucination_delta(
    rewa_results: ExperimentMetrics,
    baseline_results: ExperimentMetrics
) -> Dict[str, float]:
    """
    Compute the change in hallucination rate between REWA and baseline.

    Negative delta = REWA reduced hallucinations (good)
    Positive delta = REWA increased hallucinations (bad)
    """
    rewa_regression = rewa_results.details.get("regression_score", 0)
    baseline_regression = baseline_results.details.get("regression_score", 0)

    return {
        "rewa_regression_score": rewa_regression,
        "baseline_regression_score": baseline_regression,
        "delta": rewa_regression - baseline_regression,
        "improvement": baseline_regression - rewa_regression,
        "improved": rewa_regression < baseline_regression,
    }
