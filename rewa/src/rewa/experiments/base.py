"""
Base classes for experiments.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from rewa.api import REWA
from rewa.models import RewaResponse, RewaStatus


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""
    experiment_name: str
    success: bool
    expected: Any
    actual: Any
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ExperimentMetrics:
    """Aggregated metrics for an experiment."""
    experiment_name: str
    total_cases: int
    passed: int
    failed: int
    accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def success_rate(self) -> float:
        return self.passed / self.total_cases if self.total_cases > 0 else 0.0


@dataclass
class TestCase:
    """A single test case for an experiment."""
    id: str
    query: str
    chunks: List[Dict[str, Any]]
    expected_status: RewaStatus
    expected_properties: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    tags: List[str] = field(default_factory=list)


class BaseExperiment(ABC):
    """Base class for all REWA experiments."""

    def __init__(self, rewa: Optional[REWA] = None):
        """
        Initialize experiment.

        Args:
            rewa: Optional REWA instance (created if not provided)
        """
        self.rewa = rewa or REWA()
        self.results: List[ExperimentResult] = []

    @property
    @abstractmethod
    def name(self) -> str:
        """Experiment name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Experiment description."""
        pass

    @abstractmethod
    def get_test_cases(self) -> List[TestCase]:
        """Generate test cases for this experiment."""
        pass

    def run_case(self, case: TestCase) -> ExperimentResult:
        """
        Run a single test case.

        Args:
            case: Test case to run

        Returns:
            ExperimentResult
        """
        try:
            response = self.rewa.verify(case.query, case.chunks)

            success = self._evaluate_response(case, response)

            return ExperimentResult(
                experiment_name=self.name,
                success=success,
                expected=case.expected_status.value,
                actual=response.status.value,
                details={
                    "case_id": case.id,
                    "query": case.query,
                    "response_confidence": response.confidence,
                    "explanation": response.explanation,
                    "safe_facts_count": len(response.safe_facts),
                    "contradictions_count": len(response.contradictions),
                    "impossibilities_count": len(response.impossibilities),
                },
            )
        except Exception as e:
            return ExperimentResult(
                experiment_name=self.name,
                success=False,
                expected=case.expected_status.value,
                actual="ERROR",
                error=str(e),
                details={"case_id": case.id},
            )

    def _evaluate_response(
        self,
        case: TestCase,
        response: RewaResponse
    ) -> bool:
        """Evaluate if response matches expected outcome."""
        # Primary check: status match
        if response.status != case.expected_status:
            return False

        # Additional property checks
        for prop, expected in case.expected_properties.items():
            actual = getattr(response, prop, None)
            if actual != expected:
                return False

        return True

    def run(self) -> ExperimentMetrics:
        """
        Run all test cases and compute metrics.

        Returns:
            ExperimentMetrics with aggregated results
        """
        self.results = []
        test_cases = self.get_test_cases()

        for case in test_cases:
            result = self.run_case(case)
            self.results.append(result)

        return self._compute_metrics()

    def _compute_metrics(self) -> ExperimentMetrics:
        """Compute aggregated metrics from results."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.success)
        failed = total - passed

        # Compute false positive/negative rates
        # These are domain-specific - subclasses may override
        fp_count = 0
        fn_count = 0
        for result in self.results:
            if not result.success:
                if result.actual in ["VALID", "INSUFFICIENT_EVIDENCE"]:
                    fn_count += 1  # Should have been flagged
                else:
                    fp_count += 1  # Incorrectly flagged

        fp_rate = fp_count / total if total > 0 else 0.0
        fn_rate = fn_count / total if total > 0 else 0.0

        return ExperimentMetrics(
            experiment_name=self.name,
            total_cases=total,
            passed=passed,
            failed=failed,
            accuracy=passed / total if total > 0 else 0.0,
            false_positive_rate=fp_rate,
            false_negative_rate=fn_rate,
            details={
                "failed_cases": [
                    r.details.get("case_id")
                    for r in self.results
                    if not r.success
                ],
            },
        )
