"""
Experiment Runner

Orchestrates running all REWA experiments and collecting results.
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Type

from rewa.api import REWA, RewaConfig
from rewa.experiments.base import BaseExperiment, ExperimentMetrics
from rewa.experiments.impossible_query import ImpossibleQueryExperiment
from rewa.experiments.negation_sensitivity import NegationSensitivityExperiment
from rewa.experiments.temporal_override import TemporalOverrideExperiment
from rewa.experiments.ambiguity_stress import AmbiguityStressExperiment
from rewa.experiments.factual_regression import FactualRegressionExperiment


class ExperimentRunner:
    """
    Runs REWA validation experiments.

    Provides methods to run individual experiments or all experiments,
    and collects/reports results.
    """

    # Registry of all experiments
    EXPERIMENTS: Dict[str, Type[BaseExperiment]] = {
        "impossible_query": ImpossibleQueryExperiment,
        "negation_sensitivity": NegationSensitivityExperiment,
        "temporal_override": TemporalOverrideExperiment,
        "ambiguity_stress": AmbiguityStressExperiment,
        "factual_regression": FactualRegressionExperiment,
    }

    def __init__(
        self,
        rewa: Optional[REWA] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize the experiment runner.

        Args:
            rewa: Optional REWA instance (created with default config if not provided)
            output_dir: Optional directory to save results
        """
        self.rewa = rewa or REWA(RewaConfig())
        self.output_dir = Path(output_dir) if output_dir else None
        self.results: Dict[str, ExperimentMetrics] = {}

    def run_experiment(self, experiment_name: str) -> ExperimentMetrics:
        """
        Run a single experiment by name.

        Args:
            experiment_name: Name of experiment to run

        Returns:
            ExperimentMetrics with results

        Raises:
            ValueError: If experiment name is not recognized
        """
        if experiment_name not in self.EXPERIMENTS:
            raise ValueError(
                f"Unknown experiment: {experiment_name}. "
                f"Available: {list(self.EXPERIMENTS.keys())}"
            )

        experiment_class = self.EXPERIMENTS[experiment_name]
        experiment = experiment_class(self.rewa)

        print(f"Running experiment: {experiment.name}")
        print(f"Description: {experiment.description}")
        print("-" * 50)

        metrics = experiment.run()
        self.results[experiment_name] = metrics

        self._print_metrics(metrics)

        return metrics

    def run_all(self) -> Dict[str, ExperimentMetrics]:
        """
        Run all registered experiments.

        Returns:
            Dictionary mapping experiment names to their metrics
        """
        print("=" * 60)
        print("REWA Validation Experiments")
        print(f"Started: {datetime.now().isoformat()}")
        print("=" * 60)

        for name in self.EXPERIMENTS:
            print()
            self.run_experiment(name)
            print()

        # Print summary
        self._print_summary()

        # Save results if output directory specified
        if self.output_dir:
            self._save_results()

        return self.results

    def _print_metrics(self, metrics: ExperimentMetrics) -> None:
        """Print experiment metrics."""
        print(f"\nResults for: {metrics.experiment_name}")
        print(f"  Total cases: {metrics.total_cases}")
        print(f"  Passed: {metrics.passed}")
        print(f"  Failed: {metrics.failed}")
        print(f"  Accuracy: {metrics.accuracy:.2%}")
        print(f"  False Positive Rate: {metrics.false_positive_rate:.2%}")
        print(f"  False Negative Rate: {metrics.false_negative_rate:.2%}")

        if metrics.details.get("failed_cases"):
            print(f"  Failed cases: {metrics.details['failed_cases']}")

    def _print_summary(self) -> None:
        """Print summary of all experiments."""
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)

        total_cases = 0
        total_passed = 0

        for name, metrics in self.results.items():
            status = "PASS" if metrics.accuracy >= 0.9 else "FAIL"
            print(f"  {name}: {metrics.accuracy:.2%} ({status})")
            total_cases += metrics.total_cases
            total_passed += metrics.passed

        overall_accuracy = total_passed / total_cases if total_cases > 0 else 0
        print("-" * 60)
        print(f"  Overall: {overall_accuracy:.2%}")

        # Check success criteria from PRD
        print("\n" + "-" * 60)
        print("Success Criteria Check:")
        self._check_success_criteria()

    def _check_success_criteria(self) -> None:
        """Check results against PRD success criteria."""
        criteria = {
            "impossible_query": {
                "target": 0.90,
                "description": "Impossible query detection > 90%",
            },
            "negation_sensitivity": {
                "target": 0.99,
                "description": "Negation rejection > 99%",
            },
            "temporal_override": {
                "target": 0.95,
                "description": "Temporal correctness > 95%",
            },
            "ambiguity_stress": {
                "target": 0.90,
                "description": "Ambiguity detection > 90%",
            },
            "factual_regression": {
                "target": 0.95,
                "description": "No factual regression",
            },
        }

        for exp_name, criterion in criteria.items():
            if exp_name in self.results:
                metrics = self.results[exp_name]
                met = metrics.accuracy >= criterion["target"]
                status = "MET" if met else "NOT MET"
                print(f"  [{status}] {criterion['description']}: "
                      f"{metrics.accuracy:.2%} (target: {criterion['target']:.0%})")

    def _save_results(self) -> None:
        """Save results to output directory."""
        if not self.output_dir:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        results_dict = {}
        for name, metrics in self.results.items():
            results_dict[name] = {
                "experiment_name": metrics.experiment_name,
                "total_cases": metrics.total_cases,
                "passed": metrics.passed,
                "failed": metrics.failed,
                "accuracy": metrics.accuracy,
                "false_positive_rate": metrics.false_positive_rate,
                "false_negative_rate": metrics.false_negative_rate,
                "details": metrics.details,
                "timestamp": metrics.timestamp.isoformat(),
            }

        output_file = self.output_dir / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)

        print(f"\nResults saved to: {output_file}")


def run_all_experiments(
    rewa: Optional[REWA] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, ExperimentMetrics]:
    """
    Convenience function to run all experiments.

    Args:
        rewa: Optional REWA instance
        output_dir: Optional output directory

    Returns:
        Dictionary of experiment metrics
    """
    runner = ExperimentRunner(rewa, output_dir)
    return runner.run_all()


def run_single_experiment(
    experiment_name: str,
    rewa: Optional[REWA] = None,
) -> ExperimentMetrics:
    """
    Run a single experiment by name.

    Args:
        experiment_name: Name of experiment
        rewa: Optional REWA instance

    Returns:
        ExperimentMetrics
    """
    runner = ExperimentRunner(rewa)
    return runner.run_experiment(experiment_name)


if __name__ == "__main__":
    # Run all experiments when executed directly
    run_all_experiments(output_dir="experiment_results")
