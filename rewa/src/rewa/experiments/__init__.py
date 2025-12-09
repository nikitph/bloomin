"""
REWA Experiments Module

Implements the validation and falsification experiments from the PRD:
1. Impossible Query Detection
2. Negation Sensitivity
3. Temporal Override (Medical)
4. Multi-Chart Ambiguity Stress Test
5. Factual Regression Safety
"""

from rewa.experiments.impossible_query import ImpossibleQueryExperiment
from rewa.experiments.negation_sensitivity import NegationSensitivityExperiment
from rewa.experiments.temporal_override import TemporalOverrideExperiment
from rewa.experiments.ambiguity_stress import AmbiguityStressExperiment
from rewa.experiments.factual_regression import FactualRegressionExperiment
from rewa.experiments.runner import ExperimentRunner, run_all_experiments

__all__ = [
    "ImpossibleQueryExperiment",
    "NegationSensitivityExperiment",
    "TemporalOverrideExperiment",
    "AmbiguityStressExperiment",
    "FactualRegressionExperiment",
    "ExperimentRunner",
    "run_all_experiments",
]
