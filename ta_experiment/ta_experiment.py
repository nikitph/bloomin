"""
Thresholded Accumulation Experiment
Demonstrates O(1) decision complexity vs O(n) search complexity
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time, perf_counter
from dataclasses import dataclass
from typing import Tuple, List
import json

# Set random seed for reproducibility
np.random.seed(42)

@dataclass
class ExperimentConfig:
    """Configuration for TA experiment"""
    embedding_dim: int = 384
    cache_sizes: List[int] = None
    n_queries: int = 1000
    similarity_threshold: float = 0.85
    n_trials: int = 5  # Multiple runs for statistical significance

    def __post_init__(self):
        if self.cache_sizes is None:
            self.cache_sizes = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000]

@dataclass
class ExperimentResults:
    """Store experiment results"""
    cache_size: int
    search_time_mean: float
    search_time_std: float
    ta_time_mean: float
    ta_time_std: float
    speedup: float
    accuracy: float
    false_positive_rate: float
    false_negative_rate: float


class ThresholdedAccumulation:
    """
    Implementation of Thresholded Accumulation primitive

    Based on Definition 2.1 from paper:
    A = (X, R, φ, ⊕, τ)
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.accumulated = None
        self.n_elements = 0

    def accumulate(self, embeddings: np.ndarray) -> None:
        """
        Compute accumulated representation: A_D = ⊕_{x∈D} φ(x)

        Args:
            embeddings: (n, d) array of embedded elements
        """
        # In our case: R = R^d, ⊕ = +
        self.accumulated = np.sum(embeddings, axis=0)
        self.n_elements = len(embeddings)

    def decide(self, query: np.ndarray, threshold: float) -> bool:
        """
        Thresholded decision: τ(A_D ⊕ φ(q))

        Args:
            query: (d,) query embedding
            threshold: similarity threshold

        Returns:
            Boolean decision (O(1) complexity)
        """
        if self.accumulated is None:
            return False

        # Compute accumulated similarity score
        score = np.dot(self.accumulated, query)

        # Calibrated threshold (accounts for accumulation)
        # Theoretical optimal: threshold * sqrt(n) for normalized embeddings
        calibrated_threshold = threshold * np.sqrt(self.n_elements)

        return score >= calibrated_threshold

    def decide_batch(self, queries: np.ndarray, threshold: float) -> np.ndarray:
        """Vectorized batch decision for efficiency"""
        scores = queries @ self.accumulated
        calibrated_threshold = threshold * np.sqrt(self.n_elements)
        return scores >= calibrated_threshold


class SearchBaseline:
    """Traditional search-based decision (baseline for comparison)"""

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings
        self.n_elements = len(embeddings)

    def decide(self, query: np.ndarray, threshold: float) -> bool:
        """
        Search-based decision: check if any element exceeds threshold
        O(n) complexity
        """
        similarities = self.embeddings @ query
        return np.any(similarities >= threshold)

    def decide_batch(self, queries: np.ndarray, threshold: float) -> np.ndarray:
        """Vectorized batch decision"""
        # Shape: (n_cache, n_queries)
        similarities = self.embeddings @ queries.T
        return np.any(similarities >= threshold, axis=0)

    def get_ground_truth(self, queries: np.ndarray, threshold: float) -> np.ndarray:
        """Get exact ground truth decisions"""
        return self.decide_batch(queries, threshold)


def generate_synthetic_embeddings(n: int, dim: int) -> np.ndarray:
    """
    Generate normalized random embeddings

    Simulates real-world embeddings (e.g., sentence-transformers output)
    """
    embeddings = np.random.randn(n, dim).astype(np.float32)
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def run_single_trial(
    cache_size: int,
    config: ExperimentConfig
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Run single experimental trial

    Returns:
        (search_time, ta_time, search_decisions, ta_decisions)
    """
    # Generate cache embeddings
    cache_embeddings = generate_synthetic_embeddings(cache_size, config.embedding_dim)

    # Generate query embeddings
    query_embeddings = generate_synthetic_embeddings(
        config.n_queries,
        config.embedding_dim
    )

    # Initialize methods
    search = SearchBaseline(cache_embeddings)
    ta = ThresholdedAccumulation(config.embedding_dim)

    # Pre-accumulate (one-time cost for TA)
    accumulation_start = perf_counter()
    ta.accumulate(cache_embeddings)
    accumulation_time = perf_counter() - accumulation_start

    # Benchmark search-based decision
    search_start = perf_counter()
    search_decisions = search.decide_batch(query_embeddings, config.similarity_threshold)
    search_time = perf_counter() - search_start

    # Benchmark TA decision
    ta_start = perf_counter()
    ta_decisions = ta.decide_batch(query_embeddings, config.similarity_threshold)
    ta_time = perf_counter() - ta_start

    return search_time, ta_time, search_decisions, ta_decisions, accumulation_time


def compute_accuracy_metrics(
    ground_truth: np.ndarray,
    predictions: np.ndarray
) -> Tuple[float, float, float]:
    """
    Compute accuracy, FPR, FNR

    Returns:
        (accuracy, false_positive_rate, false_negative_rate)
    """
    accuracy = np.mean(ground_truth == predictions)

    # False positives: predicted True, actually False
    false_positives = np.sum((predictions == True) & (ground_truth == False))
    # False negatives: predicted False, actually True
    false_negatives = np.sum((predictions == False) & (ground_truth == True))

    n_negatives = np.sum(ground_truth == False)
    n_positives = np.sum(ground_truth == True)

    fpr = false_positives / n_negatives if n_negatives > 0 else 0.0
    fnr = false_negatives / n_positives if n_positives > 0 else 0.0

    return accuracy, fpr, fnr


def run_scaling_experiment(config: ExperimentConfig) -> List[ExperimentResults]:
    """
    Main experiment: test TA vs Search across cache sizes
    """
    results = []

    print("="*80)
    print("THRESHOLDED ACCUMULATION SCALING EXPERIMENT")
    print("="*80)
    print(f"Configuration:")
    print(f"  Embedding dimension: {config.embedding_dim}")
    print(f"  Number of queries: {config.n_queries}")
    print(f"  Similarity threshold: {config.similarity_threshold}")
    print(f"  Trials per cache size: {config.n_trials}")
    print(f"  Cache sizes: {config.cache_sizes}")
    print("="*80)
    print()

    for cache_size in config.cache_sizes:
        print(f"Testing cache size: {cache_size:,}")

        search_times = []
        ta_times = []
        accuracies = []
        fprs = []
        fnrs = []
        accumulation_times = []

        # Run multiple trials for statistical significance
        for trial in range(config.n_trials):
            search_time, ta_time, search_decisions, ta_decisions, acc_time = \
                run_single_trial(cache_size, config)

            search_times.append(search_time)
            ta_times.append(ta_time)
            accumulation_times.append(acc_time)

            # Compute accuracy (search is ground truth)
            accuracy, fpr, fnr = compute_accuracy_metrics(search_decisions, ta_decisions)
            accuracies.append(accuracy)
            fprs.append(fpr)
            fnrs.append(fnr)

        # Aggregate statistics
        result = ExperimentResults(
            cache_size=cache_size,
            search_time_mean=np.mean(search_times),
            search_time_std=np.std(search_times),
            ta_time_mean=np.mean(ta_times),
            ta_time_std=np.std(ta_times),
            speedup=np.mean(search_times) / np.mean(ta_times),
            accuracy=np.mean(accuracies),
            false_positive_rate=np.mean(fprs),
            false_negative_rate=np.mean(fnrs)
        )

        results.append(result)

        # Print summary
        print(f"  Search: {result.search_time_mean*1000:.2f}ms ±{result.search_time_std*1000:.2f}ms")
        print(f"  TA:     {result.ta_time_mean*1000:.2f}ms ±{result.ta_time_std*1000:.2f}ms")
        print(f"  Speedup: {result.speedup:.1f}x")
        print(f"  Accuracy: {result.accuracy*100:.1f}%")
        print(f"  FPR: {result.false_positive_rate*100:.2f}%, FNR: {result.false_negative_rate*100:.2f}%")
        print(f"  Accumulation overhead: {np.mean(accumulation_times)*1000:.2f}ms (one-time)")
        print()

    return results


def save_results(results: List[ExperimentResults], filename: str):
    """Save results to JSON"""
    data = [
        {
            'cache_size': r.cache_size,
            'search_time_mean': r.search_time_mean,
            'search_time_std': r.search_time_std,
            'ta_time_mean': r.ta_time_mean,
            'ta_time_std': r.ta_time_std,
            'speedup': r.speedup,
            'accuracy': r.accuracy,
            'false_positive_rate': r.false_positive_rate,
            'false_negative_rate': r.false_negative_rate
        }
        for r in results
    ]

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Results saved to {filename}")


if __name__ == "__main__":
    # Run experiment
    config = ExperimentConfig()
    results = run_scaling_experiment(config)

    # Save results
    save_results(results, "results/experiment_results.json")

    print("="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
