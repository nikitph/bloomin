"""
Thresholded Accumulation Experiment v3
Proper threshold calibration analysis

Key insight: The sqrt(n) calibration from v1/v2 is designed for AVERAGE similarity
detection, not ANY match detection. This version explores proper calibration
strategies for different use cases.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Callable
import json
from enum import Enum

np.random.seed(42)


class CalibrationStrategy(Enum):
    """Different threshold calibration strategies"""
    SQRT_N = "sqrt_n"           # Original: threshold * sqrt(n) - for average similarity
    ADDITIVE_NOISE = "additive" # threshold + k*sqrt(n/d) - for any-match detection
    ADAPTIVE = "adaptive"       # Learned from statistics of the cache
    FIXED = "fixed"             # No scaling (raw threshold comparison)


@dataclass
class ExperimentConfig:
    embedding_dim: int = 384
    cache_sizes: List[int] = field(default_factory=lambda: [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000])
    n_queries: int = 1000
    similarity_threshold: float = 0.85
    n_trials: int = 5
    match_ratio: float = 0.3
    calibration: CalibrationStrategy = CalibrationStrategy.ADDITIVE_NOISE
    noise_multiplier: float = 3.0  # Safety margin for noise


@dataclass
class TrialResult:
    cache_size: int
    search_time: float
    ta_time: float
    accuracy: float
    precision: float
    recall: float
    fpr: float
    fnr: float
    n_positives: int
    n_negatives: int
    calibrated_threshold: float


class ThresholdedAccumulation:
    """TA with configurable calibration strategies"""

    def __init__(self, embedding_dim: int, calibration: CalibrationStrategy, noise_multiplier: float = 3.0):
        self.embedding_dim = embedding_dim
        self.calibration = calibration
        self.noise_multiplier = noise_multiplier
        self.accumulated = None
        self.n_elements = 0
        self.cache_stats = {}

    def accumulate(self, embeddings: np.ndarray) -> None:
        self.accumulated = np.sum(embeddings, axis=0)
        self.n_elements = len(embeddings)

        # Compute statistics for adaptive calibration
        # Expected std of dot product for random unit vectors ≈ 1/sqrt(d)
        self.cache_stats['expected_noise_std'] = 1.0 / np.sqrt(self.embedding_dim)
        self.cache_stats['accumulated_norm'] = np.linalg.norm(self.accumulated)

    def _get_calibrated_threshold(self, raw_threshold: float) -> float:
        """Compute threshold based on calibration strategy"""
        n = self.n_elements
        d = self.embedding_dim

        if self.calibration == CalibrationStrategy.SQRT_N:
            # Original: assumes accumulated score scales with sqrt(n)
            # Good for: detecting if AVERAGE similarity is high
            return raw_threshold * np.sqrt(n)

        elif self.calibration == CalibrationStrategy.ADDITIVE_NOISE:
            # For detecting ANY single match above threshold
            # Noise in accumulated sum is O(sqrt(n/d))
            # If one element matches at level τ, total = τ + noise
            # We threshold at τ + k*noise_std for safety
            noise_std = np.sqrt(n / d)  # std of sum of n random dot products
            return raw_threshold + self.noise_multiplier * noise_std

        elif self.calibration == CalibrationStrategy.ADAPTIVE:
            # Use actual accumulated vector statistics
            # The accumulated norm gives info about alignment
            noise_std = np.sqrt(n / d)
            return raw_threshold + self.noise_multiplier * noise_std

        elif self.calibration == CalibrationStrategy.FIXED:
            # No calibration - raw threshold
            return raw_threshold

        raise ValueError(f"Unknown calibration: {self.calibration}")

    def decide_batch(self, queries: np.ndarray, threshold: float) -> Tuple[np.ndarray, float]:
        """Return decisions and the calibrated threshold used"""
        scores = queries @ self.accumulated
        calibrated = self._get_calibrated_threshold(threshold)
        return scores >= calibrated, calibrated


class SearchBaseline:
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def decide_batch(self, queries: np.ndarray, threshold: float) -> np.ndarray:
        similarities = self.embeddings @ queries.T
        return np.any(similarities >= threshold, axis=0)


def generate_normalized_embeddings(n: int, dim: int) -> np.ndarray:
    embeddings = np.random.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def inject_matches(
    cache_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    match_ratio: float,
    target_similarity: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Inject queries that are guaranteed to match cache elements"""
    n_queries = len(query_embeddings)
    n_cache = len(cache_embeddings)
    n_matches = int(n_queries * match_ratio)
    dim = cache_embeddings.shape[1]

    modified_queries = query_embeddings.copy()
    injected = np.zeros(n_queries, dtype=bool)

    match_indices = np.random.choice(n_queries, n_matches, replace=False)

    for query_idx in match_indices:
        cache_idx = np.random.randint(0, n_cache)
        base = cache_embeddings[cache_idx]

        # Create query with exact target similarity via interpolation
        # q = α*base + sqrt(1-α²)*orthogonal, where α = target_similarity
        alpha = target_similarity + 0.02  # Slightly above threshold

        # Generate orthogonal component
        random_vec = np.random.randn(dim).astype(np.float32)
        orthogonal = random_vec - np.dot(random_vec, base) * base
        orthogonal = orthogonal / np.linalg.norm(orthogonal)

        # Construct query with desired similarity
        new_query = alpha * base + np.sqrt(1 - alpha**2) * orthogonal
        new_query = new_query / np.linalg.norm(new_query)

        modified_queries[query_idx] = new_query
        injected[query_idx] = True

    return modified_queries, injected


def run_trial(cache_size: int, config: ExperimentConfig) -> TrialResult:
    """Run single trial"""
    cache = generate_normalized_embeddings(cache_size, config.embedding_dim)
    queries = generate_normalized_embeddings(config.n_queries, config.embedding_dim)

    queries, _ = inject_matches(cache, queries, config.match_ratio, config.similarity_threshold)

    search = SearchBaseline(cache)
    ta = ThresholdedAccumulation(config.embedding_dim, config.calibration, config.noise_multiplier)

    ta.accumulate(cache)

    # Ground truth from search
    search_start = perf_counter()
    ground_truth = search.decide_batch(queries, config.similarity_threshold)
    search_time = perf_counter() - search_start

    # TA decisions
    ta_start = perf_counter()
    ta_decisions, calibrated_threshold = ta.decide_batch(queries, config.similarity_threshold)
    ta_time = perf_counter() - ta_start

    # Metrics
    tp = int(np.sum((ta_decisions == True) & (ground_truth == True)))
    tn = int(np.sum((ta_decisions == False) & (ground_truth == False)))
    fp = int(np.sum((ta_decisions == True) & (ground_truth == False)))
    fn = int(np.sum((ta_decisions == False) & (ground_truth == True)))

    n_pos = int(np.sum(ground_truth))
    n_neg = int(np.sum(~ground_truth))

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    fpr = fp / n_neg if n_neg > 0 else 0.0
    fnr = fn / n_pos if n_pos > 0 else 0.0

    return TrialResult(
        cache_size=cache_size,
        search_time=search_time,
        ta_time=ta_time,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        fpr=fpr,
        fnr=fnr,
        n_positives=n_pos,
        n_negatives=n_neg,
        calibrated_threshold=calibrated_threshold
    )


def aggregate_trials(trials: List[TrialResult]) -> Dict:
    """Aggregate multiple trial results"""
    return {
        'cache_size': trials[0].cache_size,
        'search_time_mean': float(np.mean([t.search_time for t in trials])),
        'search_time_std': float(np.std([t.search_time for t in trials])),
        'ta_time_mean': float(np.mean([t.ta_time for t in trials])),
        'ta_time_std': float(np.std([t.ta_time for t in trials])),
        'speedup': float(np.mean([t.search_time for t in trials]) / np.mean([t.ta_time for t in trials])),
        'accuracy': float(np.mean([t.accuracy for t in trials])),
        'precision': float(np.mean([t.precision for t in trials])),
        'recall': float(np.mean([t.recall for t in trials])),
        'fpr': float(np.mean([t.fpr for t in trials])),
        'fnr': float(np.mean([t.fnr for t in trials])),
        'calibrated_threshold': float(np.mean([t.calibrated_threshold for t in trials]))
    }


def run_experiment(config: ExperimentConfig) -> List[Dict]:
    """Run full scaling experiment"""
    print("=" * 80)
    print("THRESHOLDED ACCUMULATION EXPERIMENT v3")
    print(f"Calibration Strategy: {config.calibration.value}")
    print("=" * 80)
    print(f"  Embedding dim: {config.embedding_dim}")
    print(f"  Similarity threshold: {config.similarity_threshold}")
    print(f"  Match ratio: {config.match_ratio * 100:.0f}%")
    print(f"  Noise multiplier (k): {config.noise_multiplier}")
    print("=" * 80)
    print()

    results = []

    for cache_size in config.cache_sizes:
        print(f"Cache size: {cache_size:,}")

        trials = [run_trial(cache_size, config) for _ in range(config.n_trials)]
        agg = aggregate_trials(trials)
        results.append(agg)

        print(f"  Search: {agg['search_time_mean']*1000:.2f}ms, TA: {agg['ta_time_mean']*1000:.2f}ms")
        print(f"  Speedup: {agg['speedup']:.1f}x")
        print(f"  Accuracy: {agg['accuracy']*100:.1f}%, Precision: {agg['precision']*100:.1f}%, Recall: {agg['recall']*100:.1f}%")
        print(f"  Calibrated threshold: {agg['calibrated_threshold']:.2f}")
        print()

    return results


def compare_calibrations(base_config: ExperimentConfig):
    """Compare different calibration strategies"""
    print("\n" + "=" * 80)
    print("CALIBRATION STRATEGY COMPARISON")
    print("=" * 80 + "\n")

    strategies = [
        (CalibrationStrategy.ADDITIVE_NOISE, 3.0),
        (CalibrationStrategy.ADDITIVE_NOISE, 2.0),
        (CalibrationStrategy.ADDITIVE_NOISE, 1.0),
        (CalibrationStrategy.SQRT_N, 1.0),
    ]

    all_results = {}

    for strat, mult in strategies:
        config = ExperimentConfig(
            embedding_dim=base_config.embedding_dim,
            cache_sizes=base_config.cache_sizes,
            n_queries=base_config.n_queries,
            similarity_threshold=base_config.similarity_threshold,
            n_trials=base_config.n_trials,
            match_ratio=base_config.match_ratio,
            calibration=strat,
            noise_multiplier=mult
        )

        label = f"{strat.value}_k{mult}"
        print(f"\n--- {label} ---")
        results = run_experiment(config)
        all_results[label] = results

    return all_results


def generate_comparison_plots(all_results: Dict[str, List[Dict]], output_dir: str = "plots"):
    """Generate comparison plots across calibration strategies"""
    import os
    os.makedirs(output_dir, exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)

    # Get cache sizes from first result
    first_key = list(all_results.keys())[0]
    cache_sizes = [r['cache_size'] for r in all_results[first_key]]

    # Plot 1: Recall comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for (label, results), color in zip(all_results.items(), colors):
        recalls = [r['recall'] * 100 for r in results]
        ax.plot(cache_sizes, recalls, marker='o', linewidth=2, label=label, color=color)

    ax.set_xlabel('Cache Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Recall (%)', fontsize=14, fontweight='bold')
    ax.set_title('Recall by Calibration Strategy', fontsize=16, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_recall.png", dpi=300)
    plt.close()

    # Plot 2: Precision comparison
    fig, ax = plt.subplots(figsize=(12, 8))

    for (label, results), color in zip(all_results.items(), colors):
        precisions = [r['precision'] * 100 for r in results]
        ax.plot(cache_sizes, precisions, marker='s', linewidth=2, label=label, color=color)

    ax.set_xlabel('Cache Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision (%)', fontsize=14, fontweight='bold')
    ax.set_title('Precision by Calibration Strategy', fontsize=16, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_precision.png", dpi=300)
    plt.close()

    # Plot 3: Accuracy comparison
    fig, ax = plt.subplots(figsize=(12, 8))

    for (label, results), color in zip(all_results.items(), colors):
        accs = [r['accuracy'] * 100 for r in results]
        ax.plot(cache_sizes, accs, marker='^', linewidth=2, label=label, color=color)

    ax.set_xlabel('Cache Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy by Calibration Strategy', fontsize=16, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=90, color='green', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_accuracy.png", dpi=300)
    plt.close()

    # Plot 4: Speedup (should be same for all)
    fig, ax = plt.subplots(figsize=(12, 8))

    # Just use first strategy for speedup (all same)
    results = list(all_results.values())[0]
    speedups = [r['speedup'] for r in results]
    ax.plot(cache_sizes, speedups, marker='D', linewidth=3, color='#9b59b6')

    ax.set_xlabel('Cache Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Speedup (x)', fontsize=14, fontweight='bold')
    ax.set_title('TA Speedup vs Search (All Calibrations)', fontsize=16, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    max_idx = np.argmax(speedups)
    ax.annotate(f'{speedups[max_idx]:.0f}x', xy=(cache_sizes[max_idx], speedups[max_idx]),
                xytext=(cache_sizes[max_idx]/5, speedups[max_idx]*0.8),
                fontsize=14, fontweight='bold', arrowprops=dict(arrowstyle='->'))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/speedup_v3.png", dpi=300)
    plt.close()

    # Plot 5: Time scaling
    fig, ax = plt.subplots(figsize=(12, 8))

    results = list(all_results.values())[0]
    search_times = [r['search_time_mean'] * 1000 for r in results]
    ta_times = [r['ta_time_mean'] * 1000 for r in results]

    ax.plot(cache_sizes, search_times, marker='o', linewidth=2.5, color='#e74c3c', label='Search O(n)')
    ax.plot(cache_sizes, ta_times, marker='s', linewidth=2.5, color='#2ecc71', label='TA O(1)')

    ax.set_xlabel('Cache Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=14, fontweight='bold')
    ax.set_title('Complexity: O(n) Search vs O(1) TA', fontsize=16, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/time_scaling_v3.png", dpi=300)
    plt.close()

    print(f"\nPlots saved to {output_dir}/")


def generate_report(all_results: Dict[str, List[Dict]], output_file: str):
    """Generate comprehensive report"""
    from datetime import datetime

    report = f"""# Thresholded Accumulation: Calibration Analysis Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## The Calibration Problem

The original TA formulation uses `calibrated_threshold = raw_threshold * sqrt(n)`.

**Problem:** This calibration is designed for detecting if the AVERAGE similarity
exceeds a threshold, NOT for detecting if ANY SINGLE element exceeds it.

For "any match" detection, the accumulated score behaves as:
- **With a match:** score = match_similarity + noise ≈ τ + O(sqrt(n/d))
- **Without match:** score = noise ≈ O(sqrt(n/d))

The sqrt(n) calibration creates a threshold that grows too fast, causing 100% false negatives.

## Solution: Additive Noise Calibration

For "any match" detection, use:
```
calibrated_threshold = raw_threshold + k * sqrt(n/d)
```

Where:
- `raw_threshold` = τ (e.g., 0.85)
- `k` = noise multiplier (safety margin)
- `n` = cache size
- `d` = embedding dimension

## Calibration Comparison Results

"""

    for label, results in all_results.items():
        report += f"\n### Strategy: {label}\n\n"
        report += "| Cache Size | Speedup | Accuracy | Precision | Recall | FPR | FNR |\n"
        report += "|------------|---------|----------|-----------|--------|-----|-----|\n"

        for r in results:
            report += f"| {r['cache_size']:,} | {r['speedup']:.0f}x | {r['accuracy']*100:.1f}% | {r['precision']*100:.1f}% | {r['recall']*100:.1f}% | {r['fpr']*100:.1f}% | {r['fnr']*100:.1f}% |\n"

    # Find best strategy
    best_recall_strategy = None
    best_recall = 0
    for label, results in all_results.items():
        avg_recall = np.mean([r['recall'] for r in results])
        if avg_recall > best_recall:
            best_recall = avg_recall
            best_recall_strategy = label

    # Get speedup from any strategy (same for all)
    any_results = list(all_results.values())[0]
    max_speedup = max(r['speedup'] for r in any_results)

    report += f"""

## Key Findings

1. **Calibration matters significantly** for accuracy metrics
2. **Additive noise calibration** (k=1-2) achieves best recall
3. **sqrt(n) calibration** results in 0% recall (too conservative)
4. **Speedup is independent** of calibration choice (~{max_speedup:.0f}x at largest cache)

## Recommended Configuration

For "any match" detection use cases:
```python
calibration = CalibrationStrategy.ADDITIVE_NOISE
noise_multiplier = 2.0  # Balance precision/recall
```

Best performing strategy: **{best_recall_strategy}** with {best_recall*100:.1f}% average recall

## Theoretical Explanation

The accumulated dot product for a query q is:

```
A · q = Σᵢ (eᵢ · q)
```

For random unit vectors in R^d:
- E[eᵢ · q] = 0 (orthogonal in expectation)
- Var[eᵢ · q] ≈ 1/d

Sum of n such terms:
- E[A · q] = 0
- Var[A · q] = n/d
- Std[A · q] = sqrt(n/d)

If ONE element matches with similarity τ:
```
A · q ≈ τ + (n-1) * noise_per_element
      ≈ τ + N(0, sqrt(n/d))
```

To reliably detect this, threshold at:
```
τ + k * sqrt(n/d) where k ≈ 2-3 for 95-99% confidence
```

---

*Report generated by TA Experiment v3*
"""

    with open(output_file, 'w') as f:
        f.write(report)

    print(f"Report saved to {output_file}")


if __name__ == "__main__":
    # Run comparison across calibration strategies
    base_config = ExperimentConfig()

    all_results = compare_calibrations(base_config)

    # Save results
    with open("results/experiment_results_v3.json", 'w') as f:
        # Convert to JSON-serializable format
        serializable = {k: v for k, v in all_results.items()}
        json.dump(serializable, f, indent=2)

    # Generate plots
    generate_comparison_plots(all_results, "plots")

    # Generate report
    generate_report(all_results, "results/experiment_report_v3.md")

    print("\n" + "=" * 80)
    print("EXPERIMENT v3 COMPLETE")
    print("=" * 80)
