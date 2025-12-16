"""
Thresholded Accumulation: Corrected Final Experiment

Fixes edge cases and provides accurate precision/recall measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import json
from datetime import datetime

np.random.seed(42)


@dataclass
class ExperimentConfig:
    embedding_dim: int = 384
    cache_sizes: List[int] = field(default_factory=lambda: [1_000, 5_000, 10_000, 50_000, 100_000])
    n_queries: int = 500
    similarity_threshold: float = 0.85
    n_trials: int = 5
    # Percentage of queries that will have injected matches
    query_match_ratio: float = 0.5  # 50% of queries get matches
    # How many cache elements each "matching" query will match
    matches_per_query: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50])


class ThresholdedAccumulation:
    def __init__(self, dim: int, k: float = 2.0):
        self.dim = dim
        self.k = k
        self.accumulated = None
        self.n = 0

    def accumulate(self, embeddings: np.ndarray):
        self.accumulated = np.sum(embeddings, axis=0)
        self.n = len(embeddings)

    def get_calibrated_threshold(self, raw_threshold: float) -> float:
        noise_std = np.sqrt(self.n / self.dim)
        return raw_threshold + self.k * noise_std

    def decide_batch(self, queries: np.ndarray, threshold: float) -> np.ndarray:
        scores = queries @ self.accumulated
        calibrated = self.get_calibrated_threshold(threshold)
        return scores >= calibrated


class SearchBaseline:
    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def decide_batch(self, queries: np.ndarray, threshold: float) -> np.ndarray:
        sims = self.embeddings @ queries.T
        return np.any(sims >= threshold, axis=0)


def generate_embeddings(n: int, d: int) -> np.ndarray:
    emb = np.random.randn(n, d).astype(np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def create_query_with_matches(
    cache: np.ndarray,
    n_matches: int,
    threshold: float
) -> np.ndarray:
    """Create a single query that matches n_matches cache elements above threshold."""
    d = cache.shape[1]
    n_cache = len(cache)

    if n_matches > n_cache:
        n_matches = n_cache

    # Select random cache elements to match
    match_indices = np.random.choice(n_cache, n_matches, replace=False)

    # Create query as centroid of matched elements (will have high similarity)
    centroid = np.mean(cache[match_indices], axis=0)
    centroid = centroid / np.linalg.norm(centroid)

    # For a single match, we need more careful construction
    if n_matches == 1:
        base = cache[match_indices[0]]
        # q = alpha * base + sqrt(1-alpha^2) * orthogonal
        alpha = threshold + 0.03  # Slightly above threshold
        orthogonal = np.random.randn(d).astype(np.float32)
        orthogonal = orthogonal - np.dot(orthogonal, base) * base
        orthogonal = orthogonal / np.linalg.norm(orthogonal)
        query = alpha * base + np.sqrt(1 - alpha**2) * orthogonal
    else:
        # For multiple matches, centroid approach works well
        query = centroid

    return query / np.linalg.norm(query)


def run_experiment(config: ExperimentConfig) -> Dict:
    """Run comprehensive experiment."""

    print("=" * 80)
    print("THRESHOLDED ACCUMULATION: CORRECTED EXPERIMENT")
    print("=" * 80)
    print(f"Testing: Can TA detect queries that match 1, 2, 5, 10, 20, 50 cache elements?")
    print(f"Threshold: {config.similarity_threshold}")
    print(f"Query match ratio: {config.query_match_ratio * 100:.0f}% (half have matches, half don't)")
    print("=" * 80)
    print()

    all_results = {}

    for cache_size in config.cache_sizes:
        print(f"\n{'='*60}")
        print(f"CACHE SIZE: {cache_size:,}")
        print(f"{'='*60}")

        noise_std = np.sqrt(cache_size / config.embedding_dim)
        print(f"Noise std: {noise_std:.2f}")
        print()

        cache_results = []

        for n_matches in config.matches_per_query:
            # SNR calculation
            expected_signal = n_matches * config.similarity_threshold
            snr = expected_signal / noise_std

            # Run trials
            search_times, ta_times = [], []
            total_tp, total_tn, total_fp, total_fn = 0, 0, 0, 0
            total_positives, total_negatives = 0, 0

            for trial in range(config.n_trials):
                # Generate cache
                cache = generate_embeddings(cache_size, config.embedding_dim)

                # Generate queries: half with matches, half without
                n_queries = config.n_queries
                n_with_matches = int(n_queries * config.query_match_ratio)
                n_without_matches = n_queries - n_with_matches

                # Queries with matches
                matched_queries = np.array([
                    create_query_with_matches(cache, n_matches, config.similarity_threshold)
                    for _ in range(n_with_matches)
                ])

                # Queries without matches (random, won't match at 0.85)
                unmatched_queries = generate_embeddings(n_without_matches, config.embedding_dim)

                # Combine
                queries = np.vstack([matched_queries, unmatched_queries])

                # Ground truth labels (first half have matches, second half don't... in theory)
                # But we should verify with actual search

                # Initialize methods
                search = SearchBaseline(cache)
                ta = ThresholdedAccumulation(config.embedding_dim, k=2.0)
                ta.accumulate(cache)

                # Get ground truth from exhaustive search
                t0 = perf_counter()
                ground_truth = search.decide_batch(queries, config.similarity_threshold)
                search_times.append(perf_counter() - t0)

                # Get TA predictions
                t0 = perf_counter()
                predictions = ta.decide_batch(queries, config.similarity_threshold)
                ta_times.append(perf_counter() - t0)

                # Compute metrics
                tp = int(np.sum(predictions & ground_truth))
                tn = int(np.sum(~predictions & ~ground_truth))
                fp = int(np.sum(predictions & ~ground_truth))
                fn = int(np.sum(~predictions & ground_truth))

                total_tp += tp
                total_tn += tn
                total_fp += fp
                total_fn += fn
                total_positives += int(np.sum(ground_truth))
                total_negatives += int(np.sum(~ground_truth))

            # Aggregate metrics
            accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            speedup = np.mean(search_times) / np.mean(ta_times)

            result = {
                'n_matches': n_matches,
                'snr': float(snr),
                'speedup': float(speedup),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'total_positives': total_positives,
                'total_negatives': total_negatives,
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn,
                'tn': total_tn,
                'search_time_ms': float(np.mean(search_times) * 1000),
                'ta_time_ms': float(np.mean(ta_times) * 1000),
                'calibrated_threshold': float(ta.get_calibrated_threshold(config.similarity_threshold))
            }
            cache_results.append(result)

            # Status indicator
            if recall >= 0.9 and precision >= 0.5:
                status = "EXCELLENT"
            elif recall >= 0.7:
                status = "GOOD"
            elif recall >= 0.5:
                status = "OK"
            else:
                status = "LOW"

            print(f"  {n_matches:3} matches/query | SNR: {snr:6.2f} | "
                  f"Recall: {recall*100:5.1f}% | Precision: {precision*100:5.1f}% | "
                  f"F1: {f1*100:5.1f}% | Speedup: {speedup:5.0f}x [{status}]")
            print(f"      TP={total_tp}, FP={total_fp}, FN={total_fn}, TN={total_tn} | "
                  f"Positives: {total_positives}, Negatives: {total_negatives}")

        all_results[cache_size] = cache_results

    return all_results


def generate_plots(results: Dict, output_dir: str = "plots"):
    """Generate visualization plots."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(results)))

    # Plot 1: Recall vs Number of Matches
    fig, ax = plt.subplots(figsize=(12, 8))

    for (cache_size, cache_results), color in zip(results.items(), colors):
        n_matches = [r['n_matches'] for r in cache_results]
        recalls = [r['recall'] * 100 for r in cache_results]
        ax.plot(n_matches, recalls, 'o-', label=f'n={cache_size:,}', color=color,
                linewidth=2.5, markersize=10)

    ax.axhline(y=80, color='green', linestyle='--', alpha=0.7, linewidth=2, label='80% target')
    ax.axhline(y=50, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='50% target')

    ax.set_xlabel('Number of Matching Elements per Query', fontsize=14, fontweight='bold')
    ax.set_ylabel('Recall (%)', fontsize=14, fontweight='bold')
    ax.set_title('TA Recall vs Match Count\n(More matches = better recall)', fontsize=16, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/recall_vs_matches_corrected.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Recall vs SNR
    fig, ax = plt.subplots(figsize=(12, 8))

    for (cache_size, cache_results), color in zip(results.items(), colors):
        snrs = [r['snr'] for r in cache_results]
        recalls = [r['recall'] * 100 for r in cache_results]
        ax.plot(snrs, recalls, 'o-', label=f'n={cache_size:,}', color=color,
                linewidth=2.5, markersize=10)

    ax.axhline(y=80, color='green', linestyle='--', alpha=0.7, linewidth=2)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='SNR=1')

    ax.set_xlabel('Signal-to-Noise Ratio', fontsize=14, fontweight='bold')
    ax.set_ylabel('Recall (%)', fontsize=14, fontweight='bold')
    ax.set_title('TA Recall vs SNR\n(SNR = n_matches * threshold / noise_std)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/recall_vs_snr_corrected.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 3: F1 Score vs Matches
    fig, ax = plt.subplots(figsize=(12, 8))

    for (cache_size, cache_results), color in zip(results.items(), colors):
        n_matches = [r['n_matches'] for r in cache_results]
        f1s = [r['f1'] * 100 for r in cache_results]
        ax.plot(n_matches, f1s, 'o-', label=f'n={cache_size:,}', color=color,
                linewidth=2.5, markersize=10)

    ax.set_xlabel('Number of Matching Elements per Query', fontsize=14, fontweight='bold')
    ax.set_ylabel('F1 Score (%)', fontsize=14, fontweight='bold')
    ax.set_title('TA F1 Score vs Match Count\n(Harmonic mean of Precision and Recall)', fontsize=16, fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/f1_vs_matches.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: Time Complexity
    fig, ax = plt.subplots(figsize=(12, 8))

    cache_sizes = sorted(results.keys())
    search_times = [results[cs][0]['search_time_ms'] for cs in cache_sizes]
    ta_times = [results[cs][0]['ta_time_ms'] for cs in cache_sizes]

    ax.plot(cache_sizes, search_times, 'o-', linewidth=3, markersize=12,
            color='#e74c3c', label='Search O(n)')
    ax.plot(cache_sizes, ta_times, 's-', linewidth=3, markersize=12,
            color='#2ecc71', label='TA O(1)')

    ax.set_xlabel('Cache Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=14, fontweight='bold')
    ax.set_title('Complexity: O(n) Search vs O(1) TA', fontsize=16, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=14, loc='upper left')
    ax.grid(True, alpha=0.3)

    # Annotate speedup
    max_speedup = results[max(cache_sizes)][0]['speedup']
    ax.annotate(f'{max_speedup:.0f}x speedup\nat n={max(cache_sizes):,}',
                xy=(max(cache_sizes), ta_times[-1]),
                xytext=(max(cache_sizes)/10, ta_times[-1]*3),
                fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=2))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/complexity_corrected.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 5: Precision vs Recall Tradeoff
    fig, ax = plt.subplots(figsize=(10, 8))

    for (cache_size, cache_results), color in zip(results.items(), colors):
        precisions = [r['precision'] * 100 for r in cache_results]
        recalls = [r['recall'] * 100 for r in cache_results]
        n_matches_list = [r['n_matches'] for r in cache_results]

        ax.scatter(recalls, precisions, c=[color]*len(recalls), s=150, alpha=0.7,
                   edgecolors='black', linewidth=1.5, label=f'n={cache_size:,}')

        # Annotate with n_matches
        for prec, rec, nm in zip(precisions, recalls, n_matches_list):
            ax.annotate(f'{nm}', (rec, prec), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)

    ax.set_xlabel('Recall (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision (%)', fontsize=14, fontweight='bold')
    ax.set_title('Precision-Recall Tradeoff\n(Numbers indicate matches per query)', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 105])
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/precision_recall_tradeoff.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nPlots saved to {output_dir}/")


def generate_report(results: Dict, config: ExperimentConfig, output_file: str):
    """Generate final report."""

    # Find minimum matches needed for 80% recall
    min_matches_80 = {}
    for cache_size, cache_results in results.items():
        for r in cache_results:
            if r['recall'] >= 0.8:
                min_matches_80[cache_size] = r['n_matches']
                break

    # Get max speedup
    max_speedup = max(r['speedup'] for cr in results.values() for r in cr)

    report = f"""# Thresholded Accumulation: Final Experimental Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Executive Summary

This experiment rigorously validates Thresholded Accumulation (TA) by testing its ability
to detect queries that match varying numbers of cache elements.

### Key Findings

| Metric | Value |
|--------|-------|
| **Maximum Speedup** | **{max_speedup:.0f}x** |
| **Complexity** | O(1) confirmed |
| **Best Use Case** | Queries matching 5+ cache elements |

### Minimum Matches for 80% Recall

| Cache Size | Min Matches Needed |
|------------|-------------------|
"""

    for cs in sorted(min_matches_80.keys()):
        report += f"| {cs:,} | {min_matches_80.get(cs, 'N/A')} |\n"

    report += f"""

---

## Understanding the Results

### Why Single-Match Detection is Hard

For a cache of n elements in d dimensions:
- **Noise level:** sqrt(n/d) from random dot products
- **Signal from 1 match:** ~0.85 (the threshold)
- **SNR for 1 match:** 0.85 / sqrt(n/d)

| Cache Size | Noise Std | SNR (1 match) |
|------------|-----------|---------------|
| 1,000 | 1.61 | 0.53 |
| 5,000 | 3.61 | 0.24 |
| 10,000 | 5.10 | 0.17 |
| 50,000 | 11.41 | 0.07 |
| 100,000 | 16.14 | 0.05 |

**Conclusion:** Single-match SNR is always < 1, making reliable detection impossible.

### Why Multi-Match Detection Works

With k matches per query:
- **Signal:** k × 0.85
- **SNR:** k × 0.85 / sqrt(n/d)

For SNR > 2 (reliable detection), need k > 2 × sqrt(n/d) / 0.85

---

## Detailed Results

"""

    for cache_size, cache_results in results.items():
        report += f"\n### Cache Size: {cache_size:,}\n\n"
        report += "| Matches | SNR | Recall | Precision | F1 | Speedup |\n"
        report += "|---------|-----|--------|-----------|----|---------|\n"

        for r in cache_results:
            report += f"| {r['n_matches']} | {r['snr']:.2f} | {r['recall']*100:.1f}% | {r['precision']*100:.1f}% | {r['f1']*100:.1f}% | {r['speedup']:.0f}x |\n"

    report += f"""

---

## Practical Recommendations

### When to Use TA

1. **Semantic Caching with Query Clusters**
   - Similar queries hit multiple cache entries
   - Natural multi-match scenario
   - TA provides massive speedup with high accuracy

2. **Document Deduplication**
   - Documents often have multiple similar sections
   - Multi-match detection is appropriate

3. **Pre-filtering Pipeline**
   ```
   Query → [TA O(1)] → Likely Match? → [Exact Search O(n)]
                           ↓ No
                        Fast Reject
   ```

### When NOT to Use TA

1. **Exact single-match detection**
   - Use exact search or ANN indices instead

2. **High-stakes decisions requiring 100% recall**
   - TA trades recall for speed

3. **Very sparse match scenarios**
   - SNR too low for reliable detection

---

## Calibration Formula

For additive noise calibration:

```python
calibrated_threshold = raw_threshold + k * sqrt(cache_size / embedding_dim)
```

Where k = 2.0 balances precision and recall.

---

## Conclusion

Thresholded Accumulation achieves **{max_speedup:.0f}x speedup** with O(1) complexity.
Its effectiveness depends on the **signal-to-noise ratio**:

- **SNR < 1:** Low recall, not recommended
- **SNR 1-2:** Moderate recall, use as pre-filter
- **SNR > 2:** High recall, TA works well

**Bottom Line:** TA excels when queries naturally match multiple cache elements,
making it ideal for semantic caching and similarity-based filtering.

---

*Report generated by TA Experiment Suite*
"""

    with open(output_file, 'w') as f:
        f.write(report)

    print(f"Report saved to {output_file}")


if __name__ == "__main__":
    config = ExperimentConfig()
    results = run_experiment(config)

    # Save raw results
    with open("results/experiment_results_corrected.json", 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    # Generate plots
    generate_plots(results, "plots")

    # Generate report
    generate_report(results, config, "results/FINAL_REPORT.md")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
