"""
Visualization module for TA experiment results
Generates publication-ready plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from typing import List
from dataclasses import dataclass

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

# Set publication style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'serif'


def load_results(filename: str) -> List[ExperimentResults]:
    """Load results from JSON"""
    with open(filename, 'r') as f:
        data = json.load(f)

    return [
        ExperimentResults(
            cache_size=r['cache_size'],
            search_time_mean=r['search_time_mean'],
            search_time_std=r['search_time_std'],
            ta_time_mean=r['ta_time_mean'],
            ta_time_std=r['ta_time_std'],
            speedup=r['speedup'],
            accuracy=r['accuracy'],
            false_positive_rate=r['false_positive_rate'],
            false_negative_rate=r['false_negative_rate']
        )
        for r in data
    ]


def plot_time_scaling(results: List[ExperimentResults], save_path: str):
    """
    Plot 1: Decision time vs cache size
    Shows O(n) search vs O(1) TA
    """
    cache_sizes = [r.cache_size for r in results]
    search_times = [r.search_time_mean * 1000 for r in results]  # Convert to ms
    search_stds = [r.search_time_std * 1000 for r in results]
    ta_times = [r.ta_time_mean * 1000 for r in results]
    ta_stds = [r.ta_time_std * 1000 for r in results]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot with error bars
    ax.errorbar(cache_sizes, search_times, yerr=search_stds,
                marker='o', markersize=8, linewidth=2.5, capsize=5,
                label='Search-Based Decision O(n)', color='#e74c3c')

    ax.errorbar(cache_sizes, ta_times, yerr=ta_stds,
                marker='s', markersize=8, linewidth=2.5, capsize=5,
                label='Thresholded Accumulation O(1)', color='#2ecc71')

    ax.set_xlabel('Cache Size (number of elements)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Decision Time (ms)', fontsize=16, fontweight='bold')
    ax.set_title('Complexity Separation: Search O(n) vs TA O(1)',
                 fontsize=18, fontweight='bold', pad=20)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=14, loc='upper left', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add annotation for key insight
    ax.annotate('Exponential\nSeparation',
                xy=(cache_sizes[-1], ta_times[-1]),
                xytext=(cache_sizes[-3], search_times[-1]/2),
                fontsize=12, fontweight='bold',
                arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_speedup(results: List[ExperimentResults], save_path: str):
    """
    Plot 2: Speedup factor vs cache size
    """
    cache_sizes = [r.cache_size for r in results]
    speedups = [r.speedup for r in results]

    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(cache_sizes, speedups, marker='D', markersize=10,
            linewidth=3, color='#9b59b6', label='TA Speedup')

    # Add horizontal line at 1x
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='No Speedup')

    ax.set_xlabel('Cache Size (number of elements)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Speedup Factor (x)', fontsize=16, fontweight='bold')
    ax.set_title('Thresholded Accumulation Speedup vs Traditional Search',
                 fontsize=18, fontweight='bold', pad=20)

    ax.set_xscale('log')
    ax.legend(fontsize=14, loc='lower right', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Annotate maximum speedup
    max_speedup_idx = np.argmax(speedups)
    ax.annotate(f'{speedups[max_speedup_idx]:.0f}x faster',
                xy=(cache_sizes[max_speedup_idx], speedups[max_speedup_idx]),
                xytext=(cache_sizes[max_speedup_idx]/10, speedups[max_speedup_idx]*0.7),
                fontsize=14, fontweight='bold', color='#9b59b6',
                arrowprops=dict(arrowstyle='->', lw=2.5, color='#9b59b6'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_accuracy_analysis(results: List[ExperimentResults], save_path: str):
    """
    Plot 3: Accuracy and error rates
    """
    cache_sizes = [r.cache_size for r in results]
    accuracies = [r.accuracy * 100 for r in results]
    fprs = [r.false_positive_rate * 100 for r in results]
    fnrs = [r.false_negative_rate * 100 for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Accuracy plot
    ax1.plot(cache_sizes, accuracies, marker='o', markersize=8,
             linewidth=3, color='#3498db', label='Decision Accuracy')
    ax1.axhline(y=95, color='green', linestyle='--', linewidth=2,
                alpha=0.5, label='95% Threshold')
    ax1.set_xlabel('Cache Size', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax1.set_title('TA Decision Accuracy', fontsize=16, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_ylim([min(accuracies) - 5, 101])
    ax1.legend(fontsize=12, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')

    # Error rates plot
    ax2.plot(cache_sizes, fprs, marker='^', markersize=8,
             linewidth=2.5, color='#e74c3c', label='False Positive Rate')
    ax2.plot(cache_sizes, fnrs, marker='v', markersize=8,
             linewidth=2.5, color='#f39c12', label='False Negative Rate')
    ax2.set_xlabel('Cache Size', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Error Rate (%)', fontsize=14, fontweight='bold')
    ax2.set_title('TA Error Analysis', fontsize=16, fontweight='bold')
    ax2.set_xscale('log')
    ax2.legend(fontsize=12, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_complexity_comparison(results: List[ExperimentResults], save_path: str):
    """
    Plot 4: Per-query cost comparison with theoretical bounds
    """
    cache_sizes = np.array([r.cache_size for r in results])
    search_per_query = np.array([r.search_time_mean / 1000 * 1e6 for r in results])  # microseconds
    ta_per_query = np.array([r.ta_time_mean / 1000 * 1e6 for r in results])

    fig, ax = plt.subplots(figsize=(12, 8))

    # Actual measurements
    ax.scatter(cache_sizes, search_per_query, s=100, color='#e74c3c',
               label='Search (measured)', alpha=0.7, marker='o')
    ax.scatter(cache_sizes, ta_per_query, s=100, color='#2ecc71',
               label='TA (measured)', alpha=0.7, marker='s')

    # Theoretical complexity curves
    # O(n) curve fitted to search data
    linear_fit = np.polyfit(np.log(cache_sizes), np.log(search_per_query), 1)
    theoretical_linear = np.exp(linear_fit[1]) * cache_sizes ** linear_fit[0]
    ax.plot(cache_sizes, theoretical_linear, '--', linewidth=2.5,
            color='#c0392b', label=f'O(n) theoretical', alpha=0.7)

    # O(1) average for TA
    ta_mean = np.mean(ta_per_query)
    ax.axhline(y=ta_mean, linestyle='--', linewidth=2.5,
               color='#27ae60', label='O(1) theoretical', alpha=0.7)

    ax.set_xlabel('Cache Size (n)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Per-Query Cost (us)', fontsize=16, fontweight='bold')
    ax.set_title('Computational Complexity: Theory vs Practice',
                 fontsize=18, fontweight='bold', pad=20)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=13, loc='upper left', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_all_plots(results_file: str):
    """Generate all publication plots"""
    print("Loading results...")
    results = load_results(results_file)

    print("\nGenerating plots...")
    plot_time_scaling(results, "plots/01_time_scaling.png")
    plot_speedup(results, "plots/02_speedup_factor.png")
    plot_accuracy_analysis(results, "plots/03_accuracy_analysis.png")
    plot_complexity_comparison(results, "plots/04_complexity_comparison.png")

    print("\n" + "="*80)
    print("ALL PLOTS GENERATED SUCCESSFULLY")
    print("="*80)
    print("\nPlots saved in: plots/")
    print("  - 01_time_scaling.png")
    print("  - 02_speedup_factor.png")
    print("  - 03_accuracy_analysis.png")
    print("  - 04_complexity_comparison.png")


if __name__ == "__main__":
    generate_all_plots("results/experiment_results.json")
