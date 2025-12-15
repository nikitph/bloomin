#!/usr/bin/env python3
"""
Thermal Bloom Filter - Visualization Script
Generates the key plots for PoC validation
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_results():
    """Load benchmark results from JSON"""
    with open('thermal_bloom_v2_results.json', 'r') as f:
        return json.load(f)

def plot_parameter_heatmap(results):
    """Plot recall@1 heatmap for different parameter combinations"""
    # Filter thermal results
    thermal_results = [r for r in results if r['method'].startswith('Thermal')]

    # Extract unique values
    grid_sizes = sorted(set(r['grid_size'] for r in thermal_results))
    sigmas = sorted(set(r['sigma'] for r in thermal_results))
    radii = sorted(set(r['search_radius'] for r in thermal_results))

    fig, axes = plt.subplots(1, len(radii), figsize=(15, 4))

    for idx, radius in enumerate(radii):
        ax = axes[idx]

        # Build heatmap data
        data = np.zeros((len(sigmas), len(grid_sizes)))
        for r in thermal_results:
            if r['search_radius'] == radius:
                i = sigmas.index(r['sigma'])
                j = grid_sizes.index(r['grid_size'])
                data[i, j] = r['recall_at_1'] * 100

        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=30, vmax=100)
        ax.set_xticks(range(len(grid_sizes)))
        ax.set_xticklabels(grid_sizes)
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([f'{s:.1f}' for s in sigmas])
        ax.set_xlabel('Grid Size')
        ax.set_ylabel('Sigma')
        ax.set_title(f'Search Radius = {radius}')

        # Add text annotations
        for i in range(len(sigmas)):
            for j in range(len(grid_sizes)):
                ax.text(j, i, f'{data[i,j]:.0f}%', ha='center', va='center', fontsize=8)

    plt.colorbar(im, ax=axes.ravel().tolist(), label='Recall@1 (%)')
    plt.suptitle('Thermal Bloom V2: Parameter Sensitivity', fontsize=14)
    plt.tight_layout()
    plt.savefig('parameter_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: parameter_heatmap.png")

def plot_recall_vs_sigma(results):
    """Plot recall vs diffusion sigma for different grid sizes"""
    thermal_results = [r for r in results if r['method'].startswith('Thermal')]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Fixed search radius = 3 (middle value)
    r3_results = [r for r in thermal_results if r['search_radius'] == 3]

    grid_sizes = sorted(set(r['grid_size'] for r in r3_results))
    sigmas = sorted(set(r['sigma'] for r in r3_results))

    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    for grid_size, color in zip(grid_sizes, colors):
        grid_results = sorted([r for r in r3_results if r['grid_size'] == grid_size],
                             key=lambda x: x['sigma'])
        sigmas_g = [r['sigma'] for r in grid_results]
        recall1 = [r['recall_at_1'] * 100 for r in grid_results]
        steps = [r['avg_steps'] for r in grid_results]

        ax1.plot(sigmas_g, recall1, 'o-', color=color, label=f'{grid_size}x{grid_size}', linewidth=2, markersize=8)
        ax2.plot(sigmas_g, steps, 's--', color=color, label=f'{grid_size}x{grid_size}', linewidth=2, markersize=8)

    ax1.axhline(y=85, color='gray', linestyle=':', label='Holy Shit threshold (85%)')
    ax1.axhline(y=70, color='gray', linestyle='--', alpha=0.5, label='Strong threshold (70%)')
    ax1.axhline(y=50, color='gray', linestyle='-.', alpha=0.3, label='MVP threshold (50%)')

    ax1.set_xlabel('Diffusion Sigma (σ)', fontsize=12)
    ax1.set_ylabel('Recall@1 (%)', fontsize=12)
    ax1.set_title('Recall vs Diffusion Parameter', fontsize=12)
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(30, 105)

    ax2.axhline(y=20, color='gray', linestyle=':', label='MVP threshold (20 steps)')
    ax2.set_xlabel('Diffusion Sigma (σ)', fontsize=12)
    ax2.set_ylabel('Average Gradient Steps', fontsize=12)
    ax2.set_title('Steps vs Diffusion Parameter', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('recall_vs_sigma.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: recall_vs_sigma.png")

def plot_speed_accuracy_tradeoff(results):
    """Plot speed vs accuracy tradeoff"""
    thermal_results = [r for r in results if r['method'].startswith('Thermal')]
    discrete = next(r for r in results if r['method'] == 'DiscreteBloom')

    fig, ax = plt.subplots(figsize=(10, 7))

    # Color by search radius
    colors = {2: '#e41a1c', 3: '#377eb8', 5: '#4daf4a'}
    markers = {128: 'o', 256: 's', 512: '^'}

    for r in thermal_results:
        ax.scatter(r['queries_per_second'] / 1000, r['recall_at_1'] * 100,
                  c=colors[r['search_radius']],
                  marker=markers[r['grid_size']],
                  s=100, alpha=0.7)

    # Discrete baseline
    ax.scatter(discrete['queries_per_second'] / 1000, discrete['recall_at_1'] * 100,
              c='black', marker='*', s=300, label='Discrete Bloom', zorder=5)

    # Thresholds
    ax.axhline(y=85, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(y=70, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=50, color='gray', linestyle='-.', alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Grid 128'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='Grid 256'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='Grid 512'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#e41a1c', markersize=10, label='Radius 2'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#377eb8', markersize=10, label='Radius 3'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4daf4a', markersize=10, label='Radius 5'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='black', markersize=15, label='Discrete Bloom'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    ax.set_xlabel('Queries Per Second (K)', fontsize=12)
    ax.set_ylabel('Recall@1 (%)', fontsize=12)
    ax.set_title('Speed vs Accuracy Tradeoff', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(100, 900)
    ax.set_ylim(30, 105)

    # Annotate Pareto frontier
    ax.annotate('Pareto\nFrontier', xy=(400, 95), fontsize=10, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig('speed_accuracy_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: speed_accuracy_tradeoff.png")

def plot_comparison_bar(results):
    """Simple bar comparison of best thermal vs discrete"""
    best_thermal = max([r for r in results if r['method'].startswith('Thermal')],
                       key=lambda x: x['recall_at_1'])
    discrete = next(r for r in results if r['method'] == 'DiscreteBloom')

    fig, ax = plt.subplots(figsize=(8, 5))

    methods = ['Discrete Bloom', 'Thermal Bloom V2']
    recall1 = [discrete['recall_at_1'] * 100, best_thermal['recall_at_1'] * 100]

    bars = ax.bar(methods, recall1, color=['#888888', '#2ecc71'], edgecolor='black')

    # Add value labels
    for bar, val in zip(bars, recall1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Threshold lines
    ax.axhline(y=85, color='red', linestyle='--', label='Holy Shit threshold (85%)')
    ax.axhline(y=70, color='orange', linestyle='--', label='Strong threshold (70%)')
    ax.axhline(y=50, color='gray', linestyle='--', label='MVP threshold (50%)')

    ax.set_ylabel('Recall@1 (%)', fontsize=12)
    ax.set_title('Thermal Bloom vs Discrete Bloom', fontsize=14)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right')

    # Add improvement annotation
    improvement = best_thermal['recall_at_1'] / discrete['recall_at_1']
    ax.annotate(f'{improvement:.1f}x improvement',
                xy=(0.5, 60), xytext=(0.5, 75),
                ha='center', fontsize=12, color='green',
                arrowprops=dict(arrowstyle='->', color='green'))

    plt.tight_layout()
    plt.savefig('comparison_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: comparison_bar.png")

def print_summary(results):
    """Print summary table"""
    best = max([r for r in results if r['method'].startswith('Thermal')],
               key=lambda x: x['recall_at_1'])
    discrete = next(r for r in results if r['method'] == 'DiscreteBloom')

    print("\n" + "="*60)
    print("THERMAL BLOOM FILTER V2 - SUMMARY")
    print("="*60)

    print(f"\nBest Configuration: {best['method']}")
    print(f"  Grid Size:      {best['grid_size']}x{best['grid_size']}")
    print(f"  Sigma:          {best['sigma']}")
    print(f"  Search Radius:  {best['search_radius']}")
    print(f"  Recall@1:       {best['recall_at_1']*100:.1f}%")
    print(f"  Recall@5:       {best['recall_at_5']*100:.1f}%")
    print(f"  Avg Steps:      {best['avg_steps']:.1f}")
    print(f"  Avg Candidates: {best['avg_candidates']:.1f}")
    print(f"  QPS:            {best['queries_per_second']:,.0f}")

    print(f"\nDiscrete Bloom Baseline:")
    print(f"  Recall@1:       {discrete['recall_at_1']*100:.1f}%")
    print(f"  Recall@5:       {discrete['recall_at_5']*100:.1f}%")

    print(f"\nImprovement: {best['recall_at_1']/discrete['recall_at_1']:.1f}x")

    print("\n" + "-"*40)
    print("SUCCESS CRITERIA:")
    print("-"*40)
    mvp = best['recall_at_1'] > 0.5 and best['avg_steps'] < 20
    strong = best['recall_at_1'] > 0.7
    holy = best['recall_at_1'] > 0.85
    print(f"  [{'PASS' if mvp else 'FAIL'}] MVP: Recall@1 > 50%, Steps < 20")
    print(f"  [{'PASS' if strong else 'FAIL'}] Strong: Recall@1 > 70%")
    print(f"  [{'PASS' if holy else 'FAIL'}] Holy Shit: Recall@1 > 85%")

    print("\n" + "="*60)

def main():
    print("Loading results...")
    results = load_results()

    print("\nGenerating visualizations...")
    plot_parameter_heatmap(results)
    plot_recall_vs_sigma(results)
    plot_speed_accuracy_tradeoff(results)
    plot_comparison_bar(results)

    print_summary(results)

if __name__ == '__main__':
    main()
