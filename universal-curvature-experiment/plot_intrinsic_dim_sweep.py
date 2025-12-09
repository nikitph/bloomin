"""
Plot Intrinsic Dimension Results
=================================

Visualization script for intrinsic dimension sweep results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

sns.set_style('whitegrid')
sns.set_palette('husl')


def plot_intrinsic_dim_comparison(results_file: str, output_dir: str = './results'):
    """Plot intrinsic dimension comparison across models."""
    
    # Load results
    df = pd.read_csv(results_file)
    df = df[df['d_intrinsic_95'].notna()]  # Filter out errors
    
    if len(df) == 0:
        print("No valid results to plot.")
        return
    
    # Sort by intrinsic dimension
    df = df.sort_values('d_intrinsic_95')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Intrinsic Dimension by Model
    ax = axes[0, 0]
    bars = ax.barh(df['model'], df['d_intrinsic_95'], color='steelblue', alpha=0.7)
    ax.axvline(df['d_intrinsic_95'].mean(), color='red', linestyle='--', 
               label=f'Mean: {df["d_intrinsic_95"].mean():.1f}')
    ax.set_xlabel('Intrinsic Dimension (95% variance)', fontsize=12)
    ax.set_title('Intrinsic Dimension Across Models', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # 2. Compression Ratio
    ax = axes[0, 1]
    bars = ax.barh(df['model'], df['compression_95'], color='coral', alpha=0.7)
    ax.axvline(df['compression_95'].mean(), color='red', linestyle='--',
               label=f'Mean: {df["compression_95"].mean():.2f}x')
    ax.set_xlabel('Compression Ratio (95% variance)', fontsize=12)
    ax.set_title('Compression Potential', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # 3. Effective Rank
    ax = axes[1, 0]
    bars = ax.barh(df['model'], df['effective_rank'], color='mediumseagreen', alpha=0.7)
    ax.axvline(df['effective_rank'].mean(), color='red', linestyle='--',
               label=f'Mean: {df["effective_rank"].mean():.1f}')
    ax.set_xlabel('Effective Rank', fontsize=12)
    ax.set_title('Effective Rank (Participation Ratio)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # 4. Variance Explained by Top Components
    ax = axes[1, 1]
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df['variance_first_pc'] * 100, width, label='1st PC', alpha=0.7)
    ax.bar(x, df['variance_top10_pcs'] * 100, width, label='Top 10 PCs', alpha=0.7)
    ax.bar(x + width, df['variance_top50_pcs'] * 100, width, label='Top 50 PCs', alpha=0.7)
    ax.set_ylabel('Variance Explained (%)', fontsize=12)
    ax.set_title('Variance Concentration', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_file = os.path.join(output_dir, 'intrinsic_dim_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_file}")
    plt.close()


def plot_variance_spectra(results_dir: str, output_dir: str = './results'):
    """Plot variance spectra for all models."""
    
    # Load all variance spectra
    spectra = {}
    for file in os.listdir(results_dir):
        if file.startswith('variance_spectrum_') and file.endswith('.json'):
            with open(os.path.join(results_dir, file), 'r') as f:
                data = json.load(f)
                spectra[data['model']] = data['variance_spectrum']
    
    if not spectra:
        print("No variance spectra found.")
        return
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for model, spectrum in spectra.items():
        ax.plot(range(1, len(spectrum) + 1), np.cumsum(spectrum) * 100, 
                label=model, linewidth=2, alpha=0.7)
    
    ax.axhline(90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    ax.axhline(95, color='gray', linestyle=':', alpha=0.5, label='95% threshold')
    ax.axhline(99, color='gray', linestyle='-.', alpha=0.5, label='99% threshold')
    
    ax.set_xlabel('Number of Principal Components', fontsize=12)
    ax.set_ylabel('Cumulative Variance Explained (%)', fontsize=12)
    ax.set_title('Variance Spectra Across Models', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    # Save
    output_file = os.path.join(output_dir, 'variance_spectra.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved variance spectra to {output_file}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot intrinsic dimension results')
    parser.add_argument('--results-dir', type=str, default='./results',
                       help='Directory containing results')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    results_file = os.path.join(args.results_dir, 'intrinsic_dim_results.csv')
    
    if not os.path.exists(results_file):
        print(f"Results file not found: {results_file}")
        return
    
    print("Generating plots...")
    plot_intrinsic_dim_comparison(results_file, args.output_dir)
    plot_variance_spectra(args.results_dir, args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
