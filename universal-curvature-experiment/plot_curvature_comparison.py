import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def plot_curvature_results():
    results_path = 'results/curvature_results.csv'
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    df = pd.read_csv(results_path)
    
    # Sort by K_mean for better visualization
    df = df.sort_values('K_mean')
    
    # Setup style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    # Create colors based on Arch or Domain
    # Let's use Domain for color distinction
    domains = df['domain'].unique()
    palette = sns.color_palette("husl", len(domains))
    domain_colors = dict(zip(domains, palette))
    colors = df['domain'].map(domain_colors)
    
    # Bar plot with Error Bars
    # Note: K_std might be very small for some, large for others (like CodeBERT)
    bars = plt.bar(df['model'], df['K_mean'], yerr=df['K_std'], capsize=5, 
                   color=colors, alpha=0.8, edgecolor='black')
    
    # Add K=1 Reference Line (Sphere)
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='K=1 (Unit Sphere)')
    
    # Labels and Title
    plt.ylabel('Gaussian Curvature (K)', fontsize=12, fontweight='bold')
    plt.title('Universal Curvature Across Models', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0.95, 1.15) # Zoom in to see differences, assuming all near 1
    
    # Legend for Domains
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=domain_colors[d], edgecolor='black', label=d) 
                       for d in domains]
    plt.legend(handles=legend_elements, title="Domain", loc='upper left')
    
    # Add exact values on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.002, 
                 f'{height:.3f}', ha='center', va='bottom', fontsize=9, rotation=0)

    # Save
    output_path = 'results/curvature_comparison.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    plot_curvature_results()
