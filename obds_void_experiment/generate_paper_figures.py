"""
Generate publication-quality figures for the paper
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 10

# Set publication style
plt.style.use('seaborn-v0_8-paper')

def generate_figure1_scaling():
    """Figure 1: Main scaling result (CPU vs GPU)"""
    # Load data from our benchmarks
    data = {
        'N': [10000, 50000, 100000, 500000, 1000000],
        'classical': [0.1648, 0.8935, 1.7552, 9.9663, 20.7436],
        'gpu': [0.7013, 0.0190, 0.0219, 0.0385, 0.0304]
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    ax.loglog(data['N'], data['classical'], 'o-', label='Classical (CPU)', 
              linewidth=2, markersize=8, color='#d62728')
    ax.loglog(data['N'], data['gpu'], 's-', label='GPU Distance Transform', 
              linewidth=2, markersize=8, color='#2ca02c')
    
    ax.set_xlabel('Number of Points (N)', fontsize=12)
    ax.set_ylabel('Query Time (s)', fontsize=12)
    ax.set_title('Void Detection Scaling', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    
    # Add annotation
    ax.annotate('683× speedup\nat N=1M', 
                xy=(1000000, 0.0304), xytext=(500000, 0.1),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig('obds_void_experiment/paper_fig1_scaling.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('obds_void_experiment/paper_fig1_scaling.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 1 saved")

def generate_figure2_speedup():
    """Figure 2: Speedup vs N"""
    data = {
        'N': [10000, 50000, 100000, 500000, 1000000],
        'speedup': [0.2, 46.9, 80.2, 258.8, 683.2]
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    ax.semilogx(data['N'], data['speedup'], 'o-', linewidth=2.5, 
                markersize=10, color='#1f77b4')
    
    ax.set_xlabel('Number of Points (N)', fontsize=12)
    ax.set_ylabel('Speedup (Classical / GPU)', fontsize=12)
    ax.set_title('GPU Speedup Growth', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='No speedup')
    ax.legend(fontsize=10)
    
    # Annotate max
    ax.annotate(f'{data["speedup"][-1]:.0f}×', 
                xy=(data['N'][-1], data['speedup'][-1]), 
                xytext=(data['N'][-1]/2, data['speedup'][-1]+50),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=11, ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('obds_void_experiment/paper_fig2_speedup.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('obds_void_experiment/paper_fig2_speedup.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 2 saved")

def generate_figure3_lidar():
    """Figure 3: LiDAR real-time validation"""
    # Load LiDAR data
    df = pd.read_csv('obds_void_experiment/lidar_gpu_benchmark.csv')
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Query time
    ax1 = axes[0]
    ax1.plot(df['points'], df['classical_time'], 'o-', label='Classical', 
             linewidth=2, markersize=6, color='#d62728')
    ax1.plot(df['points'], df['gpu_query'], 's-', label='GPU', 
             linewidth=2, markersize=6, color='#2ca02c')
    ax1.axhline(y=0.033, color='orange', linestyle='--', linewidth=2, 
                label='30 Hz threshold (33ms)', alpha=0.8)
    
    ax1.set_xlabel('Points per Frame', fontsize=11)
    ax1.set_ylabel('Query Time (s)', fontsize=11)
    ax1.set_title('LiDAR Query Time', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Speedup
    ax2 = axes[1]
    ax2.plot(df['points'], df['speedup'], 'o-', linewidth=2, 
             markersize=7, color='#9467bd')
    ax2.set_xlabel('Points per Frame', fontsize=11)
    ax2.set_ylabel('Speedup (×)', fontsize=11)
    ax2.set_title('LiDAR Speedup per Frame', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add mean line
    mean_speedup = df['speedup'].mean()
    ax2.axhline(y=mean_speedup, color='red', linestyle='--', 
                linewidth=1.5, alpha=0.7, label=f'Mean: {mean_speedup:.1f}×')
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('obds_void_experiment/paper_fig3_lidar.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('obds_void_experiment/paper_fig3_lidar.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 3 saved")

def generate_figure4_extreme():
    """Figure 4: Extreme scale (10M) validation"""
    data = {
        'N': [100000, 500000, 1000000, 2000000, 5000000, 10000000],
        'query_time': [0.327, 0.033, 0.039, 0.044, 0.069, 0.063]
    }
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    ax.plot(data['N'], data['query_time'], 'o-', linewidth=2.5, 
            markersize=10, color='#ff7f0e')
    
    ax.set_xlabel('Number of Points (N)', fontsize=12)
    ax.set_ylabel('GPU Query Time (s)', fontsize=12)
    ax.set_title('Extreme-Scale Validation (10M Points)', fontsize=13, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    
    # Add horizontal line for mean of last 4 points
    mean_query = np.mean(data['query_time'][-4:])
    ax.axhline(y=mean_query, color='green', linestyle='--', linewidth=2, 
               alpha=0.7, label=f'Mean (1M-10M): {mean_query*1000:.0f}ms')
    
    # Annotate key points
    ax.annotate('100× increase in N\n1.9× increase in time', 
                xy=(5e6, 0.05), xytext=(1e6, 0.15),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, ha='center', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.5))
    
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('obds_void_experiment/paper_fig4_extreme.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('obds_void_experiment/paper_fig4_extreme.png', dpi=300, bbox_inches='tight')
    print("✓ Figure 4 saved")

def generate_all_figures():
    """Generate all paper figures"""
    print("="*60)
    print("Generating Publication Figures")
    print("="*60)
    
    generate_figure1_scaling()
    generate_figure2_speedup()
    generate_figure3_lidar()
    generate_figure4_extreme()
    
    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("="*60)
    print("\nFigure Summary:")
    print("  Fig 1: Core scaling result (CPU vs GPU)")
    print("  Fig 2: Speedup growth")
    print("  Fig 3: LiDAR real-time validation")
    print("  Fig 4: Extreme-scale (10M) confirmation")
    print("\nFormats: PDF (vector) + PNG (raster)")

if __name__ == "__main__":
    generate_all_figures()
