import matplotlib.pyplot as plt
import numpy as np
import json
import os
from sklearn.decomposition import PCA

def create_all_plots(results, save_dir='./results/plots'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Pack results into dashboard
    plot_hierarchy_maintenance(results, f"{save_dir}/hierarchy_maintenance.png")
    plot_sharpness_evolution(results['threepath']['history'], f"{save_dir}/sharpness_evolution.png")
    plot_computational_costs(results['threepath']['costs'], f"{save_dir}/computational_costs.png")
    
    # Save raw results
    with open(f"{save_dir}/../metrics/results_summary.json", 'w') as f:
        # Convert non-serializable types if any
        json.dump(results, f, indent=2, default=str)

def plot_hierarchy_maintenance(results, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dynamic depth detection from results
    # results['baseline']['hierarchy'] keys are integers in the fresh dict
    depths = sorted(list(results['baseline']['hierarchy'].keys()))
    
    # Baseline
    base_acc = [results['baseline']['hierarchy'][d] for d in depths]
    tp_acc = [results['threepath']['hierarchy'][d] for d in depths]
    
    ax.plot(depths, tp_acc, 'o-', color='blue', linewidth=2, markersize=8, label='Three-Path (with sleep)')
    ax.plot(depths, base_acc, 's--', color='red', linewidth=2, markersize=8, label='Baseline (no sleep)')
    
    ax.set_xlabel('Hierarchy Depth', fontsize=14)
    ax.set_ylabel('Recovery Accuracy', fontsize=14)
    ax.set_xticks(depths)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Hierarchy Maintenance', fontsize=16)
    
    plt.savefig(save_path)
    plt.close()

def plot_sharpness_evolution(history, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(len(history))
    wake = [h['wake_entropy'] for h in history]
    sleep = [h['sleep_entropy'] for h in history]
    
    ax.plot(epochs, wake, 'r-', label='Wake Entropy')
    ax.plot(epochs, sleep, 'b-', label='Sleep Entropy')
    
    # Fill between to show sharpening
    ax.fill_between(epochs, wake, sleep, color='green', alpha=0.1, label='Consolidation')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Entropy (Blur)')
    ax.legend()
    ax.set_title('Sharpness Evolution')
    
    plt.savefig(save_path)
    plt.close()

def plot_computational_costs(costs, save_path):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ops = ['Fast', 'Slow']
    times = [costs['fast_ms'], costs['slow_ms']]
    
    ax.bar(ops, times, color=['green', 'orange'])
    
    ratio = costs['ratio']
    ax.text(1, times[1], f"{ratio:.1f}x", ha='center', va='bottom')
    
    ax.set_ylabel('Time (ms)')
    ax.set_title('Computational Cost')
    
    plt.savefig(save_path)
    plt.close()
