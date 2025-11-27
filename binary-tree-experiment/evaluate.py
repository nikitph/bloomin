"""
Evaluation and Comparison Script

Compares standard vs hyperbolic branch Transformers across different model sizes.
Generates plots showing the dramatic difference in parameter efficiency.
"""

import torch
import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from tree_dataset import BinaryTreePathDataset, collate_batch
from standard_transformer import StandardTransformer
from hyperbolic_transformer import HyperbolicBranchTransformer


def load_model_and_evaluate(checkpoint_path, dataset, device):
    """Load a trained model and evaluate it."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = argparse.Namespace(**checkpoint['args'])
    
    # Create model
    vocab_size = dataset.get_vocab_size()
    
    if args.model_type == 'standard':
        model = StandardTransformer(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            num_classes=args.value_range,
            dropout=0.0  # No dropout for evaluation
        )
    else:
        model = HyperbolicBranchTransformer(
            vocab_size=vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            d_ff=args.d_ff,
            bucket_sizes=tuple(args.bucket_sizes),
            num_classes=args.value_range,
            dropout=0.0
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Evaluate
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_batch)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            targets = batch['targets'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)
            
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    
    accuracy = 100 * correct / total
    n_params = model.count_parameters()
    
    return accuracy, n_params


def plot_comparison(results, save_path='comparison.png'):
    """
    Plot accuracy vs model size for both architectures.
    
    Args:
        results: dict with keys 'standard' and 'hyperbolic', each containing
                 list of (n_params, accuracy) tuples
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot standard transformer
    if 'standard' in results and results['standard']:
        std_params, std_acc = zip(*results['standard'])
        ax.plot(std_params, std_acc, 'o-', label='Standard Transformer', 
                color='#e74c3c', linewidth=2, markersize=8)
    
    # Plot hyperbolic transformer
    if 'hyperbolic' in results and results['hyperbolic']:
        hyp_params, hyp_acc = zip(*results['hyperbolic'])
        ax.plot(hyp_params, hyp_acc, 's-', label='Hyperbolic Branch', 
                color='#2ecc71', linewidth=2, markersize=8)
    
    # Add 50% accuracy threshold line
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Binary Tree Path Prediction (Depth 20)\nStandard vs Hyperbolic Branch Transformer', 
                 fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot to {save_path}")
    
    return fig


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create test dataset
    print("Creating test dataset...")
    test_dataset = BinaryTreePathDataset(
        num_samples=args.test_samples,
        depth=args.depth,
        value_range=args.value_range,
        seed=999
    )
    
    # Collect results from saved checkpoints
    results = {'standard': [], 'hyperbolic': []}
    
    if os.path.exists(args.results_dir):
        print(f"\nLoading results from {args.results_dir}...")
        
        for filename in os.listdir(args.results_dir):
            if filename.endswith('_results.json'):
                filepath = os.path.join(args.results_dir, filename)
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                model_type = data['model_type']
                n_params = data['n_params']
                best_acc = data['best_val_acc']
                
                results[model_type].append((n_params, best_acc))
                
                print(f"  {model_type:12s} | d={data['d_model']:4d} | "
                      f"params={n_params:10,d} | acc={best_acc:5.2f}%")
    
    # Sort by number of parameters
    for model_type in results:
        results[model_type].sort(key=lambda x: x[0])
    
    # Generate comparison plot
    if any(results.values()):
        print("\nGenerating comparison plot...")
        plot_comparison(results, save_path=args.output_plot)
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        # Find smallest model that achieves >50% for each type
        for model_type in ['standard', 'hyperbolic']:
            if results[model_type]:
                above_50 = [(p, a) for p, a in results[model_type] if a > 50]
                if above_50:
                    min_params, acc = min(above_50, key=lambda x: x[0])
                    print(f"{model_type.capitalize():12s}: {min_params:,} params needed for >50% acc ({acc:.2f}%)")
                else:
                    max_acc = max(results[model_type], key=lambda x: x[1])[1]
                    print(f"{model_type.capitalize():12s}: Failed to reach 50% (best: {max_acc:.2f}%)")
        
        print("="*60)
    else:
        print("\nNo results found. Train some models first!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate and compare models')
    
    parser.add_argument('--results-dir', type=str, default='checkpoints',
                        help='Directory containing saved results')
    parser.add_argument('--output-plot', type=str, default='comparison.png',
                        help='Output plot filename')
    parser.add_argument('--depth', type=int, default=20, help='Tree depth')
    parser.add_argument('--value-range', type=int, default=1000, help='Value range')
    parser.add_argument('--test-samples', type=int, default=2000, help='Test samples')
    
    args = parser.parse_args()
    main(args)
