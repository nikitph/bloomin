#!/usr/bin/env python3
"""
Main entry point for the Ricci-REWA Continual Learning Experiment.

This experiment tests whether preserving Ricci curvature during continual
learning can prevent catastrophic forgetting more effectively than
parameter-space methods like EWC.

Usage:
    # Quick test (subset of data, fewer epochs)
    python run_experiment.py --quick

    # Full experiment
    python run_experiment.py --full

    # Hyperparameter sweep
    python run_experiment.py --sweep

    # Custom configuration
    python run_experiment.py --epochs-a 15 --epochs-b 15 --ricci-lambda 2.0
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import argparse
import torch
from src.experiment import ContinualLearningExperiment, run_continual_learning_experiment


def main():
    parser = argparse.ArgumentParser(
        description='Ricci-REWA Continual Learning Experiment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test
    python run_experiment.py --quick

    # Full experiment with default settings
    python run_experiment.py --full

    # Hyperparameter sweep
    python run_experiment.py --sweep

    # Custom settings
    python run_experiment.py --epochs-a 15 --ricci-lambda 0.5
        """
    )

    # Experiment type
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--quick', action='store_true',
                       help='Quick test with small subset (1000 samples, 3 epochs)')
    group.add_argument('--full', action='store_true',
                       help='Full experiment (all data, 10 epochs)')
    group.add_argument('--sweep', action='store_true',
                       help='Run hyperparameter sweep')

    # Configuration
    parser.add_argument('--model', type=str, default='mlp',
                        choices=['mlp', 'conv'],
                        help='Model architecture')
    parser.add_argument('--epochs-a', type=int, default=None,
                        help='Epochs for Task A (MNIST)')
    parser.add_argument('--epochs-b', type=int, default=None,
                        help='Epochs for Task B (FashionMNIST)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--ewc-lambda', type=float, default=1000.0,
                        help='EWC regularization strength')
    parser.add_argument('--ricci-lambda', type=float, default=1.0,
                        help='Ricci regularization strength')
    parser.add_argument('--subset', type=int, default=None,
                        help='Use subset of data (number of samples)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cpu/cuda/mps)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Determine device
    if args.device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    print(f"Using device: {device}")

    # Set experiment parameters based on mode
    if args.quick:
        subset_size = 1000
        epochs_a = args.epochs_a or 3
        epochs_b = args.epochs_b or 3
        print("\nRunning QUICK TEST (subset=1000, epochs=3)")
    elif args.full:
        subset_size = None
        epochs_a = args.epochs_a or 10
        epochs_b = args.epochs_b or 10
        print("\nRunning FULL EXPERIMENT")
    else:
        subset_size = args.subset
        epochs_a = args.epochs_a or 5
        epochs_b = args.epochs_b or 5

    print(f"Model: {args.model}")
    print(f"Epochs: {epochs_a} (Task A), {epochs_b} (Task B)")
    print(f"EWC lambda: {args.ewc_lambda}")
    print(f"Ricci lambda: {args.ricci_lambda}")
    if subset_size:
        print(f"Using subset: {subset_size} samples")

    # Run experiment
    experiment = ContinualLearningExperiment(
        model_type=args.model,
        device=device,
        subset_size=subset_size
    )

    if args.sweep:
        results = experiment.hyperparameter_sweep(
            epochs_task_a=epochs_a,
            epochs_task_b=epochs_b
        )
    else:
        results = experiment.run_all_methods(
            epochs_task_a=epochs_a,
            epochs_task_b=epochs_b,
            lr=args.lr,
            ewc_lambda=args.ewc_lambda,
            ricci_lambda=args.ricci_lambda
        )

    # Print analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    if 'baseline' in results and 'ewc' in results and 'ricci_reg' in results:
        baseline = results['baseline']
        ewc = results['ewc']
        ricci = results['ricci_reg']

        print("\n** Key Findings **")

        # Forgetting comparison
        print(f"\n1. Catastrophic Forgetting:")
        print(f"   Baseline: {baseline['forgetting']:.1%} accuracy lost")
        print(f"   EWC:      {ewc['forgetting']:.1%} accuracy lost")
        print(f"   Ricci:    {ricci['forgetting']:.1%} accuracy lost")

        # Weight change comparison
        print(f"\n2. Weight Space Changes (from Task A):")
        print(f"   Baseline: {baseline['weight_change']:.2f}")
        print(f"   EWC:      {ewc['weight_change']:.2f}")
        print(f"   Ricci:    {ricci['weight_change']:.2f}")

        # The key prediction
        print(f"\n3. Ricci-REWA Prediction Test:")

        if ricci['mnist_after_b'] > ewc['mnist_after_b'] + 0.05:
            print("   ✓ Ricci-Reg SIGNIFICANTLY outperforms EWC")
            print(f"   MNIST retention: {ricci['mnist_after_b']:.1%} vs {ewc['mnist_after_b']:.1%}")
        elif ricci['mnist_after_b'] > ewc['mnist_after_b']:
            print("   ~ Ricci-Reg slightly outperforms EWC")
            print(f"   MNIST retention: {ricci['mnist_after_b']:.1%} vs {ewc['mnist_after_b']:.1%}")
        else:
            print("   ✗ EWC performs better than or equal to Ricci-Reg")

        # Weight change with preserved accuracy
        if ricci['weight_change'] > ewc['weight_change'] * 0.8 and \
           ricci['mnist_after_b'] > baseline['mnist_after_b'] + 0.1:
            print("\n   ✓ CRITICAL: Large weight changes with preserved accuracy!")
            print("   This supports the Ricci-REWA hypothesis:")
            print("   'Geometry, not weights, encodes task knowledge'")

        # Task B performance (no plasticity-stability tradeoff)
        print(f"\n4. Task B (FashionMNIST) Performance:")
        print(f"   Baseline: {baseline['fashion_after_b']:.1%}")
        print(f"   EWC:      {ewc['fashion_after_b']:.1%}")
        print(f"   Ricci:    {ricci['fashion_after_b']:.1%}")

        if ricci['fashion_after_b'] >= ewc['fashion_after_b'] - 0.02:
            print("   ✓ Ricci-Reg maintains Task B performance")
            print("   No plasticity-stability tradeoff!")

    return results


if __name__ == "__main__":
    main()
