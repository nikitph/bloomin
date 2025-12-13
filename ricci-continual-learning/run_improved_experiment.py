#!/usr/bin/env python3
"""
Improved Experiment: Testing Ricci-REWA with stronger geometric regularization.

This version:
1. Uses the improved CompositeGeometryRegularizer
2. Tests a range of regularization strengths
3. Provides detailed analysis of what's happening geometrically
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.models import SimpleMLP, ConvNet, get_weight_vector, compute_weight_distance
from src.continual_learning import BaselineCL, EWCCL
from src.continual_learning_v2 import ImprovedRicciRegCL


def get_datasets(data_root: str = './data', subset_size: Optional[int] = None):
    """Load MNIST and FashionMNIST."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_train = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_root, train=False, download=True, transform=transform)
    fashion_train = datasets.FashionMNIST(data_root, train=True, download=True, transform=transform)
    fashion_test = datasets.FashionMNIST(data_root, train=False, download=True, transform=transform)

    if subset_size:
        mnist_train = Subset(mnist_train, range(min(subset_size, len(mnist_train))))
        mnist_test = Subset(mnist_test, range(min(subset_size // 5, len(mnist_test))))
        fashion_train = Subset(fashion_train, range(min(subset_size, len(fashion_train))))
        fashion_test = Subset(fashion_test, range(min(subset_size // 5, len(fashion_test))))

    batch_size = 64

    mnist_loaders = {
        'train': DataLoader(mnist_train, batch_size=batch_size, shuffle=True),
        'test': DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
    }

    fashion_loaders = {
        'train': DataLoader(fashion_train, batch_size=batch_size, shuffle=True),
        'test': DataLoader(fashion_test, batch_size=batch_size, shuffle=False)
    }

    return mnist_loaders, fashion_loaders


def run_baseline(mnist_loaders, fashion_loaders, device, epochs_a=10, epochs_b=10, lr=1e-3):
    """Run baseline (no protection)."""
    print("\n" + "=" * 60)
    print("BASELINE (No Protection)")
    print("=" * 60)

    model = SimpleMLP()
    learner = BaselineCL(model, device=device, lr=lr)

    # Task A
    print("\nTraining on MNIST...")
    learner.train_task(mnist_loaders['train'], mnist_loaders['test'], epochs=epochs_a)
    mnist_after_a = learner.evaluate(mnist_loaders['test'])
    weights_after_a = get_weight_vector(model).clone()

    # Task B
    print("\nTraining on FashionMNIST...")
    learner.train_task(fashion_loaders['train'], fashion_loaders['test'], epochs=epochs_b)
    mnist_after_b = learner.evaluate(mnist_loaders['test'])
    fashion_after_b = learner.evaluate(fashion_loaders['test'])
    weights_after_b = get_weight_vector(model).clone()

    weight_change = torch.norm(weights_after_b - weights_after_a).item()

    return {
        'method': 'baseline',
        'mnist_after_a': mnist_after_a['accuracy'],
        'mnist_after_b': mnist_after_b['accuracy'],
        'fashion_after_b': fashion_after_b['accuracy'],
        'weight_change': weight_change,
        'forgetting': mnist_after_a['accuracy'] - mnist_after_b['accuracy']
    }


def run_ewc(mnist_loaders, fashion_loaders, device, epochs_a=10, epochs_b=10, lr=1e-3, ewc_lambda=5000):
    """Run EWC."""
    print("\n" + "=" * 60)
    print(f"EWC (λ={ewc_lambda})")
    print("=" * 60)

    model = SimpleMLP()
    learner = EWCCL(model, device=device, lr=lr, ewc_lambda=ewc_lambda)

    # Task A
    print("\nTraining on MNIST...")
    learner.train_task(mnist_loaders['train'], mnist_loaders['test'], epochs=epochs_a)
    mnist_after_a = learner.evaluate(mnist_loaders['test'])
    weights_after_a = get_weight_vector(model).clone()

    learner.after_task(mnist_loaders['train'], task_id=0)

    # Task B
    print("\nTraining on FashionMNIST with EWC...")
    learner.train_task(fashion_loaders['train'], fashion_loaders['test'], epochs=epochs_b)
    mnist_after_b = learner.evaluate(mnist_loaders['test'])
    fashion_after_b = learner.evaluate(fashion_loaders['test'])
    weights_after_b = get_weight_vector(model).clone()

    weight_change = torch.norm(weights_after_b - weights_after_a).item()

    return {
        'method': 'ewc',
        'ewc_lambda': ewc_lambda,
        'mnist_after_a': mnist_after_a['accuracy'],
        'mnist_after_b': mnist_after_b['accuracy'],
        'fashion_after_b': fashion_after_b['accuracy'],
        'weight_change': weight_change,
        'forgetting': mnist_after_a['accuracy'] - mnist_after_b['accuracy']
    }


def run_ricci_improved(mnist_loaders, fashion_loaders, device, epochs_a=10, epochs_b=10,
                       lr=1e-3, ricci_lambda=10.0, k_neighbors=15):
    """Run improved Ricci regularization."""
    print("\n" + "=" * 60)
    print(f"IMPROVED RICCI REG (λ={ricci_lambda}, k={k_neighbors})")
    print("=" * 60)

    model = SimpleMLP()
    learner = ImprovedRicciRegCL(
        model,
        device=device,
        lr=lr,
        ricci_lambda=ricci_lambda,
        k_neighbors=k_neighbors,
        n_samples=500
    )

    # Task A
    print("\nTraining on MNIST...")
    learner.train_task(mnist_loaders['train'], mnist_loaders['test'], epochs=epochs_a)
    mnist_after_a = learner.evaluate(mnist_loaders['test'])
    weights_after_a = get_weight_vector(model).clone()

    learner.after_task(mnist_loaders['train'], task_id=0)

    # Task B
    print("\nTraining on FashionMNIST with Ricci Regularization...")
    learner.train_task(fashion_loaders['train'], fashion_loaders['test'], epochs=epochs_b)
    mnist_after_b = learner.evaluate(mnist_loaders['test'])
    fashion_after_b = learner.evaluate(fashion_loaders['test'])
    weights_after_b = get_weight_vector(model).clone()

    weight_change = torch.norm(weights_after_b - weights_after_a).item()

    return {
        'method': 'ricci_improved',
        'ricci_lambda': ricci_lambda,
        'k_neighbors': k_neighbors,
        'mnist_after_a': mnist_after_a['accuracy'],
        'mnist_after_b': mnist_after_b['accuracy'],
        'fashion_after_b': fashion_after_b['accuracy'],
        'weight_change': weight_change,
        'forgetting': mnist_after_a['accuracy'] - mnist_after_b['accuracy']
    }


def lambda_sweep(mnist_loaders, fashion_loaders, device, epochs_a=5, epochs_b=5):
    """Sweep over regularization strengths."""
    print("\n" + "=" * 60)
    print("LAMBDA SWEEP")
    print("=" * 60)

    # EWC sweep
    ewc_results = []
    for ewc_lambda in [1000, 5000, 10000, 20000, 50000]:
        print(f"\nTesting EWC λ={ewc_lambda}...")
        r = run_ewc(mnist_loaders, fashion_loaders, device, epochs_a, epochs_b, ewc_lambda=ewc_lambda)
        ewc_results.append(r)
        print(f"  MNIST retention: {r['mnist_after_b']:.1%}, Fashion: {r['fashion_after_b']:.1%}")

    # Ricci sweep
    ricci_results = []
    for ricci_lambda in [1.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        print(f"\nTesting Ricci λ={ricci_lambda}...")
        r = run_ricci_improved(mnist_loaders, fashion_loaders, device, epochs_a, epochs_b, ricci_lambda=ricci_lambda)
        ricci_results.append(r)
        print(f"  MNIST retention: {r['mnist_after_b']:.1%}, Fashion: {r['fashion_after_b']:.1%}")

    return {'ewc': ewc_results, 'ricci': ricci_results}


def print_summary(results: Dict):
    """Print summary of results."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'Method':<25} {'MNIST↓':<12} {'Fashion':<12} {'Weight Δ':<12} {'Forget':<12}")
    print("-" * 70)

    for name, r in results.items():
        if isinstance(r, dict) and 'mnist_after_b' in r:
            print(
                f"{name:<25} "
                f"{r['mnist_after_b']:.1%}        "
                f"{r['fashion_after_b']:.1%}        "
                f"{r['weight_change']:.2f}         "
                f"{r['forgetting']:.1%}"
            )


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Improved Ricci-REWA Experiment')
    parser.add_argument('--quick', action='store_true', help='Quick test')
    parser.add_argument('--sweep', action='store_true', help='Lambda sweep')
    parser.add_argument('--epochs-a', type=int, default=None)
    parser.add_argument('--epochs-b', type=int, default=None)
    parser.add_argument('--ricci-lambda', type=float, default=10.0)
    parser.add_argument('--ewc-lambda', type=float, default=5000.0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--subset', type=int, default=None)

    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Using device: {device}")

    # Parameters
    if args.quick:
        subset = 2000
        epochs_a = args.epochs_a or 5
        epochs_b = args.epochs_b or 5
        print("Running QUICK test")
    else:
        subset = args.subset
        epochs_a = args.epochs_a or 10
        epochs_b = args.epochs_b or 10

    # Load data
    print("Loading datasets...")
    mnist_loaders, fashion_loaders = get_datasets('./data', subset)

    if args.sweep:
        # Run lambda sweep
        sweep_results = lambda_sweep(
            mnist_loaders, fashion_loaders, device,
            epochs_a=epochs_a, epochs_b=epochs_b
        )

        print("\n" + "=" * 60)
        print("SWEEP RESULTS")
        print("=" * 60)

        print("\nEWC:")
        best_ewc = max(sweep_results['ewc'], key=lambda x: x['mnist_after_b'])
        for r in sweep_results['ewc']:
            marker = " <-- BEST" if r == best_ewc else ""
            print(f"  λ={r['ewc_lambda']:>6}: MNIST {r['mnist_after_b']:.1%}, Fashion {r['fashion_after_b']:.1%}{marker}")

        print("\nRicci:")
        best_ricci = max(sweep_results['ricci'], key=lambda x: x['mnist_after_b'])
        for r in sweep_results['ricci']:
            marker = " <-- BEST" if r == best_ricci else ""
            print(f"  λ={r['ricci_lambda']:>6}: MNIST {r['mnist_after_b']:.1%}, Fashion {r['fashion_after_b']:.1%}{marker}")

        print("\n" + "=" * 60)
        print("COMPARISON")
        print("=" * 60)
        print(f"\nBest EWC: λ={best_ewc['ewc_lambda']}, MNIST={best_ewc['mnist_after_b']:.1%}")
        print(f"Best Ricci: λ={best_ricci['ricci_lambda']}, MNIST={best_ricci['mnist_after_b']:.1%}")

        if best_ricci['mnist_after_b'] > best_ewc['mnist_after_b']:
            print("\n✓ Ricci-Reg outperforms EWC!")
            diff = best_ricci['mnist_after_b'] - best_ewc['mnist_after_b']
            print(f"  Improvement: {diff:.1%}")
        else:
            print("\n✗ EWC performs better or equal")

    else:
        # Run all methods with current settings
        results = {}

        results['baseline'] = run_baseline(
            mnist_loaders, fashion_loaders, device, epochs_a, epochs_b
        )

        results['ewc'] = run_ewc(
            mnist_loaders, fashion_loaders, device, epochs_a, epochs_b,
            ewc_lambda=args.ewc_lambda
        )

        results['ricci_improved'] = run_ricci_improved(
            mnist_loaders, fashion_loaders, device, epochs_a, epochs_b,
            ricci_lambda=args.ricci_lambda
        )

        print_summary(results)

        # Analysis
        print("\n" + "=" * 60)
        print("ANALYSIS")
        print("=" * 60)

        baseline = results['baseline']
        ewc = results['ewc']
        ricci = results['ricci_improved']

        print(f"\n1. Catastrophic Forgetting:")
        print(f"   Baseline: {baseline['forgetting']:.1%} lost")
        print(f"   EWC:      {ewc['forgetting']:.1%} lost")
        print(f"   Ricci:    {ricci['forgetting']:.1%} lost")

        print(f"\n2. Weight Changes:")
        print(f"   Baseline: {baseline['weight_change']:.2f}")
        print(f"   EWC:      {ewc['weight_change']:.2f}")
        print(f"   Ricci:    {ricci['weight_change']:.2f}")

        # Key test
        print(f"\n3. Ricci-REWA Hypothesis Test:")
        if ricci['mnist_after_b'] > ewc['mnist_after_b'] + 0.05:
            print("   ✓ CONFIRMED: Ricci-Reg significantly outperforms EWC")
        elif ricci['mnist_after_b'] > ewc['mnist_after_b']:
            print("   ~ Partial support: Ricci-Reg slightly better")
        else:
            print("   ✗ Not confirmed: EWC performs better or equal")

        # Critical test: large weight change + preserved accuracy
        if ricci['weight_change'] > ewc['weight_change'] * 0.8:
            if ricci['mnist_after_b'] > baseline['mnist_after_b'] + 0.2:
                print(f"\n   ✓ CRITICAL: Weights changed freely ({ricci['weight_change']:.2f})")
                print(f"              but MNIST accuracy preserved ({ricci['mnist_after_b']:.1%} vs baseline {baseline['mnist_after_b']:.1%})")
                print("              This supports geometry > weights hypothesis!")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"./results/improved_results_{timestamp}.json"
        os.makedirs('./results', exist_ok=True)

        serializable = {
            k: {kk: float(vv) if isinstance(vv, (float, np.floating)) else vv
                for kk, vv in v.items()}
            for k, v in results.items()
        }

        with open(results_file, 'w') as f:
            json.dump(serializable, f, indent=2)

        print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
