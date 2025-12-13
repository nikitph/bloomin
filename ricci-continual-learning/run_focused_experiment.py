#!/usr/bin/env python3
"""
Experiment with focused curvature preservation (scale-invariant).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from datetime import datetime

from src.models import SimpleMLP, get_weight_vector
from src.continual_learning import BaselineCL, EWCCL
from src.continual_learning_v3 import FocusedRicciCL, AngleOnlyRicciCL


def get_datasets(data_root='./data', subset_size=None):
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

    return {
        'mnist_train': DataLoader(mnist_train, batch_size=batch_size, shuffle=True),
        'mnist_test': DataLoader(mnist_test, batch_size=batch_size, shuffle=False),
        'fashion_train': DataLoader(fashion_train, batch_size=batch_size, shuffle=True),
        'fashion_test': DataLoader(fashion_test, batch_size=batch_size, shuffle=False)
    }


def run_experiment(device, loaders, epochs_a=5, epochs_b=5, ricci_lambda=50.0, ewc_lambda=5000.0):
    results = {}

    # Baseline
    print("\n" + "=" * 60)
    print("BASELINE")
    print("=" * 60)

    model = SimpleMLP()
    learner = BaselineCL(model, device=device)

    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=epochs_a)
    mnist_after_a = learner.evaluate(loaders['mnist_test'])
    weights_after_a = get_weight_vector(model).clone()

    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=epochs_b)
    mnist_after_b = learner.evaluate(loaders['mnist_test'])
    fashion_after_b = learner.evaluate(loaders['fashion_test'])
    weights_after_b = get_weight_vector(model).clone()

    results['baseline'] = {
        'mnist_a': mnist_after_a['accuracy'],
        'mnist_b': mnist_after_b['accuracy'],
        'fashion_b': fashion_after_b['accuracy'],
        'weight_change': torch.norm(weights_after_b - weights_after_a).item(),
        'forgetting': mnist_after_a['accuracy'] - mnist_after_b['accuracy']
    }
    print(f"MNIST: {results['baseline']['mnist_a']:.1%} → {results['baseline']['mnist_b']:.1%}")
    print(f"Fashion: {results['baseline']['fashion_b']:.1%}")

    # EWC
    print("\n" + "=" * 60)
    print(f"EWC (λ={ewc_lambda})")
    print("=" * 60)

    model = SimpleMLP()
    learner = EWCCL(model, device=device, ewc_lambda=ewc_lambda)

    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=epochs_a)
    mnist_after_a = learner.evaluate(loaders['mnist_test'])
    weights_after_a = get_weight_vector(model).clone()

    learner.after_task(loaders['mnist_train'], task_id=0)

    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=epochs_b)
    mnist_after_b = learner.evaluate(loaders['mnist_test'])
    fashion_after_b = learner.evaluate(loaders['fashion_test'])
    weights_after_b = get_weight_vector(model).clone()

    results['ewc'] = {
        'mnist_a': mnist_after_a['accuracy'],
        'mnist_b': mnist_after_b['accuracy'],
        'fashion_b': fashion_after_b['accuracy'],
        'weight_change': torch.norm(weights_after_b - weights_after_a).item(),
        'forgetting': mnist_after_a['accuracy'] - mnist_after_b['accuracy']
    }
    print(f"MNIST: {results['ewc']['mnist_a']:.1%} → {results['ewc']['mnist_b']:.1%}")
    print(f"Fashion: {results['ewc']['fashion_b']:.1%}")

    # Focused Ricci
    print("\n" + "=" * 60)
    print(f"FOCUSED RICCI (λ={ricci_lambda})")
    print("=" * 60)

    model = SimpleMLP()
    learner = FocusedRicciCL(model, device=device, ricci_lambda=ricci_lambda)

    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=epochs_a)
    mnist_after_a = learner.evaluate(loaders['mnist_test'])
    weights_after_a = get_weight_vector(model).clone()

    learner.after_task(loaders['mnist_train'], task_id=0)

    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=epochs_b)
    mnist_after_b = learner.evaluate(loaders['mnist_test'])
    fashion_after_b = learner.evaluate(loaders['fashion_test'])
    weights_after_b = get_weight_vector(model).clone()

    results['focused_ricci'] = {
        'mnist_a': mnist_after_a['accuracy'],
        'mnist_b': mnist_after_b['accuracy'],
        'fashion_b': fashion_after_b['accuracy'],
        'weight_change': torch.norm(weights_after_b - weights_after_a).item(),
        'forgetting': mnist_after_a['accuracy'] - mnist_after_b['accuracy']
    }
    print(f"MNIST: {results['focused_ricci']['mnist_a']:.1%} → {results['focused_ricci']['mnist_b']:.1%}")
    print(f"Fashion: {results['focused_ricci']['fashion_b']:.1%}")

    # Angle-only Ricci
    print("\n" + "=" * 60)
    print(f"ANGLE-ONLY RICCI (λ={ricci_lambda})")
    print("=" * 60)

    model = SimpleMLP()
    learner = AngleOnlyRicciCL(model, device=device, ricci_lambda=ricci_lambda)

    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=epochs_a)
    mnist_after_a = learner.evaluate(loaders['mnist_test'])
    weights_after_a = get_weight_vector(model).clone()

    learner.after_task(loaders['mnist_train'], task_id=0)

    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=epochs_b)
    mnist_after_b = learner.evaluate(loaders['mnist_test'])
    fashion_after_b = learner.evaluate(loaders['fashion_test'])
    weights_after_b = get_weight_vector(model).clone()

    results['angle_ricci'] = {
        'mnist_a': mnist_after_a['accuracy'],
        'mnist_b': mnist_after_b['accuracy'],
        'fashion_b': fashion_after_b['accuracy'],
        'weight_change': torch.norm(weights_after_b - weights_after_a).item(),
        'forgetting': mnist_after_a['accuracy'] - mnist_after_b['accuracy']
    }
    print(f"MNIST: {results['angle_ricci']['mnist_a']:.1%} → {results['angle_ricci']['mnist_b']:.1%}")
    print(f"Fashion: {results['angle_ricci']['fashion_b']:.1%}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs-a', type=int, default=5)
    parser.add_argument('--epochs-b', type=int, default=5)
    parser.add_argument('--ricci-lambda', type=float, default=50.0)
    parser.add_argument('--ewc-lambda', type=float, default=5000.0)
    parser.add_argument('--subset', type=int, default=2000)
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    print(f"Device: {device}")
    print(f"Epochs: {args.epochs_a}/{args.epochs_b}")
    print(f"Ricci λ: {args.ricci_lambda}")
    print(f"EWC λ: {args.ewc_lambda}")

    loaders = get_datasets('./data', args.subset)

    results = run_experiment(
        device, loaders,
        epochs_a=args.epochs_a,
        epochs_b=args.epochs_b,
        ricci_lambda=args.ricci_lambda,
        ewc_lambda=args.ewc_lambda
    )

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Method':<20} {'MNIST↓':<12} {'Fashion':<12} {'Weight Δ':<12} {'Forget':<12}")
    print("-" * 70)

    for method, r in results.items():
        print(f"{method:<20} {r['mnist_b']:.1%}        {r['fashion_b']:.1%}        {r['weight_change']:.2f}         {r['forgetting']:.1%}")

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    best_ricci = max([results.get('focused_ricci', {'mnist_b': 0}),
                      results.get('angle_ricci', {'mnist_b': 0})],
                     key=lambda x: x.get('mnist_b', 0))
    best_ricci_name = 'focused_ricci' if best_ricci == results.get('focused_ricci') else 'angle_ricci'

    print(f"\nBest Ricci method: {best_ricci_name}")
    print(f"  MNIST retention: {best_ricci['mnist_b']:.1%}")
    print(f"  Fashion accuracy: {best_ricci['fashion_b']:.1%}")
    print(f"  Weight change: {best_ricci['weight_change']:.2f}")

    ewc = results['ewc']
    print(f"\nEWC:")
    print(f"  MNIST retention: {ewc['mnist_b']:.1%}")
    print(f"  Fashion accuracy: {ewc['fashion_b']:.1%}")
    print(f"  Weight change: {ewc['weight_change']:.2f}")

    if best_ricci['mnist_b'] > ewc['mnist_b'] + 0.05:
        print("\n✓ RICCI-REWA CONFIRMED: Curvature preservation outperforms weight preservation!")
        print(f"  Improvement: {best_ricci['mnist_b'] - ewc['mnist_b']:.1%}")
    elif best_ricci['mnist_b'] > ewc['mnist_b']:
        print("\n~ Partial support: Ricci slightly better")
    else:
        print("\n✗ EWC performs better or equal")

    # The critical test: large weight change with preserved performance
    if best_ricci['weight_change'] > ewc['weight_change'] * 1.2:
        if best_ricci['mnist_b'] > results['baseline']['mnist_b'] + 0.1:
            print(f"\n✓ CRITICAL FINDING:")
            print(f"  Ricci allowed larger weight changes ({best_ricci['weight_change']:.2f} vs {ewc['weight_change']:.2f})")
            print(f"  while preserving more MNIST accuracy ({best_ricci['mnist_b']:.1%} vs baseline {results['baseline']['mnist_b']:.1%})")
            print("  This supports the geometry > weights hypothesis!")


if __name__ == "__main__":
    main()
