#!/usr/bin/env python3
"""
Experiment: Class-Conditional Curvature Preservation

Tests whether preserving the geometric structure WITHIN each class
is more effective than global curvature preservation.
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
from src.continual_learning_class_conditional import (
    ClassConditionalRicciCL,
    CentroidRicciCL,
    PrototypeRicciCL
)


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


def run_method(method_name, learner, loaders, epochs_a, epochs_b):
    """Run a single method and return results."""
    print(f"\n{'=' * 60}")
    print(f"{method_name}")
    print('=' * 60)

    # Task A: MNIST
    print("\nTraining on MNIST...")
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=epochs_a)
    mnist_after_a = learner.evaluate(loaders['mnist_test'])
    weights_after_a = get_weight_vector(learner.model).clone()

    # Store reference (if method has after_task)
    if hasattr(learner, 'after_task'):
        learner.after_task(loaders['mnist_train'], task_id=0)

    # Task B: FashionMNIST
    print("\nTraining on FashionMNIST...")
    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=epochs_b)
    mnist_after_b = learner.evaluate(loaders['mnist_test'])
    fashion_after_b = learner.evaluate(loaders['fashion_test'])
    weights_after_b = get_weight_vector(learner.model).clone()

    weight_change = torch.norm(weights_after_b - weights_after_a).item()

    print(f"\nResults:")
    print(f"  MNIST: {mnist_after_a['accuracy']:.1%} → {mnist_after_b['accuracy']:.1%}")
    print(f"  Fashion: {fashion_after_b['accuracy']:.1%}")
    print(f"  Weight change: {weight_change:.2f}")

    return {
        'method': method_name,
        'mnist_a': mnist_after_a['accuracy'],
        'mnist_b': mnist_after_b['accuracy'],
        'fashion_b': fashion_after_b['accuracy'],
        'weight_change': weight_change,
        'forgetting': mnist_after_a['accuracy'] - mnist_after_b['accuracy']
    }


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs-a', type=int, default=5)
    parser.add_argument('--epochs-b', type=int, default=5)
    parser.add_argument('--ricci-lambda', type=float, default=10.0)
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
    print(f"Subset: {args.subset}")

    loaders = get_datasets('./data', args.subset)

    results = {}

    # Baseline
    model = SimpleMLP()
    learner = BaselineCL(model, device=device)
    results['baseline'] = run_method('BASELINE', learner, loaders, args.epochs_a, args.epochs_b)

    # EWC
    model = SimpleMLP()
    learner = EWCCL(model, device=device, ewc_lambda=args.ewc_lambda)
    results['ewc'] = run_method(f'EWC (λ={args.ewc_lambda})', learner, loaders, args.epochs_a, args.epochs_b)

    # Class-Conditional Ricci
    model = SimpleMLP()
    learner = ClassConditionalRicciCL(model, device=device, ricci_lambda=args.ricci_lambda)
    results['class_conditional'] = run_method(
        f'CLASS-CONDITIONAL RICCI (λ={args.ricci_lambda})',
        learner, loaders, args.epochs_a, args.epochs_b
    )

    # Centroid Ricci
    model = SimpleMLP()
    learner = CentroidRicciCL(model, device=device, ricci_lambda=args.ricci_lambda)
    results['centroid'] = run_method(
        f'CENTROID RICCI (λ={args.ricci_lambda})',
        learner, loaders, args.epochs_a, args.epochs_b
    )

    # Prototype Ricci
    model = SimpleMLP()
    learner = PrototypeRicciCL(model, device=device, ricci_lambda=args.ricci_lambda)
    results['prototype'] = run_method(
        f'PROTOTYPE RICCI (λ={args.ricci_lambda})',
        learner, loaders, args.epochs_a, args.epochs_b
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<30} {'MNIST↓':<12} {'Fashion':<12} {'Weight Δ':<12} {'Forget':<12}")
    print("-" * 80)

    for name, r in results.items():
        print(f"{name:<30} {r['mnist_b']:.1%}        {r['fashion_b']:.1%}        {r['weight_change']:.2f}         {r['forgetting']:.1%}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    baseline = results['baseline']
    ewc = results['ewc']

    # Find best Ricci method
    ricci_methods = ['class_conditional', 'centroid', 'prototype']
    best_ricci_name = max(ricci_methods, key=lambda x: results[x]['mnist_b'])
    best_ricci = results[best_ricci_name]

    print(f"\nBaseline MNIST retention: {baseline['mnist_b']:.1%}")
    print(f"EWC MNIST retention: {ewc['mnist_b']:.1%}")
    print(f"Best Ricci ({best_ricci_name}): {best_ricci['mnist_b']:.1%}")

    # Comparison
    print("\n--- Comparison ---")

    if best_ricci['mnist_b'] > ewc['mnist_b'] + 0.05:
        print(f"✓ CLASS-CONDITIONAL RICCI WINS!")
        print(f"  {best_ricci_name} retains {best_ricci['mnist_b']:.1%} vs EWC {ewc['mnist_b']:.1%}")
        improvement = best_ricci['mnist_b'] - ewc['mnist_b']
        print(f"  Improvement: +{improvement:.1%}")
    elif best_ricci['mnist_b'] > ewc['mnist_b']:
        print(f"~ Ricci slightly better: {best_ricci['mnist_b']:.1%} vs {ewc['mnist_b']:.1%}")
    else:
        print(f"✗ EWC still better: {ewc['mnist_b']:.1%} vs Ricci {best_ricci['mnist_b']:.1%}")

    # Weight change analysis
    print("\n--- Weight Changes ---")
    print(f"EWC weight change: {ewc['weight_change']:.2f}")
    print(f"Best Ricci weight change: {best_ricci['weight_change']:.2f}")

    if best_ricci['weight_change'] > ewc['weight_change'] * 1.5:
        if best_ricci['mnist_b'] > baseline['mnist_b'] + 0.15:
            print(f"\n✓ KEY FINDING: Ricci allows {best_ricci['weight_change']/ewc['weight_change']:.1f}x more weight change")
            print(f"   while preserving {best_ricci['mnist_b']:.1%} MNIST accuracy (baseline: {baseline['mnist_b']:.1%})")

    # Fashion accuracy (plasticity check)
    print("\n--- Plasticity Check ---")
    print(f"Fashion accuracies:")
    print(f"  Baseline: {baseline['fashion_b']:.1%}")
    print(f"  EWC: {ewc['fashion_b']:.1%}")
    print(f"  Best Ricci: {best_ricci['fashion_b']:.1%}")

    if best_ricci['fashion_b'] >= ewc['fashion_b'] - 0.03:
        print("✓ Ricci maintains plasticity (good Fashion performance)")

    # Save results
    import json
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"./results/class_conditional_{timestamp}.json"
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
