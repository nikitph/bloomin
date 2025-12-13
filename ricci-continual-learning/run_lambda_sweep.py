#!/usr/bin/env python3
"""Quick lambda sweep for class-conditional methods."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

from src.models import SimpleMLP, get_weight_vector
from src.continual_learning import BaselineCL, EWCCL
from src.continual_learning_class_conditional import CentroidRicciCL, ClassConditionalRicciCL


def get_datasets(data_root='./data', subset_size=2000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_train = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(data_root, train=False, download=True, transform=transform)
    fashion_train = datasets.FashionMNIST(data_root, train=True, download=True, transform=transform)
    fashion_test = datasets.FashionMNIST(data_root, train=False, download=True, transform=transform)

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


def run_single(learner, loaders, epochs=5):
    """Run a single experiment."""
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=epochs, verbose=False)
    mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
    weights_a = get_weight_vector(learner.model).clone()

    if hasattr(learner, 'after_task'):
        learner.after_task(loaders['mnist_train'], task_id=0)

    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=epochs, verbose=False)
    mnist_b = learner.evaluate(loaders['mnist_test'])['accuracy']
    fashion_b = learner.evaluate(loaders['fashion_test'])['accuracy']
    weights_b = get_weight_vector(learner.model).clone()

    return {
        'mnist_a': mnist_a,
        'mnist_b': mnist_b,
        'fashion_b': fashion_b,
        'weight_change': torch.norm(weights_b - weights_a).item(),
        'forgetting': mnist_a - mnist_b
    }


def main():
    device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    loaders = get_datasets('./data', 2000)

    # Run baseline and EWC first
    print("Running baseline...")
    model = SimpleMLP()
    learner = BaselineCL(model, device=device)
    baseline = run_single(learner, loaders)
    print(f"Baseline: MNIST {baseline['mnist_b']:.1%}, Fashion {baseline['fashion_b']:.1%}")

    print("\nRunning EWC (λ=5000)...")
    model = SimpleMLP()
    learner = EWCCL(model, device=device, ewc_lambda=5000)
    ewc = run_single(learner, loaders)
    print(f"EWC: MNIST {ewc['mnist_b']:.1%}, Fashion {ewc['fashion_b']:.1%}")

    # Sweep Centroid lambda
    print("\n" + "=" * 60)
    print("CENTROID RICCI LAMBDA SWEEP")
    print("=" * 60)

    centroid_results = []
    for lam in [25, 50, 75, 100, 150, 200]:
        print(f"\nCentroid λ={lam}...", end=" ")
        model = SimpleMLP()
        learner = CentroidRicciCL(model, device=device, ricci_lambda=lam)
        r = run_single(learner, loaders)
        r['lambda'] = lam
        centroid_results.append(r)
        print(f"MNIST {r['mnist_b']:.1%}, Fashion {r['fashion_b']:.1%}, Weight Δ {r['weight_change']:.2f}")

    # Sweep Class-Conditional lambda
    print("\n" + "=" * 60)
    print("CLASS-CONDITIONAL RICCI LAMBDA SWEEP")
    print("=" * 60)

    cc_results = []
    for lam in [25, 50, 75, 100, 150, 200]:
        print(f"\nClass-Conditional λ={lam}...", end=" ")
        model = SimpleMLP()
        learner = ClassConditionalRicciCL(model, device=device, ricci_lambda=lam)
        r = run_single(learner, loaders)
        r['lambda'] = lam
        cc_results.append(r)
        print(f"MNIST {r['mnist_b']:.1%}, Fashion {r['fashion_b']:.1%}, Weight Δ {r['weight_change']:.2f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nBaseline: MNIST {baseline['mnist_b']:.1%}, Fashion {baseline['fashion_b']:.1%}")
    print(f"EWC:      MNIST {ewc['mnist_b']:.1%}, Fashion {ewc['fashion_b']:.1%}")

    print("\nCentroid Ricci:")
    for r in centroid_results:
        marker = " <-- BEST" if r['mnist_b'] == max(x['mnist_b'] for x in centroid_results) else ""
        print(f"  λ={r['lambda']:>3}: MNIST {r['mnist_b']:.1%}, Fashion {r['fashion_b']:.1%}{marker}")

    print("\nClass-Conditional Ricci:")
    for r in cc_results:
        marker = " <-- BEST" if r['mnist_b'] == max(x['mnist_b'] for x in cc_results) else ""
        print(f"  λ={r['lambda']:>3}: MNIST {r['mnist_b']:.1%}, Fashion {r['fashion_b']:.1%}{marker}")

    # Best results
    best_centroid = max(centroid_results, key=lambda x: x['mnist_b'])
    best_cc = max(cc_results, key=lambda x: x['mnist_b'])

    print("\n" + "=" * 60)
    print("BEST RESULTS")
    print("=" * 60)
    print(f"EWC:           MNIST {ewc['mnist_b']:.1%}, Fashion {ewc['fashion_b']:.1%}")
    print(f"Best Centroid: MNIST {best_centroid['mnist_b']:.1%}, Fashion {best_centroid['fashion_b']:.1%} (λ={best_centroid['lambda']})")
    print(f"Best CC:       MNIST {best_cc['mnist_b']:.1%}, Fashion {best_cc['fashion_b']:.1%} (λ={best_cc['lambda']})")

    overall_best = max([best_centroid, best_cc], key=lambda x: x['mnist_b'])

    if overall_best['mnist_b'] > ewc['mnist_b']:
        improvement = overall_best['mnist_b'] - ewc['mnist_b']
        print(f"\n✓ RICCI BEATS EWC by +{improvement:.1%}!")
    else:
        print(f"\n✗ EWC still wins")


if __name__ == "__main__":
    main()
