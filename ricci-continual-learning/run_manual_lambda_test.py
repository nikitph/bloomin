#!/usr/bin/env python3
"""
Direct Test: Manual λ based on known task similarity.

We KNOW:
- MNIST→Fashion: Different domains, LOW interference expected
- MNIST→KMNIST: Similar domains (both digits), HIGH interference expected

Test whether setting λ appropriately helps.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.models import SimpleMLP
from src.continual_learning import BaselineCL, EWCCL
from src.continual_learning_class_conditional import CentroidRicciCL


def get_datasets(data_root='./data', subset_size=2000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    loaders = {}
    for name, cls in [
        ('mnist', datasets.MNIST),
        ('fashion', datasets.FashionMNIST),
        ('kmnist', datasets.KMNIST),
    ]:
        train = cls(data_root, train=True, download=True, transform=transform)
        test = cls(data_root, train=False, download=True, transform=transform)

        if subset_size:
            train = Subset(train, range(min(subset_size, len(train))))
            test = Subset(test, range(min(subset_size // 5, len(test))))
        loaders[f'{name}_train'] = DataLoader(train, batch_size=64, shuffle=True)
        loaders[f'{name}_test'] = DataLoader(test, batch_size=64, shuffle=False)

    return loaders


def run_2task_test(device, loaders, task_b_name, task_b_train, task_b_test, lambdas_to_test):
    """Test different λ values for a 2-task scenario."""
    results = {}

    for lam in lambdas_to_test:
        model = SimpleMLP()
        learner = CentroidRicciCL(model, device=device, ricci_lambda=lam)

        # Task A: MNIST
        learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
        mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
        learner.after_task(loaders['mnist_train'], task_id=0)

        # Task B
        learner.train_task(task_b_train, task_b_test, epochs=5, verbose=False)
        mnist_b = learner.evaluate(loaders['mnist_test'])['accuracy']
        task_b_acc = learner.evaluate(task_b_test)['accuracy']

        results[lam] = {
            'mnist_a': mnist_a,
            'mnist_b': mnist_b,
            'task_b': task_b_acc,
            'forgetting': mnist_a - mnist_b
        }

        print(f"  λ={lam:>3}: MNIST {mnist_a:.1%}→{mnist_b:.1%}, {task_b_name} {task_b_acc:.1%}, "
              f"forget {results[lam]['forgetting']:.1%}")

    return results


def main():
    device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    loaders = get_datasets('./data', subset_size=2000)

    print("\n" + "="*70)
    print("MANUAL λ TEST: Known Task Similarity")
    print("="*70)

    # ========================================
    # MNIST → Fashion (LOW similarity expected)
    # ========================================
    print("\n" + "-"*70)
    print("MNIST → Fashion (Different domains - expect LOW λ to be optimal)")
    print("-"*70)
    print("\nExpectation: Lower λ should work well (less interference)")

    fashion_results = run_2task_test(
        device, loaders,
        "Fashion",
        loaders['fashion_train'],
        loaders['fashion_test'],
        [25, 50, 75, 100, 150]
    )

    # Find best λ for Fashion task
    best_fashion_lam = max(fashion_results.items(), key=lambda x: x[1]['mnist_b'])[0]
    print(f"\nBest λ for MNIST→Fashion: {best_fashion_lam}")

    # ========================================
    # MNIST → KMNIST (HIGH similarity expected)
    # ========================================
    print("\n" + "-"*70)
    print("MNIST → KMNIST (Similar domains - expect HIGH λ to be optimal)")
    print("-"*70)
    print("\nExpectation: Higher λ should work better (more interference)")

    kmnist_results = run_2task_test(
        device, loaders,
        "KMNIST",
        loaders['kmnist_train'],
        loaders['kmnist_test'],
        [25, 50, 75, 100, 150]
    )

    # Find best λ for KMNIST task
    best_kmnist_lam = max(kmnist_results.items(), key=lambda x: x[1]['mnist_b'])[0]
    print(f"\nBest λ for MNIST→KMNIST: {best_kmnist_lam}")

    # ========================================
    # Analysis
    # ========================================
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    print("\nMNIST → Fashion results:")
    print(f"{'λ':<6} {'MNIST Ret':<12} {'Fashion':<12} {'Forgetting':<12}")
    print("-"*45)
    for lam, r in sorted(fashion_results.items()):
        marker = " ← best" if lam == best_fashion_lam else ""
        print(f"{lam:<6} {r['mnist_b']:.1%}        {r['task_b']:.1%}        {r['forgetting']:.1%}{marker}")

    print("\nMNIST → KMNIST results:")
    print(f"{'λ':<6} {'MNIST Ret':<12} {'KMNIST':<12} {'Forgetting':<12}")
    print("-"*45)
    for lam, r in sorted(kmnist_results.items()):
        marker = " ← best" if lam == best_kmnist_lam else ""
        print(f"{lam:<6} {r['mnist_b']:.1%}        {r['task_b']:.1%}        {r['forgetting']:.1%}{marker}")

    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if best_kmnist_lam > best_fashion_lam:
        print(f"\n✓ HYPOTHESIS CONFIRMED!")
        print(f"  Similar tasks (MNIST→KMNIST) need HIGHER λ ({best_kmnist_lam})")
        print(f"  Different tasks (MNIST→Fashion) work with LOWER λ ({best_fashion_lam})")
        print(f"\n  This validates the predictive λ approach:")
        print(f"  → High similarity = high interference = need more regularization")
    elif best_kmnist_lam == best_fashion_lam:
        print(f"\n~ Same optimal λ ({best_kmnist_lam}) for both")
        print(f"  Task similarity might not matter much for optimal λ")
    else:
        print(f"\n✗ UNEXPECTED: Fashion needs higher λ ({best_fashion_lam}) than KMNIST ({best_kmnist_lam})")

    # Compare best adaptive vs fixed
    print("\n" + "-"*50)
    print("Comparison: Adaptive vs Fixed λ=50")
    print("-"*50)

    fixed_fashion = fashion_results[50]['mnist_b']
    best_fashion = fashion_results[best_fashion_lam]['mnist_b']
    print(f"Fashion: Fixed λ=50 gives {fixed_fashion:.1%}, Best λ={best_fashion_lam} gives {best_fashion:.1%}")
    if best_fashion > fixed_fashion:
        print(f"  → Adaptive would gain +{(best_fashion - fixed_fashion):.1%}")

    fixed_kmnist = kmnist_results[50]['mnist_b']
    best_kmnist = kmnist_results[best_kmnist_lam]['mnist_b']
    print(f"KMNIST:  Fixed λ=50 gives {fixed_kmnist:.1%}, Best λ={best_kmnist_lam} gives {best_kmnist:.1%}")
    if best_kmnist > fixed_kmnist:
        print(f"  → Adaptive would gain +{(best_kmnist - fixed_kmnist):.1%}")


if __name__ == "__main__":
    main()
