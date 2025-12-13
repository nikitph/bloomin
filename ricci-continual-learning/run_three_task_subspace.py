#!/usr/bin/env python3
"""Test subspace orthogonalization for 3 tasks."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.models import SimpleMLP
from src.continual_learning import EWCCL
from run_breakthrough_experiments import SubspaceRicciCL


def get_datasets(data_root='./data', subset_size=2000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    loaders = {}
    for name, cls in [('mnist', datasets.MNIST), ('fashion', datasets.FashionMNIST), ('kmnist', datasets.KMNIST)]:
        train = cls(data_root, train=True, download=True, transform=transform)
        test = cls(data_root, train=False, download=True, transform=transform)
        if subset_size:
            train = Subset(train, range(min(subset_size, len(train))))
            test = Subset(test, range(min(subset_size // 5, len(test))))
        loaders[f'{name}_train'] = DataLoader(train, batch_size=64, shuffle=True)
        loaders[f'{name}_test'] = DataLoader(test, batch_size=64, shuffle=False)

    return loaders


def main():
    device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    loaders = get_datasets('./data', 2000)

    # Test different lambda values for Subspace
    for ricci_lambda in [25, 50, 75, 100]:
        print(f"\n{'='*60}")
        print(f"SUBSPACE RICCI (λ={ricci_lambda}, subspace_dim=32)")
        print('='*60)

        model = SimpleMLP()
        learner = SubspaceRicciCL(model, device=device, ricci_lambda=ricci_lambda, subspace_dim=32)

        # Task A
        learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
        mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
        learner.after_task(loaders['mnist_train'], task_id=0)

        # Task B
        learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=5, verbose=False)
        mnist_b = learner.evaluate(loaders['mnist_test'])['accuracy']
        fashion_b = learner.evaluate(loaders['fashion_test'])['accuracy']
        learner.after_task(loaders['fashion_train'], task_id=1)

        # Task C
        learner.train_task(loaders['kmnist_train'], loaders['kmnist_test'], epochs=5, verbose=False)
        mnist_c = learner.evaluate(loaders['mnist_test'])['accuracy']
        fashion_c = learner.evaluate(loaders['fashion_test'])['accuracy']
        kmnist_c = learner.evaluate(loaders['kmnist_test'])['accuracy']

        print(f"MNIST: {mnist_a:.1%} → {mnist_b:.1%} → {mnist_c:.1%}")
        print(f"Fashion: {fashion_b:.1%} → {fashion_c:.1%}")
        print(f"KMNIST: {kmnist_c:.1%}")
        print(f"Average final: {(mnist_c + fashion_c + kmnist_c)/3:.1%}")


if __name__ == "__main__":
    main()
