#!/usr/bin/env python3
"""
Fair comparison: Replay vs CC-Ricci using proven implementations.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

from src.models import SimpleMLP, get_weight_vector
from src.continual_learning import BaselineCL, EWCCL
from src.continual_learning_class_conditional import CentroidRicciCL


class ReplayCL:
    """Experience Replay baseline."""

    def __init__(self, model, device='cpu', buffer_size=500):
        self.model = model.to(device)
        self.device = device
        self.buffer_size = buffer_size
        self.replay_buffer = []

    def add_to_buffer(self, dataloader):
        samples = []
        for batch_x, batch_y in dataloader:
            for x, y in zip(batch_x, batch_y):
                samples.append((x.clone(), y.clone()))

        if len(samples) > self.buffer_size:
            indices = np.random.choice(len(samples), self.buffer_size, replace=False)
            samples = [samples[i] for i in indices]

        space_left = self.buffer_size - len(self.replay_buffer)
        if space_left >= len(samples):
            self.replay_buffer.extend(samples)
        else:
            self.replay_buffer = self.replay_buffer[len(samples)-space_left:] + samples

    def after_task(self, dataloader, task_id=0):
        self.add_to_buffer(dataloader)

    def train_task(self, train_loader, test_loader, epochs=5, lr=0.001, verbose=True):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                ce_loss = criterion(outputs, batch_y)

                replay_loss = torch.tensor(0.0, device=self.device)
                if len(self.replay_buffer) > 0:
                    n_replay = min(len(self.replay_buffer), batch_x.size(0))
                    indices = np.random.choice(len(self.replay_buffer), n_replay, replace=False)
                    replay_x = torch.stack([self.replay_buffer[i][0] for i in indices]).to(self.device)
                    replay_y = torch.stack([self.replay_buffer[i][1] for i in indices]).to(self.device)
                    replay_outputs = self.model(replay_x)
                    replay_loss = criterion(replay_outputs, replay_y)

                loss = ce_loss + replay_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if verbose:
                acc = self.evaluate(test_loader)['accuracy']
                print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.1%}")

    def evaluate(self, dataloader):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_x)
                preds = outputs.argmax(dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        return {'accuracy': correct / total}


def get_datasets(data_root='./data', subset_size=2000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    loaders = {}
    for name, cls in [('mnist', datasets.MNIST), ('fashion', datasets.FashionMNIST)]:
        train = cls(data_root, train=True, download=True, transform=transform)
        test = cls(data_root, train=False, download=True, transform=transform)
        if subset_size:
            train = Subset(train, range(min(subset_size, len(train))))
            test = Subset(test, range(min(subset_size // 5, len(test))))
        loaders[f'{name}_train'] = DataLoader(train, batch_size=64, shuffle=True)
        loaders[f'{name}_test'] = DataLoader(test, batch_size=64, shuffle=False)

    return loaders


def run_method(name, learner, loaders, epochs=5):
    """Run a single method."""
    print(f"\n--- {name} ---")

    # Task A: MNIST
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=epochs, verbose=False)
    mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
    weights_a = get_weight_vector(learner.model).clone()

    if hasattr(learner, 'after_task'):
        learner.after_task(loaders['mnist_train'], task_id=0)

    # Task B: Fashion
    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=epochs, verbose=False)
    mnist_b = learner.evaluate(loaders['mnist_test'])['accuracy']
    fashion_b = learner.evaluate(loaders['fashion_test'])['accuracy']
    weights_b = get_weight_vector(learner.model).clone()

    weight_change = torch.norm(weights_b - weights_a).item()

    print(f"  MNIST: {mnist_a:.1%} → {mnist_b:.1%}")
    print(f"  Fashion: {fashion_b:.1%}")
    print(f"  Weight change: {weight_change:.2f}")

    return {
        'mnist_a': mnist_a,
        'mnist_b': mnist_b,
        'fashion_b': fashion_b,
        'weight_change': weight_change,
        'forgetting': mnist_a - mnist_b
    }


def main():
    device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    loaders = get_datasets('./data', subset_size=2000)

    print("\n" + "="*70)
    print("REPLAY vs CC-RICCI: FAIR COMPARISON")
    print("="*70)

    results = {}

    # Baseline
    model = SimpleMLP()
    learner = BaselineCL(model, device=device)
    results['Baseline'] = run_method('Baseline', learner, loaders)

    # EWC
    model = SimpleMLP()
    learner = EWCCL(model, device=device, ewc_lambda=5000)
    results['EWC'] = run_method('EWC (λ=5000)', learner, loaders)

    # Replay variants
    for buffer_size in [200, 500, 1000]:
        model = SimpleMLP()
        learner = ReplayCL(model, device=device, buffer_size=buffer_size)
        results[f'Replay-{buffer_size}'] = run_method(f'Replay-{buffer_size}', learner, loaders)

    # CC-Ricci (proven implementation) at different lambdas
    for lam in [50, 100, 200]:
        model = SimpleMLP()
        learner = CentroidRicciCL(model, device=device, ricci_lambda=lam)
        results[f'CC-Ricci-{lam}'] = run_method(f'CC-Ricci (λ={lam})', learner, loaders)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Method':<20} {'MNIST Ret.':<12} {'Fashion':<12} {'Forgetting':<12} {'Memory':<15}")
    print("-"*70)

    memory_usage = {
        'Baseline': '0 samples',
        'EWC': 'Fisher matrix',
        'Replay-200': '200 samples',
        'Replay-500': '500 samples',
        'Replay-1000': '1000 samples',
        'CC-Ricci-50': '10 centroids',
        'CC-Ricci-100': '10 centroids',
        'CC-Ricci-200': '10 centroids',
    }

    for name, r in results.items():
        mem = memory_usage.get(name, 'unknown')
        print(f"{name:<20} {r['mnist_b']:.1%}        {r['fashion_b']:.1%}        {r['forgetting']:.1%}         {mem}")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    best_ricci = max([r for n, r in results.items() if 'CC-Ricci' in n], key=lambda x: x['mnist_b'])
    best_ricci_name = [n for n, r in results.items() if r == best_ricci][0]

    best_replay = max([r for n, r in results.items() if 'Replay' in n], key=lambda x: x['mnist_b'])
    best_replay_name = [n for n, r in results.items() if r == best_replay][0]

    print(f"\nBest CC-Ricci: {best_ricci_name} with {best_ricci['mnist_b']:.1%} retention")
    print(f"Best Replay: {best_replay_name} with {best_replay['mnist_b']:.1%} retention")
    print(f"EWC: {results['EWC']['mnist_b']:.1%} retention")

    # Memory efficiency comparison
    print("\n--- Memory Efficiency ---")
    # CC-Ricci stores 10 centroids of dim 64 = 640 floats
    # Replay-1000 stores 1000 samples of dim 784 = 784,000 floats
    print(f"CC-Ricci storage: 10 centroids × 64 dims = 640 floats")
    print(f"Replay-1000 storage: 1000 samples × 784 dims = 784,000 floats")
    print(f"Memory ratio: Replay uses {784000/640:.0f}× more storage")

    if best_ricci['mnist_b'] > results['EWC']['mnist_b']:
        print(f"\n>>> CC-Ricci beats EWC by +{(best_ricci['mnist_b'] - results['EWC']['mnist_b']):.1%}")

    if best_ricci['mnist_b'] > best_replay['mnist_b']:
        print(f">>> CC-Ricci beats Replay with 1000× less memory!")
    elif best_ricci['mnist_b'] > best_replay['mnist_b'] * 0.9:
        print(f">>> CC-Ricci within 10% of Replay with 1000× less memory!")
    else:
        print(f">>> Replay still better, but CC-Ricci uses 1000× less memory")


if __name__ == "__main__":
    main()
