#!/usr/bin/env python3
"""
Adaptive λ Adjustment Based on Predicted Geometric Flow

Key insight: λ should be higher when forgetting risk is high.
Risk = displacement × ambiguity

Predictions:
- Match EWC on 3-task (~39.5%)
- Exceed on 2-task (current 53% → target 65%+)
- Scale better to longer sequences (5+ tasks)
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
from collections import defaultdict

from src.models import SimpleMLP, get_weight_vector
from src.continual_learning import BaselineCL, EWCCL
from src.continual_learning_class_conditional import CentroidRicciCL


class AdaptiveLambdaRicciCL:
    """
    CC-Ricci with adaptive λ based on displacement × ambiguity risk.

    Key idea: Automatically increase λ when forgetting risk is high,
    decrease when safe to learn new information.
    """

    def __init__(
        self,
        model,
        device='cpu',
        base_lambda=50.0,
        risk_threshold=1.0,
        lambda_min=10.0,
        lambda_max=500.0,
        num_classes=10,
        adaptation_rate=0.1
    ):
        self.model = model.to(device)
        self.device = device
        self.base_lambda = base_lambda
        self.risk_threshold = risk_threshold
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.num_classes = num_classes
        self.adaptation_rate = adaptation_rate

        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.initial_weights = get_weight_vector(model).clone()

        # Reference structures for previous tasks
        self.reference_centroids = {}  # task_id -> {class: centroid}
        self.reference_ambiguity = {}  # task_id -> {class: ambiguity}

        # Tracking
        self.current_lambda = base_lambda
        self.lambda_history = []
        self.risk_history = []

    def compute_centroids(self, dataloader):
        """Compute current class centroids."""
        self.model.eval()
        embeddings_by_class = defaultdict(list)

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                embs = self.model.get_embeddings(batch_x)
                for emb, label in zip(embs, batch_y):
                    embeddings_by_class[label.item()].append(emb)

        centroids = {}
        for c, embs in embeddings_by_class.items():
            centroids[c] = torch.stack(embs).mean(dim=0)
        return centroids

    def compute_ambiguity(self, centroids):
        """
        Compute per-class ambiguity based on inter-class distances.
        Ambiguity = intra-class-spread / min-inter-class-distance
        """
        ambiguity = {}
        classes = sorted(centroids.keys())

        for c in classes:
            # Find minimum distance to other centroids
            min_dist = float('inf')
            for other_c in classes:
                if other_c != c:
                    dist = torch.norm(centroids[c] - centroids[other_c]).item()
                    min_dist = min(min_dist, dist)

            # Ambiguity is inverse of separation
            if min_dist > 0:
                ambiguity[c] = 1.0 / min_dist
            else:
                ambiguity[c] = 1.0

        return ambiguity

    def compute_displacement(self, current_centroids):
        """Compute displacement from reference centroids."""
        if not self.reference_centroids:
            return {}

        displacement = {}
        for task_id, ref_cents in self.reference_centroids.items():
            for c, ref_cent in ref_cents.items():
                if c in current_centroids:
                    disp = torch.norm(current_centroids[c] - ref_cent).item()
                    if c in displacement:
                        displacement[c] = max(displacement[c], disp)
                    else:
                        displacement[c] = disp

        return displacement

    def compute_risk(self, current_centroids):
        """
        Compute forgetting risk = displacement × ambiguity.
        Averaged across all reference classes.
        """
        if not self.reference_centroids:
            return 0.0

        displacement = self.compute_displacement(current_centroids)

        # Get reference ambiguity
        total_risk = 0.0
        count = 0
        for task_id, ref_amb in self.reference_ambiguity.items():
            for c, amb in ref_amb.items():
                if c in displacement:
                    risk = displacement[c] * amb
                    total_risk += risk
                    count += 1

        return total_risk / max(count, 1)

    def adapt_lambda(self, risk):
        """Adapt λ based on current risk."""
        # λ_t = base_λ × (1 + risk / threshold)
        adapted = self.base_lambda * (1 + risk / self.risk_threshold)

        # Smooth adaptation
        self.current_lambda = (
            (1 - self.adaptation_rate) * self.current_lambda +
            self.adaptation_rate * adapted
        )

        # Clamp to bounds
        self.current_lambda = max(self.lambda_min, min(self.lambda_max, self.current_lambda))

        return self.current_lambda

    def compute_centroid_loss(self, embeddings, labels):
        """Compute centroid preservation loss."""
        if not self.reference_centroids:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        # Compute current batch centroids
        batch_centroids = {}
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                batch_centroids[c] = embeddings[mask].mean(dim=0)

        # Loss against all reference centroids
        for task_id, ref_cents in self.reference_centroids.items():
            for c, ref_cent in ref_cents.items():
                if c in batch_centroids:
                    loss = loss + F.mse_loss(batch_centroids[c], ref_cent)

        return loss

    def after_task(self, dataloader, task_id=0):
        """Store reference after task completion."""
        print(f"Computing adaptive reference for task {task_id}...")

        centroids = self.compute_centroids(dataloader)
        ambiguity = self.compute_ambiguity(centroids)

        self.reference_centroids[task_id] = {c: v.detach().clone() for c, v in centroids.items()}
        self.reference_ambiguity[task_id] = ambiguity

        print(f"  Stored {len(centroids)} centroids")
        print(f"  Mean ambiguity: {np.mean(list(ambiguity.values())):.4f}")

    def train_task(self, train_loader, test_loader, epochs=5, verbose=True):
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_ce_loss = 0
            total_ricci_loss = 0

            # Compute risk ONCE at start of epoch (not every batch!)
            if self.reference_centroids:
                with torch.no_grad():
                    current_cents = self.compute_centroids(train_loader)
                    risk = self.compute_risk(current_cents)
                    lambda_t = self.adapt_lambda(risk)
            else:
                risk = 0.0
                lambda_t = self.base_lambda

            self.lambda_history.append(lambda_t)
            self.risk_history.append(risk)

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()

                logits, embeddings = self.model.forward_with_embeddings(batch_x)
                ce_loss = criterion(logits, batch_y)
                ricci_loss = self.compute_centroid_loss(embeddings, batch_y)

                loss = ce_loss + lambda_t * ricci_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_ricci_loss += ricci_loss.item()

            if verbose:
                acc = self.evaluate(test_loader)['accuracy']
                print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, "
                      f"Acc={acc:.1%}, λ={lambda_t:.1f}, risk={risk:.4f}")

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


def run_2_task_experiment(device, loaders):
    """Test adaptive λ on 2-task setting."""
    print("\n" + "="*70)
    print("2-TASK EXPERIMENT: MNIST → Fashion")
    print("="*70)

    results = {}

    # Baseline
    print("\n--- Baseline ---")
    model = SimpleMLP()
    learner = BaselineCL(model, device=device)
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
    mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=5, verbose=False)
    mnist_b = learner.evaluate(loaders['mnist_test'])['accuracy']
    fashion_b = learner.evaluate(loaders['fashion_test'])['accuracy']
    print(f"  MNIST: {mnist_a:.1%} → {mnist_b:.1%}, Fashion: {fashion_b:.1%}")
    results['Baseline'] = {'mnist_b': mnist_b, 'fashion_b': fashion_b}

    # EWC
    print("\n--- EWC (λ=5000) ---")
    model = SimpleMLP()
    learner = EWCCL(model, device=device, ewc_lambda=5000)
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
    mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
    learner.after_task(loaders['mnist_train'], task_id=0)
    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=5, verbose=False)
    mnist_b = learner.evaluate(loaders['mnist_test'])['accuracy']
    fashion_b = learner.evaluate(loaders['fashion_test'])['accuracy']
    print(f"  MNIST: {mnist_a:.1%} → {mnist_b:.1%}, Fashion: {fashion_b:.1%}")
    results['EWC'] = {'mnist_b': mnist_b, 'fashion_b': fashion_b}

    # Fixed λ CC-Ricci
    for lam in [50, 100]:
        print(f"\n--- Fixed CC-Ricci (λ={lam}) ---")
        model = SimpleMLP()
        learner = CentroidRicciCL(model, device=device, ricci_lambda=lam)
        learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
        mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
        learner.after_task(loaders['mnist_train'], task_id=0)
        learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=5, verbose=False)
        mnist_b = learner.evaluate(loaders['mnist_test'])['accuracy']
        fashion_b = learner.evaluate(loaders['fashion_test'])['accuracy']
        print(f"  MNIST: {mnist_a:.1%} → {mnist_b:.1%}, Fashion: {fashion_b:.1%}")
        results[f'Fixed-{lam}'] = {'mnist_b': mnist_b, 'fashion_b': fashion_b}

    # Adaptive λ
    for base_lam in [25, 50, 75]:
        print(f"\n--- Adaptive CC-Ricci (base_λ={base_lam}) ---")
        model = SimpleMLP()
        learner = AdaptiveLambdaRicciCL(
            model, device=device,
            base_lambda=base_lam,
            risk_threshold=0.5,
            lambda_min=10,
            lambda_max=300
        )
        learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
        mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
        learner.after_task(loaders['mnist_train'], task_id=0)
        learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=5, verbose=True)
        mnist_b = learner.evaluate(loaders['mnist_test'])['accuracy']
        fashion_b = learner.evaluate(loaders['fashion_test'])['accuracy']
        print(f"  FINAL: MNIST {mnist_b:.1%}, Fashion {fashion_b:.1%}")

        # Lambda stats
        if learner.lambda_history:
            print(f"  λ range: [{min(learner.lambda_history):.1f}, {max(learner.lambda_history):.1f}]")
            print(f"  λ mean: {np.mean(learner.lambda_history):.1f}")

        results[f'Adaptive-{base_lam}'] = {'mnist_b': mnist_b, 'fashion_b': fashion_b}

    return results


def run_3_task_experiment(device, loaders):
    """Test adaptive λ on 3-task setting."""
    print("\n" + "="*70)
    print("3-TASK EXPERIMENT: MNIST → Fashion → KMNIST")
    print("="*70)

    results = {}

    # Baseline
    print("\n--- Baseline ---")
    model = SimpleMLP()
    learner = BaselineCL(model, device=device)
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=5, verbose=False)
    learner.train_task(loaders['kmnist_train'], loaders['kmnist_test'], epochs=5, verbose=False)
    mnist = learner.evaluate(loaders['mnist_test'])['accuracy']
    fashion = learner.evaluate(loaders['fashion_test'])['accuracy']
    kmnist = learner.evaluate(loaders['kmnist_test'])['accuracy']
    avg = (mnist + fashion + kmnist) / 3
    print(f"  MNIST: {mnist:.1%}, Fashion: {fashion:.1%}, KMNIST: {kmnist:.1%}, Avg: {avg:.1%}")
    results['Baseline'] = {'mnist': mnist, 'fashion': fashion, 'kmnist': kmnist, 'avg': avg}

    # EWC
    print("\n--- EWC (λ=5000) ---")
    model = SimpleMLP()
    learner = EWCCL(model, device=device, ewc_lambda=5000)
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
    learner.after_task(loaders['mnist_train'], task_id=0)
    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=5, verbose=False)
    learner.after_task(loaders['fashion_train'], task_id=1)
    learner.train_task(loaders['kmnist_train'], loaders['kmnist_test'], epochs=5, verbose=False)
    mnist = learner.evaluate(loaders['mnist_test'])['accuracy']
    fashion = learner.evaluate(loaders['fashion_test'])['accuracy']
    kmnist = learner.evaluate(loaders['kmnist_test'])['accuracy']
    avg = (mnist + fashion + kmnist) / 3
    print(f"  MNIST: {mnist:.1%}, Fashion: {fashion:.1%}, KMNIST: {kmnist:.1%}, Avg: {avg:.1%}")
    results['EWC'] = {'mnist': mnist, 'fashion': fashion, 'kmnist': kmnist, 'avg': avg}

    # Fixed λ
    print("\n--- Fixed CC-Ricci (λ=50) ---")
    model = SimpleMLP()
    learner = CentroidRicciCL(model, device=device, ricci_lambda=50)
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
    learner.after_task(loaders['mnist_train'], task_id=0)
    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=5, verbose=False)
    learner.after_task(loaders['fashion_train'], task_id=1)
    learner.train_task(loaders['kmnist_train'], loaders['kmnist_test'], epochs=5, verbose=False)
    mnist = learner.evaluate(loaders['mnist_test'])['accuracy']
    fashion = learner.evaluate(loaders['fashion_test'])['accuracy']
    kmnist = learner.evaluate(loaders['kmnist_test'])['accuracy']
    avg = (mnist + fashion + kmnist) / 3
    print(f"  MNIST: {mnist:.1%}, Fashion: {fashion:.1%}, KMNIST: {kmnist:.1%}, Avg: {avg:.1%}")
    results['Fixed-50'] = {'mnist': mnist, 'fashion': fashion, 'kmnist': kmnist, 'avg': avg}

    # Adaptive λ
    for base_lam in [25, 50, 75]:
        print(f"\n--- Adaptive CC-Ricci (base_λ={base_lam}) ---")
        model = SimpleMLP()
        learner = AdaptiveLambdaRicciCL(
            model, device=device,
            base_lambda=base_lam,
            risk_threshold=0.5,
            lambda_min=10,
            lambda_max=300
        )

        # Task 1: MNIST
        print("  Task 1: MNIST")
        learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
        learner.after_task(loaders['mnist_train'], task_id=0)

        # Task 2: Fashion
        print("  Task 2: Fashion")
        learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=5, verbose=False)
        learner.after_task(loaders['fashion_train'], task_id=1)

        # Task 3: KMNIST
        print("  Task 3: KMNIST")
        learner.train_task(loaders['kmnist_train'], loaders['kmnist_test'], epochs=5, verbose=False)

        mnist = learner.evaluate(loaders['mnist_test'])['accuracy']
        fashion = learner.evaluate(loaders['fashion_test'])['accuracy']
        kmnist = learner.evaluate(loaders['kmnist_test'])['accuracy']
        avg = (mnist + fashion + kmnist) / 3

        print(f"  FINAL: MNIST {mnist:.1%}, Fashion {fashion:.1%}, KMNIST {kmnist:.1%}, Avg {avg:.1%}")

        if learner.lambda_history:
            print(f"  λ range: [{min(learner.lambda_history):.1f}, {max(learner.lambda_history):.1f}]")

        results[f'Adaptive-{base_lam}'] = {'mnist': mnist, 'fashion': fashion, 'kmnist': kmnist, 'avg': avg}

    return results


def run_5_task_experiment(device, loaders):
    """Test adaptive λ on 5-task setting for scalability."""
    print("\n" + "="*70)
    print("5-TASK EXPERIMENT: MNIST → Fashion → KMNIST → (MNIST rotated) → (Fashion rotated)")
    print("Note: Using augmented versions for tasks 4-5")
    print("="*70)

    # Create rotated versions
    rotate_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation((90, 90)),  # Fixed 90 degree rotation
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_rot = datasets.MNIST('./data', train=True, download=True, transform=rotate_transform)
    fashion_rot = datasets.FashionMNIST('./data', train=True, download=True, transform=rotate_transform)
    mnist_rot_test = datasets.MNIST('./data', train=False, transform=rotate_transform)
    fashion_rot_test = datasets.FashionMNIST('./data', train=False, transform=rotate_transform)

    mnist_rot = Subset(mnist_rot, range(2000))
    fashion_rot = Subset(fashion_rot, range(2000))
    mnist_rot_test = Subset(mnist_rot_test, range(400))
    fashion_rot_test = Subset(fashion_rot_test, range(400))

    loaders['mnist_rot_train'] = DataLoader(mnist_rot, batch_size=64, shuffle=True)
    loaders['mnist_rot_test'] = DataLoader(mnist_rot_test, batch_size=64, shuffle=False)
    loaders['fashion_rot_train'] = DataLoader(fashion_rot, batch_size=64, shuffle=True)
    loaders['fashion_rot_test'] = DataLoader(fashion_rot_test, batch_size=64, shuffle=False)

    task_sequence = [
        ('mnist', 'MNIST'),
        ('fashion', 'Fashion'),
        ('kmnist', 'KMNIST'),
        ('mnist_rot', 'MNIST-90°'),
        ('fashion_rot', 'Fashion-90°'),
    ]

    results = {}

    # EWC baseline
    print("\n--- EWC (λ=5000) ---")
    model = SimpleMLP()
    learner = EWCCL(model, device=device, ewc_lambda=5000)

    for i, (name, label) in enumerate(task_sequence):
        learner.train_task(loaders[f'{name}_train'], loaders[f'{name}_test'], epochs=5, verbose=False)
        if i < len(task_sequence) - 1:
            learner.after_task(loaders[f'{name}_train'], task_id=i)

    ewc_results = {}
    for name, label in task_sequence:
        acc = learner.evaluate(loaders[f'{name}_test'])['accuracy']
        ewc_results[name] = acc
        print(f"  {label}: {acc:.1%}")

    ewc_avg = np.mean(list(ewc_results.values()))
    print(f"  Average: {ewc_avg:.1%}")
    results['EWC'] = ewc_results
    results['EWC']['avg'] = ewc_avg

    # Adaptive λ
    print("\n--- Adaptive CC-Ricci (base_λ=50) ---")
    model = SimpleMLP()
    learner = AdaptiveLambdaRicciCL(
        model, device=device,
        base_lambda=50,
        risk_threshold=0.5,
        lambda_min=10,
        lambda_max=400
    )

    for i, (name, label) in enumerate(task_sequence):
        print(f"  Task {i+1}: {label}")
        learner.train_task(loaders[f'{name}_train'], loaders[f'{name}_test'], epochs=5, verbose=False)
        if i < len(task_sequence) - 1:
            learner.after_task(loaders[f'{name}_train'], task_id=i)

    adaptive_results = {}
    for name, label in task_sequence:
        acc = learner.evaluate(loaders[f'{name}_test'])['accuracy']
        adaptive_results[name] = acc
        print(f"  {label}: {acc:.1%}")

    adaptive_avg = np.mean(list(adaptive_results.values()))
    print(f"  Average: {adaptive_avg:.1%}")

    if learner.lambda_history:
        print(f"  λ range: [{min(learner.lambda_history):.1f}, {max(learner.lambda_history):.1f}]")
        print(f"  λ final: {learner.current_lambda:.1f}")

    results['Adaptive'] = adaptive_results
    results['Adaptive']['avg'] = adaptive_avg

    return results


def main():
    device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    loaders = get_datasets('./data', subset_size=2000)

    all_results = {}

    # Run experiments
    all_results['2-task'] = run_2_task_experiment(device, loaders)
    all_results['3-task'] = run_3_task_experiment(device, loaders)
    all_results['5-task'] = run_5_task_experiment(device, loaders)

    # Final Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY: ADAPTIVE λ RESULTS")
    print("="*70)

    print("\n2-TASK (MNIST → Fashion)")
    print("-"*50)
    print(f"{'Method':<20} {'MNIST Ret.':<12} {'Fashion':<12}")
    print("-"*50)
    for name, r in all_results['2-task'].items():
        print(f"{name:<20} {r['mnist_b']:.1%}        {r['fashion_b']:.1%}")

    best_2task = max(all_results['2-task'].items(), key=lambda x: x[1]['mnist_b'])
    print(f"\nBest 2-task: {best_2task[0]} with {best_2task[1]['mnist_b']:.1%} MNIST retention")

    print("\n3-TASK (MNIST → Fashion → KMNIST)")
    print("-"*50)
    print(f"{'Method':<20} {'Average':<12}")
    print("-"*50)
    for name, r in all_results['3-task'].items():
        print(f"{name:<20} {r['avg']:.1%}")

    best_3task = max(all_results['3-task'].items(), key=lambda x: x[1]['avg'])
    ewc_3task = all_results['3-task']['EWC']['avg']
    print(f"\nBest 3-task: {best_3task[0]} with {best_3task[1]['avg']:.1%} average")
    print(f"EWC baseline: {ewc_3task:.1%}")

    if best_3task[1]['avg'] >= ewc_3task:
        print(f">>> Adaptive MATCHES or BEATS EWC!")
    else:
        print(f">>> EWC still better by {(ewc_3task - best_3task[1]['avg']):.1%}")

    print("\n5-TASK Scalability")
    print("-"*50)
    ewc_5 = all_results['5-task']['EWC']['avg']
    adaptive_5 = all_results['5-task']['Adaptive']['avg']
    print(f"EWC: {ewc_5:.1%}")
    print(f"Adaptive: {adaptive_5:.1%}")

    if adaptive_5 > ewc_5:
        print(f">>> Adaptive scales BETTER to long sequences (+{(adaptive_5-ewc_5):.1%})")
    else:
        print(f">>> EWC scales better by {(ewc_5-adaptive_5):.1%}")

    # Overall verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    targets_met = 0
    print("\nTarget 1: 2-task MNIST retention > 65%")
    if best_2task[1]['mnist_b'] > 0.65:
        print(f"  ✓ ACHIEVED: {best_2task[1]['mnist_b']:.1%}")
        targets_met += 1
    else:
        print(f"  ✗ Not achieved: {best_2task[1]['mnist_b']:.1%}")

    print("\nTarget 2: 3-task average ≥ EWC (39.5%)")
    if best_3task[1]['avg'] >= 0.395:
        print(f"  ✓ ACHIEVED: {best_3task[1]['avg']:.1%}")
        targets_met += 1
    else:
        print(f"  ✗ Not achieved: {best_3task[1]['avg']:.1%}")

    print("\nTarget 3: 5-task scales better than EWC")
    if adaptive_5 > ewc_5:
        print(f"  ✓ ACHIEVED: {adaptive_5:.1%} > {ewc_5:.1%}")
        targets_met += 1
    else:
        print(f"  ✗ Not achieved: {adaptive_5:.1%} ≤ {ewc_5:.1%}")

    print(f"\nTargets met: {targets_met}/3")


if __name__ == "__main__":
    main()
