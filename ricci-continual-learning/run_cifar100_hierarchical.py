#!/usr/bin/env python3
"""
Experiment D: Hierarchical Class Structure (CIFAR-100)

Tests whether preserving inter-superclass geometry gives better
transfer than flat CC-Ricci.

CIFAR-100: 100 classes, 20 superclasses (5 classes each)

Hypothesis: Multi-scale Ricci (superclass + class level) works even better.
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

from src.models import get_weight_vector


# CIFAR-100 superclass mapping (20 superclasses, 5 fine classes each)
CIFAR100_SUPERCLASSES = {
    'aquatic_mammals': [4, 30, 55, 72, 95],
    'fish': [1, 32, 67, 73, 91],
    'flowers': [54, 62, 70, 82, 92],
    'food_containers': [9, 10, 16, 28, 61],
    'fruit_and_vegetables': [0, 51, 53, 57, 83],
    'household_electrical': [22, 39, 40, 86, 87],
    'household_furniture': [5, 20, 25, 84, 94],
    'insects': [6, 7, 14, 18, 24],
    'large_carnivores': [3, 42, 43, 88, 97],
    'large_man-made_outdoor': [12, 17, 37, 68, 76],
    'large_natural_outdoor': [23, 33, 49, 60, 71],
    'large_omnivores_herbivores': [15, 19, 21, 31, 38],
    'medium_mammals': [34, 63, 64, 66, 75],
    'non-insect_invertebrates': [26, 45, 77, 79, 99],
    'people': [2, 11, 35, 46, 98],
    'reptiles': [27, 29, 44, 78, 93],
    'small_mammals': [36, 50, 65, 74, 80],
    'trees': [47, 52, 56, 59, 96],
    'vehicles_1': [8, 13, 48, 58, 90],
    'vehicles_2': [41, 69, 81, 85, 89],
}

# Create reverse mapping: fine class -> superclass index
FINE_TO_SUPER = {}
for super_idx, (super_name, fine_classes) in enumerate(CIFAR100_SUPERCLASSES.items()):
    for fine_class in fine_classes:
        FINE_TO_SUPER[fine_class] = super_idx


class ConvNetCIFAR(nn.Module):
    """CNN for CIFAR-100."""

    def __init__(self, num_classes=100, embedding_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.fc1 = nn.Linear(256 * 4 * 4, embedding_dim)
        self.fc1_bn = nn.BatchNorm1d(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        return self.classifier(x)

    def get_embeddings(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1_bn(self.fc1(x)))

    def forward_with_embeddings(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        embeddings = F.relu(self.fc1_bn(self.fc1(x)))
        return self.classifier(embeddings), embeddings


class HierarchicalCentroidRegularizer(nn.Module):
    """
    Multi-scale centroid regularizer.

    Preserves:
    1. Fine-class centroids (within superclass structure)
    2. Superclass centroids (coarse structure)
    3. Relative distances between superclasses
    """

    def __init__(self, num_fine_classes=100, num_super_classes=20,
                 fine_weight=1.0, super_weight=1.0, inter_super_weight=1.0):
        super().__init__()
        self.num_fine_classes = num_fine_classes
        self.num_super_classes = num_super_classes
        self.fine_weight = fine_weight
        self.super_weight = super_weight
        self.inter_super_weight = inter_super_weight

        self.reference_fine_centroids = None
        self.reference_super_centroids = None
        self.reference_inter_super_distances = None

    def set_reference(self, embeddings, labels):
        """Store reference centroids at both levels."""
        # Fine-class centroids
        fine_centroids = {}
        for c in range(self.num_fine_classes):
            mask = labels == c
            if mask.sum() > 0:
                fine_centroids[c] = embeddings[mask].mean(dim=0).detach()

        self.reference_fine_centroids = fine_centroids

        # Superclass centroids
        super_centroids = {}
        for super_idx in range(self.num_super_classes):
            # Get all embeddings for this superclass
            super_mask = torch.zeros(len(labels), dtype=torch.bool, device=labels.device)
            for fine_c, super_c in FINE_TO_SUPER.items():
                if super_c == super_idx:
                    super_mask = super_mask | (labels == fine_c)

            if super_mask.sum() > 0:
                super_centroids[super_idx] = embeddings[super_mask].mean(dim=0).detach()

        self.reference_super_centroids = super_centroids

        # Inter-superclass distances
        inter_distances = {}
        super_ids = sorted(super_centroids.keys())
        for i, si in enumerate(super_ids):
            for j, sj in enumerate(super_ids):
                if i < j:
                    d = torch.norm(super_centroids[si] - super_centroids[sj])
                    inter_distances[(si, sj)] = d.detach()

        self.reference_inter_super_distances = inter_distances

    def forward(self, embeddings, labels):
        if self.reference_fine_centroids is None:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        total_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # 1. Fine-class centroid loss
        fine_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        for c in self.reference_fine_centroids:
            mask = labels == c
            if mask.sum() > 0:
                current_centroid = embeddings[mask].mean(dim=0)
                fine_loss = fine_loss + F.mse_loss(
                    current_centroid, self.reference_fine_centroids[c]
                )

        # 2. Superclass centroid loss
        super_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        for super_idx in self.reference_super_centroids:
            super_mask = torch.zeros(len(labels), dtype=torch.bool, device=labels.device)
            for fine_c, super_c in FINE_TO_SUPER.items():
                if super_c == super_idx:
                    super_mask = super_mask | (labels == fine_c)

            if super_mask.sum() > 0:
                current_super_centroid = embeddings[super_mask].mean(dim=0)
                super_loss = super_loss + F.mse_loss(
                    current_super_centroid, self.reference_super_centroids[super_idx]
                )

        # 3. Inter-superclass distance preservation
        inter_loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        # Compute current superclass centroids
        current_super_centroids = {}
        for super_idx in self.reference_super_centroids:
            super_mask = torch.zeros(len(labels), dtype=torch.bool, device=labels.device)
            for fine_c, super_c in FINE_TO_SUPER.items():
                if super_c == super_idx:
                    super_mask = super_mask | (labels == fine_c)

            if super_mask.sum() > 0:
                current_super_centroids[super_idx] = embeddings[super_mask].mean(dim=0)

        for (si, sj), ref_dist in self.reference_inter_super_distances.items():
            if si in current_super_centroids and sj in current_super_centroids:
                current_dist = torch.norm(
                    current_super_centroids[si] - current_super_centroids[sj]
                )
                inter_loss = inter_loss + (current_dist - ref_dist) ** 2

        total_loss = (self.fine_weight * fine_loss +
                      self.super_weight * super_loss +
                      self.inter_super_weight * inter_loss)

        return total_loss


class FlatCentroidRegularizer(nn.Module):
    """Standard flat centroid preservation (no hierarchy)."""

    def __init__(self, num_classes=100):
        super().__init__()
        self.num_classes = num_classes
        self.reference_centroids = None

    def set_reference(self, embeddings, labels):
        centroids = {}
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                centroids[c] = embeddings[mask].mean(dim=0).detach()
        self.reference_centroids = centroids

    def forward(self, embeddings, labels):
        if self.reference_centroids is None:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        for c in self.reference_centroids:
            mask = labels == c
            if mask.sum() > 0:
                current = embeddings[mask].mean(dim=0)
                loss = loss + F.mse_loss(current, self.reference_centroids[c])

        return loss


class HierarchicalRicciCL:
    """Continual learning with hierarchical centroid preservation."""

    def __init__(self, model, device='cpu', ricci_lambda=50.0,
                 fine_weight=1.0, super_weight=1.0, inter_super_weight=1.0):
        self.model = model.to(device)
        self.device = device
        self.ricci_lambda = ricci_lambda
        self.fine_weight = fine_weight
        self.super_weight = super_weight
        self.inter_super_weight = inter_super_weight

        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.regularizers = {}

    def after_task(self, dataloader, task_id=0):
        print(f"Computing hierarchical reference for task {task_id}...")
        self.model.eval()

        all_embs = []
        all_labels = []

        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(self.device)
                embs = self.model.get_embeddings(data)
                all_embs.append(embs)
                all_labels.append(target)

        all_embs = torch.cat(all_embs, dim=0)
        all_labels = torch.cat(all_labels, dim=0).to(self.device)

        reg = HierarchicalCentroidRegularizer(
            fine_weight=self.fine_weight,
            super_weight=self.super_weight,
            inter_super_weight=self.inter_super_weight
        )
        reg.set_reference(all_embs, all_labels)
        self.regularizers[task_id] = reg
        print(f"Hierarchical reference stored.")

    def train_task(self, train_loader, test_loader, epochs=5, verbose=True):
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                logits, embs = self.model.forward_with_embeddings(data)

                ce_loss = criterion(logits, target)
                ricci_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                for reg in self.regularizers.values():
                    ricci_loss = ricci_loss + reg(embs, target)

                loss = ce_loss + self.ricci_lambda * ricci_loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if verbose:
                acc = self.evaluate(test_loader)['accuracy']
                print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.1%}")

    def evaluate(self, dataloader):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                out = self.model(data)
                pred = out.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        return {'accuracy': correct / total}


class FlatRicciCL:
    """Continual learning with flat centroid preservation."""

    def __init__(self, model, device='cpu', ricci_lambda=50.0):
        self.model = model.to(device)
        self.device = device
        self.ricci_lambda = ricci_lambda
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.regularizers = {}

    def after_task(self, dataloader, task_id=0):
        print(f"Computing flat reference for task {task_id}...")
        self.model.eval()

        all_embs = []
        all_labels = []

        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(self.device)
                embs = self.model.get_embeddings(data)
                all_embs.append(embs)
                all_labels.append(target)

        all_embs = torch.cat(all_embs, dim=0)
        all_labels = torch.cat(all_labels, dim=0).to(self.device)

        reg = FlatCentroidRegularizer(num_classes=100)
        reg.set_reference(all_embs, all_labels)
        self.regularizers[task_id] = reg
        print(f"Flat reference stored.")

    def train_task(self, train_loader, test_loader, epochs=5, verbose=True):
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                logits, embs = self.model.forward_with_embeddings(data)

                ce_loss = criterion(logits, target)
                ricci_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

                for reg in self.regularizers.values():
                    ricci_loss = ricci_loss + reg(embs, target)

                loss = ce_loss + self.ricci_lambda * ricci_loss
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if verbose:
                acc = self.evaluate(test_loader)['accuracy']
                print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.1%}")

    def evaluate(self, dataloader):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                out = self.model(data)
                pred = out.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        return {'accuracy': correct / total}


class BaselineCL:
    """Simple baseline with no regularization."""

    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def after_task(self, dataloader, task_id=0):
        pass

    def train_task(self, train_loader, test_loader, epochs=5, verbose=True):
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(data)
                loss = criterion(out, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if verbose:
                acc = self.evaluate(test_loader)['accuracy']
                print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.1%}")

    def evaluate(self, dataloader):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                out = self.model(data)
                pred = out.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)
        return {'accuracy': correct / total}


def get_cifar100_by_superclass(data_root='./data', superclass_ids=[0, 1], subset_per_class=100):
    """Get CIFAR-100 data filtered by superclass."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = datasets.CIFAR100(data_root, train=True, download=True, transform=transform)
    test_data = datasets.CIFAR100(data_root, train=False, download=True, transform=transform)

    # Get fine classes for selected superclasses
    selected_fine_classes = []
    for super_idx in superclass_ids:
        super_name = list(CIFAR100_SUPERCLASSES.keys())[super_idx]
        selected_fine_classes.extend(CIFAR100_SUPERCLASSES[super_name])

    # Filter datasets
    def filter_by_classes(dataset, classes, subset_per_class):
        indices = []
        class_counts = defaultdict(int)
        for idx in range(len(dataset)):
            label = dataset.targets[idx]
            if label in classes and class_counts[label] < subset_per_class:
                indices.append(idx)
                class_counts[label] += 1
        return Subset(dataset, indices)

    train_subset = filter_by_classes(train_data, selected_fine_classes, subset_per_class)
    test_subset = filter_by_classes(test_data, selected_fine_classes, subset_per_class // 5)

    return {
        'train': DataLoader(train_subset, batch_size=64, shuffle=True),
        'test': DataLoader(test_subset, batch_size=64, shuffle=False),
        'classes': selected_fine_classes
    }


def main():
    device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    print("\n" + "="*70)
    print("EXPERIMENT D: HIERARCHICAL vs FLAT CENTROID PRESERVATION (CIFAR-100)")
    print("="*70)

    # Task sequence: 4 superclasses (2 at a time)
    # Task A: superclasses 0, 1 (aquatic_mammals, fish)
    # Task B: superclasses 2, 3 (flowers, food_containers)

    print("\nLoading CIFAR-100 data...")
    task_a_data = get_cifar100_by_superclass('./data', [0, 1], subset_per_class=100)
    task_b_data = get_cifar100_by_superclass('./data', [2, 3], subset_per_class=100)

    print(f"Task A classes: {task_a_data['classes'][:5]}... (10 fine classes)")
    print(f"Task B classes: {task_b_data['classes'][:5]}... (10 fine classes)")

    results = {}

    # Baseline
    print("\n--- Baseline ---")
    model = ConvNetCIFAR()
    learner = BaselineCL(model, device=device)
    learner.train_task(task_a_data['train'], task_a_data['test'], epochs=10, verbose=False)
    task_a_acc_after_a = learner.evaluate(task_a_data['test'])['accuracy']
    print(f"  After Task A: {task_a_acc_after_a:.1%}")

    learner.train_task(task_b_data['train'], task_b_data['test'], epochs=10, verbose=False)
    task_a_acc_after_b = learner.evaluate(task_a_data['test'])['accuracy']
    task_b_acc = learner.evaluate(task_b_data['test'])['accuracy']
    print(f"  After Task B: Task A={task_a_acc_after_b:.1%}, Task B={task_b_acc:.1%}")

    results['Baseline'] = {
        'task_a_after_a': task_a_acc_after_a,
        'task_a_after_b': task_a_acc_after_b,
        'task_b': task_b_acc,
        'forgetting': task_a_acc_after_a - task_a_acc_after_b
    }

    # Flat CC-Ricci
    print("\n--- Flat CC-Ricci (λ=50) ---")
    model = ConvNetCIFAR()
    learner = FlatRicciCL(model, device=device, ricci_lambda=50)
    learner.train_task(task_a_data['train'], task_a_data['test'], epochs=10, verbose=False)
    task_a_acc_after_a = learner.evaluate(task_a_data['test'])['accuracy']
    learner.after_task(task_a_data['train'], task_id=0)
    print(f"  After Task A: {task_a_acc_after_a:.1%}")

    learner.train_task(task_b_data['train'], task_b_data['test'], epochs=10, verbose=False)
    task_a_acc_after_b = learner.evaluate(task_a_data['test'])['accuracy']
    task_b_acc = learner.evaluate(task_b_data['test'])['accuracy']
    print(f"  After Task B: Task A={task_a_acc_after_b:.1%}, Task B={task_b_acc:.1%}")

    results['Flat CC-Ricci'] = {
        'task_a_after_a': task_a_acc_after_a,
        'task_a_after_b': task_a_acc_after_b,
        'task_b': task_b_acc,
        'forgetting': task_a_acc_after_a - task_a_acc_after_b
    }

    # Hierarchical CC-Ricci
    print("\n--- Hierarchical CC-Ricci (λ=50) ---")
    model = ConvNetCIFAR()
    learner = HierarchicalRicciCL(model, device=device, ricci_lambda=50,
                                   fine_weight=1.0, super_weight=1.0, inter_super_weight=0.5)
    learner.train_task(task_a_data['train'], task_a_data['test'], epochs=10, verbose=False)
    task_a_acc_after_a = learner.evaluate(task_a_data['test'])['accuracy']
    learner.after_task(task_a_data['train'], task_id=0)
    print(f"  After Task A: {task_a_acc_after_a:.1%}")

    learner.train_task(task_b_data['train'], task_b_data['test'], epochs=10, verbose=False)
    task_a_acc_after_b = learner.evaluate(task_a_data['test'])['accuracy']
    task_b_acc = learner.evaluate(task_b_data['test'])['accuracy']
    print(f"  After Task B: Task A={task_a_acc_after_b:.1%}, Task B={task_b_acc:.1%}")

    results['Hierarchical CC-Ricci'] = {
        'task_a_after_a': task_a_acc_after_a,
        'task_a_after_b': task_a_acc_after_b,
        'task_b': task_b_acc,
        'forgetting': task_a_acc_after_a - task_a_acc_after_b
    }

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Method':<25} {'Task A Ret.':<12} {'Task B':<12} {'Forgetting':<12}")
    print("-"*70)
    for name, r in results.items():
        print(f"{name:<25} {r['task_a_after_b']:.1%}        {r['task_b']:.1%}        {r['forgetting']:.1%}")

    # Analysis
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    flat = results['Flat CC-Ricci']
    hier = results['Hierarchical CC-Ricci']
    base = results['Baseline']

    print(f"\nHierarchical vs Flat:")
    print(f"  Task A retention: {hier['task_a_after_b']:.1%} vs {flat['task_a_after_b']:.1%}")
    print(f"  Task B accuracy: {hier['task_b']:.1%} vs {flat['task_b']:.1%}")
    print(f"  Forgetting: {hier['forgetting']:.1%} vs {flat['forgetting']:.1%}")

    if hier['task_a_after_b'] > flat['task_a_after_b']:
        diff = hier['task_a_after_b'] - flat['task_a_after_b']
        print(f"\n>>> HIERARCHICAL beats FLAT by +{diff:.1%}!")
        print(">>> Multi-scale Ricci works better!")
    else:
        diff = flat['task_a_after_b'] - hier['task_a_after_b']
        print(f"\n>>> Flat slightly better by +{diff:.1%}")

    if hier['forgetting'] < flat['forgetting']:
        print(f">>> Hierarchical has LESS forgetting ({hier['forgetting']:.1%} vs {flat['forgetting']:.1%})")


if __name__ == "__main__":
    main()
