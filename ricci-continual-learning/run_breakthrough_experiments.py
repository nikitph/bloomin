#!/usr/bin/env python3
"""
Breakthrough Experiments: Testing the Geometric Hypothesis

Experiment 1: Universal Geometry
- Train jointly on MNIST+Fashion to get g_universal
- Use g_universal as target for sequential training
- Prediction: Should improve BOTH tasks (70-80% MNIST, 80-82% Fashion)

Experiment 2: Subspace Orthogonalization
- Project centroid constraints into task-specific subspaces
- Allow full freedom in orthogonal directions
- Prediction: Reduce plasticity-stability tradeoff

Experiment 3: Three-Task Sequence
- MNIST → Fashion → KMNIST (or EMNIST)
- Test if geometric preservation composes
- Prediction: All tasks maintained above baseline
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import numpy as np
from datetime import datetime

from src.models import SimpleMLP, get_weight_vector
from src.continual_learning import BaselineCL, EWCCL


# ============================================================================
# EXPERIMENT 1: UNIVERSAL GEOMETRY
# ============================================================================

class UniversalGeometryCL:
    """
    Train with a universal target geometry learned from joint training.

    Instead of preserving MNIST-specific geometry (which constrains Fashion),
    preserve a universal geometry that works for both tasks.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        lr: float = 1e-3,
        ricci_lambda: float = 50.0,
        num_classes: int = 10
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.ricci_lambda = ricci_lambda
        self.num_classes = num_classes

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.initial_weights = get_weight_vector(model).clone()

        # Universal geometry target (set externally)
        self.universal_centroids = None
        self.universal_centroid_dists = None
        self.universal_centroid_angles = None

    def set_universal_geometry(self, centroids: torch.Tensor):
        """Set the universal geometry target."""
        self.universal_centroids = centroids.detach().clone()
        self.universal_centroid_dists = torch.cdist(centroids, centroids, p=2).detach().clone()

        # Angular structure
        global_centroid = centroids.mean(dim=0, keepdim=True)
        centered = centroids - global_centroid
        normalized = F.normalize(centered, dim=1, eps=1e-8)
        self.universal_centroid_angles = (normalized @ normalized.T).detach().clone()

    def compute_geometry_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Preserve universal geometry."""
        if self.universal_centroids is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        device = self.device
        dim = embeddings.shape[1]

        # Compute current centroids
        curr_centroids = torch.zeros(self.num_classes, dim, device=device)
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                curr_centroids[c] = embeddings[mask].mean(dim=0)

        # Distance structure loss (normalized)
        curr_dists = torch.cdist(curr_centroids, curr_centroids, p=2)
        ref_norm = self.universal_centroid_dists / (self.universal_centroid_dists.mean() + 1e-8)
        curr_norm = curr_dists / (curr_dists.mean() + 1e-8)
        dist_loss = F.mse_loss(curr_norm, ref_norm)

        # Angular structure loss
        global_centroid = curr_centroids.mean(dim=0, keepdim=True)
        centered = curr_centroids - global_centroid
        normalized = F.normalize(centered, dim=1, eps=1e-8)
        curr_angles = normalized @ normalized.T
        angle_loss = F.mse_loss(curr_angles, self.universal_centroid_angles)

        return dist_loss + angle_loss

    def train_epoch(self, dataloader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0
        total_ce = 0
        total_geom = 0
        correct = 0
        total = 0

        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            logits, embeddings = self.model.forward_with_embeddings(data)

            ce_loss = F.cross_entropy(logits, target)
            geom_loss = self.compute_geometry_loss(embeddings, target)

            loss = ce_loss + self.ricci_lambda * geom_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_geom += geom_loss.item()
            correct += (logits.argmax(1) == target).sum().item()
            total += target.size(0)

        return {
            'loss': total_ce / len(dataloader),
            'geom_loss': total_geom / len(dataloader),
            'accuracy': correct / total
        }

    def evaluate(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                correct += (output.argmax(1) == target).sum().item()
                total += target.size(0)
        return {'accuracy': correct / total}

    def train_task(self, train_loader, val_loader=None, epochs=5, verbose=True):
        for epoch in range(epochs):
            metrics = self.train_epoch(train_loader)
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                metrics['val_accuracy'] = val_metrics['accuracy']
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {metrics['loss']:.4f}, Geom: {metrics['geom_loss']:.4f}, Acc: {metrics['accuracy']:.4f}", end="")
                if val_loader:
                    print(f" - Val: {metrics['val_accuracy']:.4f}")
                else:
                    print()

    def get_weight_change(self):
        return torch.norm(get_weight_vector(self.model) - self.initial_weights).item()


def extract_centroids(model, dataloader, device, num_classes=10):
    """Extract class centroids from a trained model."""
    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            embeddings = model.get_embeddings(data)
            all_embeddings.append(embeddings)
            all_labels.append(target)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0).to(device)

    dim = all_embeddings.shape[1]
    centroids = torch.zeros(num_classes, dim, device=device)

    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            centroids[c] = all_embeddings[mask].mean(dim=0)

    return centroids


# ============================================================================
# EXPERIMENT 2: SUBSPACE ORTHOGONALIZATION
# ============================================================================

class SubspaceRicciCL:
    """
    Project geometric constraints into task-specific subspaces.

    MNIST geometry is preserved in MNIST-relevant directions.
    Fashion is free to use orthogonal directions.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        lr: float = 1e-3,
        ricci_lambda: float = 50.0,
        num_classes: int = 10,
        subspace_dim: int = 32  # Dimension of task-specific subspace
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.ricci_lambda = ricci_lambda
        self.num_classes = num_classes
        self.subspace_dim = subspace_dim

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.initial_weights = get_weight_vector(model).clone()

        # Task-specific projections
        self.task_projections = {}  # task_id -> projection matrix
        self.task_centroids = {}    # task_id -> centroids in subspace

    def compute_task_subspace(self, embeddings: torch.Tensor, labels: torch.Tensor, task_id: int):
        """Compute task-specific subspace via PCA on class means."""
        device = self.device
        dim = embeddings.shape[1]

        # Compute class centroids
        centroids = torch.zeros(self.num_classes, dim, device=device)
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                centroids[c] = embeddings[mask].mean(dim=0)

        # Center the centroids
        mean_centroid = centroids.mean(dim=0, keepdim=True)
        centered = centroids - mean_centroid

        # SVD to get principal directions
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)

        # Take top subspace_dim directions
        k = min(self.subspace_dim, Vh.shape[0])
        projection = Vh[:k]  # (k, dim)

        # Project centroids into subspace
        centroids_projected = centered @ projection.T  # (num_classes, k)

        self.task_projections[task_id] = projection.detach().clone()
        self.task_centroids[task_id] = centroids_projected.detach().clone()

    def compute_geometry_loss(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Preserve geometry in task-specific subspaces."""
        if not self.task_projections:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        device = self.device
        dim = embeddings.shape[1]
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Compute current centroids
        curr_centroids = torch.zeros(self.num_classes, dim, device=device)
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                curr_centroids[c] = embeddings[mask].mean(dim=0)

        mean_centroid = curr_centroids.mean(dim=0, keepdim=True)
        centered = curr_centroids - mean_centroid

        # Loss for each previous task
        for task_id, projection in self.task_projections.items():
            ref_centroids = self.task_centroids[task_id]

            # Project current centroids into task's subspace
            curr_projected = centered @ projection.T

            # Compute distances in subspace
            curr_dists = torch.cdist(curr_projected, curr_projected, p=2)
            ref_dists = torch.cdist(ref_centroids, ref_centroids, p=2)

            # Normalized distance loss
            curr_norm = curr_dists / (curr_dists.mean() + 1e-8)
            ref_norm = ref_dists / (ref_dists.mean() + 1e-8)

            total_loss = total_loss + F.mse_loss(curr_norm, ref_norm)

        return total_loss

    def train_epoch(self, dataloader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0
        total_ce = 0
        total_geom = 0
        correct = 0
        total = 0

        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            logits, embeddings = self.model.forward_with_embeddings(data)

            ce_loss = F.cross_entropy(logits, target)
            geom_loss = self.compute_geometry_loss(embeddings, target)

            loss = ce_loss + self.ricci_lambda * geom_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_geom += geom_loss.item()
            correct += (logits.argmax(1) == target).sum().item()
            total += target.size(0)

        return {
            'loss': total_ce / len(dataloader),
            'geom_loss': total_geom / len(dataloader),
            'accuracy': correct / total
        }

    def evaluate(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                correct += (output.argmax(1) == target).sum().item()
                total += target.size(0)
        return {'accuracy': correct / total}

    def train_task(self, train_loader, val_loader=None, epochs=5, verbose=True):
        for epoch in range(epochs):
            metrics = self.train_epoch(train_loader)
            if val_loader:
                val_metrics = self.evaluate(val_loader)
                metrics['val_accuracy'] = val_metrics['accuracy']
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {metrics['loss']:.4f}, Geom: {metrics['geom_loss']:.4f}, Acc: {metrics['accuracy']:.4f}", end="")
                if val_loader:
                    print(f" - Val: {metrics['val_accuracy']:.4f}")
                else:
                    print()

    def after_task(self, dataloader: DataLoader, task_id: int):
        """Store task-specific subspace."""
        print(f"Computing task {task_id} subspace...")
        self.model.eval()

        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for data, target in dataloader:
                data = data.to(self.device)
                embeddings = self.model.get_embeddings(data)
                all_embeddings.append(embeddings)
                all_labels.append(target)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0).to(self.device)

        self.compute_task_subspace(all_embeddings, all_labels, task_id)
        print(f"Task {task_id} subspace stored.")

    def get_weight_change(self):
        return torch.norm(get_weight_vector(self.model) - self.initial_weights).item()


# ============================================================================
# EXPERIMENT 3: THREE-TASK SEQUENCE
# ============================================================================

from src.continual_learning_class_conditional import CentroidRicciCL


# ============================================================================
# DATA LOADING
# ============================================================================

def get_all_datasets(data_root='./data', subset_size=2000):
    """Load MNIST, FashionMNIST, and KMNIST."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    datasets_dict = {}

    for name, dataset_class in [
        ('mnist', datasets.MNIST),
        ('fashion', datasets.FashionMNIST),
        ('kmnist', datasets.KMNIST)
    ]:
        train_data = dataset_class(data_root, train=True, download=True, transform=transform)
        test_data = dataset_class(data_root, train=False, download=True, transform=transform)

        if subset_size:
            train_data = Subset(train_data, range(min(subset_size, len(train_data))))
            test_data = Subset(test_data, range(min(subset_size // 5, len(test_data))))

        datasets_dict[f'{name}_train'] = DataLoader(train_data, batch_size=64, shuffle=True)
        datasets_dict[f'{name}_test'] = DataLoader(test_data, batch_size=64, shuffle=False)

    return datasets_dict


def get_joint_dataloader(data_root='./data', subset_size=2000):
    """Create a joint MNIST+Fashion dataloader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_train = datasets.MNIST(data_root, train=True, download=True, transform=transform)
    fashion_train = datasets.FashionMNIST(data_root, train=True, download=True, transform=transform)

    if subset_size:
        mnist_train = Subset(mnist_train, range(min(subset_size, len(mnist_train))))
        fashion_train = Subset(fashion_train, range(min(subset_size, len(fashion_train))))

    joint_dataset = ConcatDataset([mnist_train, fashion_train])
    return DataLoader(joint_dataset, batch_size=64, shuffle=True)


# ============================================================================
# MAIN EXPERIMENTS
# ============================================================================

def run_experiment_1_universal_geometry(loaders, device, ricci_lambda=50.0, epochs=5):
    """
    Experiment 1: Universal Geometry

    1. Train jointly on MNIST+Fashion
    2. Extract universal geometry
    3. Sequential training with universal target
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: UNIVERSAL GEOMETRY")
    print("=" * 70)

    # Step 1: Train joint model
    print("\nStep 1: Training joint model on MNIST+Fashion...")
    joint_loader = get_joint_dataloader('./data', 2000)

    joint_model = SimpleMLP()
    joint_learner = BaselineCL(joint_model, device=device)
    joint_learner.train_task(joint_loader, epochs=epochs, verbose=True)

    # Step 2: Extract universal geometry from MNIST test (has consistent labels)
    print("\nStep 2: Extracting universal geometry...")
    universal_centroids = extract_centroids(joint_model, loaders['mnist_train'], device)
    print(f"Universal centroids shape: {universal_centroids.shape}")

    # Step 3: Sequential training with universal target
    print("\nStep 3: Sequential training with universal geometry target...")

    model = SimpleMLP()
    learner = UniversalGeometryCL(model, device=device, ricci_lambda=ricci_lambda)
    learner.set_universal_geometry(universal_centroids)

    # Task A: MNIST
    print("\nTask A: MNIST")
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=epochs)
    mnist_after_a = learner.evaluate(loaders['mnist_test'])['accuracy']

    # Task B: Fashion
    print("\nTask B: FashionMNIST")
    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=epochs)
    mnist_after_b = learner.evaluate(loaders['mnist_test'])['accuracy']
    fashion_after_b = learner.evaluate(loaders['fashion_test'])['accuracy']

    print(f"\nResults:")
    print(f"  MNIST: {mnist_after_a:.1%} → {mnist_after_b:.1%}")
    print(f"  Fashion: {fashion_after_b:.1%}")

    return {
        'mnist_a': mnist_after_a,
        'mnist_b': mnist_after_b,
        'fashion_b': fashion_after_b,
        'forgetting': mnist_after_a - mnist_after_b
    }


def run_experiment_2_subspace(loaders, device, ricci_lambda=50.0, epochs=5):
    """
    Experiment 2: Subspace Orthogonalization
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: SUBSPACE ORTHOGONALIZATION")
    print("=" * 70)

    model = SimpleMLP()
    learner = SubspaceRicciCL(model, device=device, ricci_lambda=ricci_lambda, subspace_dim=32)

    # Task A: MNIST
    print("\nTask A: MNIST")
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=epochs)
    mnist_after_a = learner.evaluate(loaders['mnist_test'])['accuracy']
    learner.after_task(loaders['mnist_train'], task_id=0)

    # Task B: Fashion
    print("\nTask B: FashionMNIST")
    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=epochs)
    mnist_after_b = learner.evaluate(loaders['mnist_test'])['accuracy']
    fashion_after_b = learner.evaluate(loaders['fashion_test'])['accuracy']

    print(f"\nResults:")
    print(f"  MNIST: {mnist_after_a:.1%} → {mnist_after_b:.1%}")
    print(f"  Fashion: {fashion_after_b:.1%}")

    return {
        'mnist_a': mnist_after_a,
        'mnist_b': mnist_after_b,
        'fashion_b': fashion_after_b,
        'forgetting': mnist_after_a - mnist_after_b
    }


def run_experiment_3_three_tasks(loaders, device, ricci_lambda=50.0, epochs=5):
    """
    Experiment 3: Three-Task Sequence (MNIST → Fashion → KMNIST)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: THREE-TASK SEQUENCE")
    print("=" * 70)

    model = SimpleMLP()
    learner = CentroidRicciCL(model, device=device, ricci_lambda=ricci_lambda)

    # Task A: MNIST
    print("\nTask A: MNIST")
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=epochs)
    mnist_after_a = learner.evaluate(loaders['mnist_test'])['accuracy']
    learner.after_task(loaders['mnist_train'], task_id=0)

    # Task B: Fashion
    print("\nTask B: FashionMNIST")
    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=epochs)
    mnist_after_b = learner.evaluate(loaders['mnist_test'])['accuracy']
    fashion_after_b = learner.evaluate(loaders['fashion_test'])['accuracy']
    learner.after_task(loaders['fashion_train'], task_id=1)

    # Task C: KMNIST
    print("\nTask C: KMNIST")
    learner.train_task(loaders['kmnist_train'], loaders['kmnist_test'], epochs=epochs)
    mnist_after_c = learner.evaluate(loaders['mnist_test'])['accuracy']
    fashion_after_c = learner.evaluate(loaders['fashion_test'])['accuracy']
    kmnist_after_c = learner.evaluate(loaders['kmnist_test'])['accuracy']

    print(f"\nResults after all three tasks:")
    print(f"  MNIST: {mnist_after_a:.1%} → {mnist_after_b:.1%} → {mnist_after_c:.1%}")
    print(f"  Fashion: {fashion_after_b:.1%} → {fashion_after_c:.1%}")
    print(f"  KMNIST: {kmnist_after_c:.1%}")

    return {
        'mnist_a': mnist_after_a,
        'mnist_b': mnist_after_b,
        'mnist_c': mnist_after_c,
        'fashion_b': fashion_after_b,
        'fashion_c': fashion_after_c,
        'kmnist_c': kmnist_after_c
    }


def run_baselines(loaders, device, epochs=5):
    """Run baseline and EWC for comparison."""
    results = {}

    # Baseline (three tasks)
    print("\n" + "=" * 70)
    print("BASELINE (Three Tasks)")
    print("=" * 70)

    model = SimpleMLP()
    learner = BaselineCL(model, device=device)

    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=epochs, verbose=False)
    mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']

    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=epochs, verbose=False)
    mnist_b = learner.evaluate(loaders['mnist_test'])['accuracy']
    fashion_b = learner.evaluate(loaders['fashion_test'])['accuracy']

    learner.train_task(loaders['kmnist_train'], loaders['kmnist_test'], epochs=epochs, verbose=False)
    mnist_c = learner.evaluate(loaders['mnist_test'])['accuracy']
    fashion_c = learner.evaluate(loaders['fashion_test'])['accuracy']
    kmnist_c = learner.evaluate(loaders['kmnist_test'])['accuracy']

    print(f"MNIST: {mnist_a:.1%} → {mnist_b:.1%} → {mnist_c:.1%}")
    print(f"Fashion: {fashion_b:.1%} → {fashion_c:.1%}")
    print(f"KMNIST: {kmnist_c:.1%}")

    results['baseline'] = {
        'mnist_a': mnist_a, 'mnist_b': mnist_b, 'mnist_c': mnist_c,
        'fashion_b': fashion_b, 'fashion_c': fashion_c,
        'kmnist_c': kmnist_c
    }

    # EWC (three tasks)
    print("\n" + "=" * 70)
    print("EWC (Three Tasks)")
    print("=" * 70)

    model = SimpleMLP()
    learner = EWCCL(model, device=device, ewc_lambda=5000)

    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=epochs, verbose=False)
    mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
    learner.after_task(loaders['mnist_train'], task_id=0)

    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=epochs, verbose=False)
    mnist_b = learner.evaluate(loaders['mnist_test'])['accuracy']
    fashion_b = learner.evaluate(loaders['fashion_test'])['accuracy']
    learner.after_task(loaders['fashion_train'], task_id=1)

    learner.train_task(loaders['kmnist_train'], loaders['kmnist_test'], epochs=epochs, verbose=False)
    mnist_c = learner.evaluate(loaders['mnist_test'])['accuracy']
    fashion_c = learner.evaluate(loaders['fashion_test'])['accuracy']
    kmnist_c = learner.evaluate(loaders['kmnist_test'])['accuracy']

    print(f"MNIST: {mnist_a:.1%} → {mnist_b:.1%} → {mnist_c:.1%}")
    print(f"Fashion: {fashion_b:.1%} → {fashion_c:.1%}")
    print(f"KMNIST: {kmnist_c:.1%}")

    results['ewc'] = {
        'mnist_a': mnist_a, 'mnist_b': mnist_b, 'mnist_c': mnist_c,
        'fashion_b': fashion_b, 'fashion_c': fashion_c,
        'kmnist_c': kmnist_c
    }

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--ricci-lambda', type=float, default=50.0)
    parser.add_argument('--subset', type=int, default=2000)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', '1', '2', '3', 'baselines'])

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
    print(f"Epochs: {args.epochs}")
    print(f"Ricci λ: {args.ricci_lambda}")

    # Load all datasets
    print("\nLoading datasets...")
    loaders = get_all_datasets('./data', args.subset)

    results = {}

    # Run baselines first
    if args.experiment in ['all', 'baselines']:
        baseline_results = run_baselines(loaders, device, args.epochs)
        results.update(baseline_results)

    # Experiment 1: Universal Geometry
    if args.experiment in ['all', '1']:
        results['exp1_universal'] = run_experiment_1_universal_geometry(
            loaders, device, args.ricci_lambda, args.epochs
        )

    # Experiment 2: Subspace
    if args.experiment in ['all', '2']:
        results['exp2_subspace'] = run_experiment_2_subspace(
            loaders, device, args.ricci_lambda, args.epochs
        )

    # Experiment 3: Three Tasks
    if args.experiment in ['all', '3']:
        results['exp3_three_tasks'] = run_experiment_3_three_tasks(
            loaders, device, args.ricci_lambda, args.epochs
        )

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    if 'baseline' in results:
        b = results['baseline']
        print(f"\nBaseline (3 tasks):")
        print(f"  MNIST final: {b['mnist_c']:.1%}")
        print(f"  Fashion final: {b['fashion_c']:.1%}")
        print(f"  KMNIST final: {b['kmnist_c']:.1%}")

    if 'ewc' in results:
        e = results['ewc']
        print(f"\nEWC (3 tasks):")
        print(f"  MNIST final: {e['mnist_c']:.1%}")
        print(f"  Fashion final: {e['fashion_c']:.1%}")
        print(f"  KMNIST final: {e['kmnist_c']:.1%}")

    if 'exp1_universal' in results:
        r = results['exp1_universal']
        print(f"\nExp 1 - Universal Geometry (2 tasks):")
        print(f"  MNIST: {r['mnist_b']:.1%}")
        print(f"  Fashion: {r['fashion_b']:.1%}")

    if 'exp2_subspace' in results:
        r = results['exp2_subspace']
        print(f"\nExp 2 - Subspace (2 tasks):")
        print(f"  MNIST: {r['mnist_b']:.1%}")
        print(f"  Fashion: {r['fashion_b']:.1%}")

    if 'exp3_three_tasks' in results:
        r = results['exp3_three_tasks']
        print(f"\nExp 3 - Centroid Ricci (3 tasks):")
        print(f"  MNIST final: {r['mnist_c']:.1%}")
        print(f"  Fashion final: {r['fashion_c']:.1%}")
        print(f"  KMNIST final: {r['kmnist_c']:.1%}")

    # Comparison
    if 'ewc' in results and 'exp3_three_tasks' in results:
        e = results['ewc']
        r = results['exp3_three_tasks']

        print("\n" + "=" * 70)
        print("COMPARISON: EWC vs Centroid Ricci (3 tasks)")
        print("=" * 70)
        print(f"MNIST:   EWC {e['mnist_c']:.1%} vs Ricci {r['mnist_c']:.1%}")
        print(f"Fashion: EWC {e['fashion_c']:.1%} vs Ricci {r['fashion_c']:.1%}")
        print(f"KMNIST:  EWC {e['kmnist_c']:.1%} vs Ricci {r['kmnist_c']:.1%}")

        # Average
        ewc_avg = (e['mnist_c'] + e['fashion_c'] + e['kmnist_c']) / 3
        ricci_avg = (r['mnist_c'] + r['fashion_c'] + r['kmnist_c']) / 3
        print(f"\nAverage: EWC {ewc_avg:.1%} vs Ricci {ricci_avg:.1%}")

        if ricci_avg > ewc_avg:
            print(f"✓ RICCI WINS by +{(ricci_avg - ewc_avg):.1%}!")


if __name__ == "__main__":
    main()
