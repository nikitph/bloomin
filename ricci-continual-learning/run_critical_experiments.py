#!/usr/bin/env python3
"""
Critical Experiments for Ricci-REWA Validation

(A) Angle-only vs distance-only ablation
(B) Explicit curvature measurement on centroid simplex
(C) Replay comparison at equal compute
(D) CIFAR-100 hierarchical (separate script)
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
from scipy import stats
import json
from datetime import datetime

from src.models import SimpleMLP, get_weight_vector
from src.continual_learning import BaselineCL, EWCCL


# =============================================================================
# EXPERIMENT A: Angle-only vs Distance-only Ablation
# =============================================================================

class DistanceOnlyRicciCL:
    """Preserve only inter-centroid DISTANCES (not angles)."""

    def __init__(self, model, device='cpu', ricci_lambda=50.0):
        self.model = model.to(device)
        self.device = device
        self.ricci_lambda = ricci_lambda
        self.reference_distances = None
        self.reference_centroids = None

    def compute_centroids(self, dataloader):
        """Compute class centroids from embeddings."""
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

    def compute_distance_matrix(self, centroids):
        """Compute pairwise distances between centroids."""
        classes = sorted(centroids.keys())
        n = len(classes)
        distances = torch.zeros(n, n, device=self.device)

        for i, ci in enumerate(classes):
            for j, cj in enumerate(classes):
                if i < j:
                    d = torch.norm(centroids[ci] - centroids[cj])
                    distances[i, j] = d
                    distances[j, i] = d
        return distances

    def after_task(self, dataloader, task_id=0):
        """Store reference distances after task."""
        centroids = self.compute_centroids(dataloader)
        self.reference_centroids = {c: v.clone() for c, v in centroids.items()}
        self.reference_distances = self.compute_distance_matrix(centroids)

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

                # Distance preservation loss
                ricci_loss = torch.tensor(0.0, device=self.device)
                if self.reference_distances is not None:
                    current_centroids = self.compute_centroids(train_loader)
                    current_distances = self.compute_distance_matrix(current_centroids)

                    # Only preserve distances, normalize to be scale-invariant
                    ref_norm = self.reference_distances / (self.reference_distances.max() + 1e-8)
                    curr_norm = current_distances / (current_distances.max() + 1e-8)
                    ricci_loss = F.mse_loss(curr_norm, ref_norm)

                loss = ce_loss + self.ricci_lambda * ricci_loss
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


class AngleOnlyRicciCL:
    """Preserve only inter-centroid ANGLES (not distances)."""

    def __init__(self, model, device='cpu', ricci_lambda=50.0):
        self.model = model.to(device)
        self.device = device
        self.ricci_lambda = ricci_lambda
        self.reference_angles = None
        self.reference_centroids = None

    def compute_centroids(self, dataloader):
        """Compute class centroids from embeddings."""
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

    def compute_angle_matrix(self, centroids):
        """Compute pairwise cosine similarities (angles) between centroids."""
        classes = sorted(centroids.keys())
        n = len(classes)
        angles = torch.zeros(n, n, device=self.device)

        for i, ci in enumerate(classes):
            for j, cj in enumerate(classes):
                if i != j:
                    # Cosine similarity
                    cos_sim = F.cosine_similarity(
                        centroids[ci].unsqueeze(0),
                        centroids[cj].unsqueeze(0)
                    )
                    angles[i, j] = cos_sim
        return angles

    def after_task(self, dataloader, task_id=0):
        """Store reference angles after task."""
        centroids = self.compute_centroids(dataloader)
        self.reference_centroids = {c: v.clone() for c, v in centroids.items()}
        self.reference_angles = self.compute_angle_matrix(centroids)

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

                # Angle preservation loss
                ricci_loss = torch.tensor(0.0, device=self.device)
                if self.reference_angles is not None:
                    current_centroids = self.compute_centroids(train_loader)
                    current_angles = self.compute_angle_matrix(current_centroids)
                    ricci_loss = F.mse_loss(current_angles, self.reference_angles)

                loss = ce_loss + self.ricci_lambda * ricci_loss
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


class BothRicciCL:
    """Preserve BOTH distances and angles (full geometry)."""

    def __init__(self, model, device='cpu', ricci_lambda=50.0):
        self.model = model.to(device)
        self.device = device
        self.ricci_lambda = ricci_lambda
        self.reference_centroids = None

    def compute_centroids(self, dataloader):
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

    def after_task(self, dataloader, task_id=0):
        centroids = self.compute_centroids(dataloader)
        self.reference_centroids = {c: v.clone() for c, v in centroids.items()}

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

                ricci_loss = torch.tensor(0.0, device=self.device)
                if self.reference_centroids is not None:
                    current_centroids = self.compute_centroids(train_loader)

                    # Direct centroid preservation (both position = distance + angle)
                    for c in self.reference_centroids:
                        if c in current_centroids:
                            ricci_loss += F.mse_loss(
                                current_centroids[c],
                                self.reference_centroids[c]
                            )

                loss = ce_loss + self.ricci_lambda * ricci_loss
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


# =============================================================================
# EXPERIMENT B: Explicit Curvature Measurement
# =============================================================================

def compute_sectional_curvature(centroids):
    """
    Compute sectional curvature on the centroid simplex.

    For a simplex of K centroids in R^d, we compute curvature by measuring
    how the geometry deviates from flat (Euclidean) space.

    Method: Compare actual geodesic distances to those predicted by
    flat geometry using the Cayley-Menger determinant approach.
    """
    classes = sorted(centroids.keys())
    K = len(classes)

    if K < 3:
        return {'mean_curvature': 0.0, 'curvature_per_triple': {}}

    # Compute all pairwise distances
    distances = {}
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            if i < j:
                d = torch.norm(centroids[ci] - centroids[cj]).item()
                distances[(ci, cj)] = d
                distances[(cj, ci)] = d

    # For each triple of centroids, compute local curvature
    # Using Gaussian curvature approximation via angle deficit
    curvatures = {}

    for i in range(K):
        for j in range(i+1, K):
            for k in range(j+1, K):
                ci, cj, ck = classes[i], classes[j], classes[k]

                # Triangle sides
                a = distances[(cj, ck)]
                b = distances[(ci, ck)]
                c = distances[(ci, cj)]

                # Compute angles using law of cosines
                # cos(A) = (b² + c² - a²) / (2bc)
                try:
                    cos_A = (b**2 + c**2 - a**2) / (2*b*c + 1e-8)
                    cos_B = (a**2 + c**2 - b**2) / (2*a*c + 1e-8)
                    cos_C = (a**2 + b**2 - c**2) / (2*a*b + 1e-8)

                    # Clamp to valid range
                    cos_A = max(-1, min(1, cos_A))
                    cos_B = max(-1, min(1, cos_B))
                    cos_C = max(-1, min(1, cos_C))

                    angle_A = np.arccos(cos_A)
                    angle_B = np.arccos(cos_B)
                    angle_C = np.arccos(cos_C)

                    # Angle sum (should be π for flat space)
                    angle_sum = angle_A + angle_B + angle_C

                    # Angle deficit = deviation from flat
                    # Positive = positive curvature (spherical)
                    # Negative = negative curvature (hyperbolic)
                    angle_deficit = angle_sum - np.pi

                    # Area of triangle (Heron's formula)
                    s = (a + b + c) / 2
                    area_sq = s * (s-a) * (s-b) * (s-c)
                    area = np.sqrt(max(0, area_sq))

                    # Gaussian curvature ≈ angle_deficit / area
                    if area > 1e-8:
                        curvature = angle_deficit / area
                    else:
                        curvature = 0.0

                    curvatures[(ci, cj, ck)] = curvature

                except:
                    curvatures[(ci, cj, ck)] = 0.0

    mean_curvature = np.mean(list(curvatures.values())) if curvatures else 0.0

    return {
        'mean_curvature': mean_curvature,
        'curvature_per_triple': curvatures,
        'num_triples': len(curvatures)
    }


class CurvatureTracker:
    """Track curvature evolution during training."""

    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.curvature_history = []

    def compute_centroids(self, dataloader):
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

    def snapshot(self, dataloader, label=""):
        centroids = self.compute_centroids(dataloader)
        curvature_info = compute_sectional_curvature(centroids)
        curvature_info['label'] = label
        self.curvature_history.append(curvature_info)
        return curvature_info


# =============================================================================
# EXPERIMENT C: Replay Comparison at Equal Compute
# =============================================================================

class ReplayCL:
    """Experience Replay baseline."""

    def __init__(self, model, device='cpu', buffer_size=500):
        self.model = model.to(device)
        self.device = device
        self.buffer_size = buffer_size
        self.replay_buffer = []  # List of (x, y) tuples

    def add_to_buffer(self, dataloader):
        """Add samples from dataloader to replay buffer."""
        samples = []
        for batch_x, batch_y in dataloader:
            for x, y in zip(batch_x, batch_y):
                samples.append((x.clone(), y.clone()))

        # Random subsample if too many
        if len(samples) > self.buffer_size:
            indices = np.random.choice(len(samples), self.buffer_size, replace=False)
            samples = [samples[i] for i in indices]

        # Add to buffer (replace oldest if full)
        space_left = self.buffer_size - len(self.replay_buffer)
        if space_left >= len(samples):
            self.replay_buffer.extend(samples)
        else:
            # Remove oldest, add new
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

                # Current task loss
                outputs = self.model(batch_x)
                ce_loss = criterion(outputs, batch_y)

                # Replay loss (if buffer has samples)
                replay_loss = torch.tensor(0.0, device=self.device)
                if len(self.replay_buffer) > 0:
                    # Sample from buffer
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


# =============================================================================
# Main Experiments
# =============================================================================

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


def run_ablation_experiment(device, loaders, ricci_lambda=50):
    """Experiment A: Angle vs Distance ablation."""
    print("\n" + "="*70)
    print("EXPERIMENT A: ANGLE-ONLY vs DISTANCE-ONLY ABLATION")
    print("="*70)

    results = {}

    methods = [
        ("Distance-Only", DistanceOnlyRicciCL),
        ("Angle-Only", AngleOnlyRicciCL),
        ("Both (Full)", BothRicciCL),
    ]

    for name, cls in methods:
        print(f"\n--- {name} (λ={ricci_lambda}) ---")
        model = SimpleMLP()
        learner = cls(model, device=device, ricci_lambda=ricci_lambda)

        # Task A: MNIST
        learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
        mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
        learner.after_task(loaders['mnist_train'], task_id=0)

        # Task B: Fashion
        learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=5, verbose=False)
        mnist_b = learner.evaluate(loaders['mnist_test'])['accuracy']
        fashion_b = learner.evaluate(loaders['fashion_test'])['accuracy']

        print(f"  MNIST: {mnist_a:.1%} → {mnist_b:.1%}")
        print(f"  Fashion: {fashion_b:.1%}")

        results[name] = {
            'mnist_a': mnist_a,
            'mnist_b': mnist_b,
            'fashion_b': fashion_b,
            'forgetting': mnist_a - mnist_b
        }

    # Analysis
    print("\n" + "-"*50)
    print("ABLATION SUMMARY")
    print("-"*50)
    print(f"{'Method':<20} {'MNIST Ret.':<12} {'Fashion':<12} {'Forgetting':<12}")
    print("-"*50)
    for name, r in results.items():
        print(f"{name:<20} {r['mnist_b']:.1%}        {r['fashion_b']:.1%}        {r['forgetting']:.1%}")

    # Determine winner
    best = max(results.items(), key=lambda x: x[1]['mnist_b'])
    print(f"\nBest method: {best[0]} with {best[1]['mnist_b']:.1%} MNIST retention")

    if results['Angle-Only']['mnist_b'] > results['Distance-Only']['mnist_b']:
        print(">>> ANGLES matter more for decision boundaries!")
    else:
        print(">>> DISTANCES matter more for decision boundaries!")

    return results


def run_curvature_experiment(device, loaders):
    """Experiment B: Explicit curvature measurement."""
    print("\n" + "="*70)
    print("EXPERIMENT B: EXPLICIT CURVATURE MEASUREMENT")
    print("="*70)

    results = {}

    for method_name, learner_cls, kwargs in [
        ("Baseline", BaselineCL, {}),
        ("EWC", EWCCL, {'ewc_lambda': 5000}),
        ("CC-Ricci", BothRicciCL, {'ricci_lambda': 50}),
    ]:
        print(f"\n--- {method_name} ---")
        model = SimpleMLP()
        learner = learner_cls(model, device=device, **kwargs)
        tracker = CurvatureTracker(model, device)

        # Initial curvature (random weights)
        c0 = tracker.snapshot(loaders['mnist_train'], "initial")
        print(f"  Initial curvature: {c0['mean_curvature']:.6f}")

        # After Task A
        learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
        mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
        c1 = tracker.snapshot(loaders['mnist_train'], "after_mnist")
        print(f"  After MNIST: curvature={c1['mean_curvature']:.6f}, acc={mnist_a:.1%}")

        if hasattr(learner, 'after_task'):
            learner.after_task(loaders['mnist_train'], task_id=0)

        # After Task B
        learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=5, verbose=False)
        mnist_b = learner.evaluate(loaders['mnist_test'])['accuracy']
        fashion_b = learner.evaluate(loaders['fashion_test'])['accuracy']
        c2 = tracker.snapshot(loaders['mnist_train'], "after_fashion")
        print(f"  After Fashion: curvature={c2['mean_curvature']:.6f}, MNIST={mnist_b:.1%}")

        # Curvature change
        curvature_spike = abs(c2['mean_curvature'] - c1['mean_curvature'])
        forgetting = mnist_a - mnist_b

        results[method_name] = {
            'curvature_initial': c0['mean_curvature'],
            'curvature_after_a': c1['mean_curvature'],
            'curvature_after_b': c2['mean_curvature'],
            'curvature_spike': curvature_spike,
            'forgetting': forgetting,
            'mnist_b': mnist_b,
            'fashion_b': fashion_b
        }

    # Correlation analysis
    print("\n" + "-"*50)
    print("CURVATURE vs FORGETTING CORRELATION")
    print("-"*50)

    spikes = [r['curvature_spike'] for r in results.values()]
    forgettings = [r['forgetting'] for r in results.values()]

    if len(spikes) >= 3:
        corr, p_val = stats.pearsonr(spikes, forgettings)
        print(f"Pearson r: {corr:.4f} (p={p_val:.4f})")

        if corr > 0.5:
            print(">>> Forgetting CORRELATES with curvature spikes!")
        elif corr < -0.5:
            print(">>> Forgetting ANTI-correlates with curvature spikes!")

    print(f"\n{'Method':<15} {'Curv. Spike':<15} {'Forgetting':<15}")
    print("-"*50)
    for name, r in results.items():
        print(f"{name:<15} {r['curvature_spike']:.6f}       {r['forgetting']:.1%}")

    return results


def run_replay_comparison(device, loaders):
    """Experiment C: Replay vs CC-Ricci at equal compute."""
    print("\n" + "="*70)
    print("EXPERIMENT C: REPLAY vs CC-RICCI (EQUAL COMPUTE)")
    print("="*70)

    results = {}

    # Note: We compare at "equal compute" meaning same number of forward/backward passes
    # Replay stores samples and replays them, CC-Ricci computes centroid loss
    # Both add ~1x overhead per batch

    for name, cls, kwargs in [
        ("Baseline", BaselineCL, {}),
        ("Replay-500", ReplayCL, {'buffer_size': 500}),
        ("Replay-1000", ReplayCL, {'buffer_size': 1000}),
        ("CC-Ricci (λ=50)", BothRicciCL, {'ricci_lambda': 50}),
    ]:
        print(f"\n--- {name} ---")
        model = SimpleMLP()
        learner = cls(model, device=device, **kwargs)

        # Task A
        learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
        mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
        if hasattr(learner, 'after_task'):
            learner.after_task(loaders['mnist_train'], task_id=0)

        # Task B
        learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=5, verbose=False)
        mnist_b = learner.evaluate(loaders['mnist_test'])['accuracy']
        fashion_b = learner.evaluate(loaders['fashion_test'])['accuracy']

        print(f"  MNIST: {mnist_a:.1%} → {mnist_b:.1%}")
        print(f"  Fashion: {fashion_b:.1%}")

        results[name] = {
            'mnist_a': mnist_a,
            'mnist_b': mnist_b,
            'fashion_b': fashion_b,
            'forgetting': mnist_a - mnist_b
        }

    # Summary
    print("\n" + "-"*50)
    print("REPLAY vs CC-RICCI COMPARISON")
    print("-"*50)
    print(f"{'Method':<20} {'MNIST Ret.':<12} {'Fashion':<12} {'Forgetting':<12}")
    print("-"*50)
    for name, r in results.items():
        print(f"{name:<20} {r['mnist_b']:.1%}        {r['fashion_b']:.1%}        {r['forgetting']:.1%}")

    # Comparison
    cc_ricci = results['CC-Ricci (λ=50)']
    replay_1000 = results['Replay-1000']

    print("\n" + "-"*50)
    if cc_ricci['mnist_b'] > replay_1000['mnist_b']:
        diff = cc_ricci['mnist_b'] - replay_1000['mnist_b']
        print(f">>> CC-RICCI beats Replay-1000 by +{diff:.1%}!")
        print(">>> No sample storage needed!")
    else:
        diff = replay_1000['mnist_b'] - cc_ricci['mnist_b']
        print(f">>> Replay-1000 beats CC-Ricci by +{diff:.1%}")

    return results


def main():
    device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    loaders = get_datasets('./data', subset_size=2000)

    all_results = {}

    # Run all experiments
    all_results['ablation'] = run_ablation_experiment(device, loaders)
    all_results['curvature'] = run_curvature_experiment(device, loaders)
    all_results['replay'] = run_replay_comparison(device, loaders)

    # Final summary
    print("\n" + "="*70)
    print("CRITICAL EXPERIMENTS SUMMARY")
    print("="*70)

    print("\n(A) ABLATION: Which geometric property matters more?")
    ablation = all_results['ablation']
    if ablation['Angle-Only']['mnist_b'] > ablation['Distance-Only']['mnist_b']:
        print(f"    >>> ANGLES: {ablation['Angle-Only']['mnist_b']:.1%}")
        print(f"    >>> Distances: {ablation['Distance-Only']['mnist_b']:.1%}")
        print("    VERDICT: Angles matter more for decision boundaries")
    else:
        print(f"    >>> DISTANCES: {ablation['Distance-Only']['mnist_b']:.1%}")
        print(f"    >>> Angles: {ablation['Angle-Only']['mnist_b']:.1%}")
        print("    VERDICT: Distances matter more for decision boundaries")

    print("\n(B) CURVATURE: Does forgetting correlate with curvature spikes?")
    curvature = all_results['curvature']
    spikes = [r['curvature_spike'] for r in curvature.values()]
    forgettings = [r['forgetting'] for r in curvature.values()]
    corr, _ = stats.pearsonr(spikes, forgettings)
    print(f"    Correlation: r={corr:.4f}")
    if corr > 0.3:
        print("    VERDICT: Forgetting correlates with curvature disruption")

    print("\n(C) REPLAY: Does CC-Ricci beat replay at equal compute?")
    replay = all_results['replay']
    cc = replay['CC-Ricci (λ=50)']['mnist_b']
    rep = replay['Replay-1000']['mnist_b']
    if cc > rep:
        print(f"    >>> CC-Ricci: {cc:.1%}")
        print(f"    >>> Replay-1000: {rep:.1%}")
        print("    VERDICT: CC-Ricci beats replay (no storage needed!)")
    else:
        print(f"    >>> Replay-1000: {rep:.1%}")
        print(f"    >>> CC-Ricci: {cc:.1%}")
        print("    VERDICT: Replay still better (but requires storage)")

    # Save results
    os.makedirs('./results', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert to serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, float)):
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, tuple):
            return str(obj)
        else:
            return obj

    with open(f'./results/critical_experiments_{timestamp}.json', 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)

    print(f"\nResults saved to ./results/critical_experiments_{timestamp}.json")


if __name__ == "__main__":
    main()
