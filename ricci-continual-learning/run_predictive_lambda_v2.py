#!/usr/bin/env python3
"""
Predictive λ v2: Fix the class collision problem.

Key insight: Different tasks have different class semantics.
Don't try to align class 0 of Fashion with class 0 of MNIST.

Instead: Preserve the STRUCTURE (relative distances/angles) not absolute positions.
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


def compute_task_similarity(model, loader_a, loader_b, device):
    """Compute task similarity based on embedding distributions."""
    model.eval()

    def get_stats(loader):
        all_embs = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(device)
                embs = model.get_embeddings(batch_x)
                all_embs.append(embs.cpu())
        embs = torch.cat(all_embs)
        return embs.mean(dim=0), embs.std(dim=0), embs

    mean_a, std_a, embs_a = get_stats(loader_a)
    mean_b, std_b, embs_b = get_stats(loader_b)

    # Cosine similarity of distributions
    mean_sim = F.cosine_similarity(mean_a.unsqueeze(0), mean_b.unsqueeze(0)).item()
    std_sim = F.cosine_similarity(std_a.unsqueeze(0), std_b.unsqueeze(0)).item()

    # Activation correlation
    act_corr = torch.corrcoef(torch.stack([mean_a, mean_b]))[0, 1].item()

    # Combined similarity
    similarity = 0.4 * (mean_sim + 1) / 2 + 0.3 * (std_sim + 1) / 2 + 0.3 * (act_corr + 1) / 2

    return similarity, {'mean_sim': mean_sim, 'std_sim': std_sim, 'act_corr': act_corr}


class StructuralCentroidRegularizer(nn.Module):
    """
    Preserve centroid STRUCTURE not positions.

    Key idea: Store pairwise distances and angles between centroids.
    During training, penalize changes to these relationships.

    This way, MNIST's class structure is preserved even when learning Fashion.
    """

    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.reference_distances = None  # Pairwise distances
        self.reference_angles = None     # Pairwise cosine similarities
        self.reference_centroid_norms = None  # Centroid magnitudes

    def compute_structure(self, embeddings, labels):
        """Compute structural properties of centroids."""
        centroids = {}
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                centroids[c] = embeddings[mask].mean(dim=0)

        classes = sorted(centroids.keys())
        n = len(classes)

        # Pairwise distances
        distances = torch.zeros(n, n, device=embeddings.device)
        angles = torch.zeros(n, n, device=embeddings.device)

        for i, ci in enumerate(classes):
            for j, cj in enumerate(classes):
                if i != j:
                    distances[i, j] = torch.norm(centroids[ci] - centroids[cj])
                    angles[i, j] = F.cosine_similarity(
                        centroids[ci].unsqueeze(0),
                        centroids[cj].unsqueeze(0)
                    )

        # Centroid norms
        norms = torch.tensor([torch.norm(centroids[c]).item() for c in classes],
                            device=embeddings.device)

        return distances, angles, norms, centroids, classes

    def set_reference(self, embeddings, labels):
        """Store reference structure."""
        distances, angles, norms, centroids, classes = self.compute_structure(embeddings, labels)

        self.reference_distances = distances.detach()
        self.reference_angles = angles.detach()
        self.reference_centroid_norms = norms.detach()
        self.reference_classes = classes

    def forward(self, embeddings, labels):
        """Compute structural preservation loss."""
        if self.reference_distances is None:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        distances, angles, norms, _, classes = self.compute_structure(embeddings, labels)

        # Only compute loss for classes we have in both reference and current
        common_classes = set(classes) & set(self.reference_classes)
        if len(common_classes) < 2:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Map class indices
        ref_idx = {c: i for i, c in enumerate(self.reference_classes)}
        cur_idx = {c: i for i, c in enumerate(classes)}

        loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        count = 0

        for ci in common_classes:
            for cj in common_classes:
                if ci != cj:
                    ri, rj = ref_idx[ci], ref_idx[cj]
                    ui, uj = cur_idx[ci], cur_idx[cj]

                    # Preserve relative distances (normalized)
                    ref_d = self.reference_distances[ri, rj]
                    cur_d = distances[ui, uj]
                    if ref_d > 0:
                        loss = loss + (cur_d / ref_d - 1) ** 2  # Preserve ratio

                    # Preserve angles
                    ref_a = self.reference_angles[ri, rj]
                    cur_a = angles[ui, uj]
                    loss = loss + (cur_a - ref_a) ** 2

                    count += 1

        return loss / max(count, 1)


class PredictiveLambdaStructuralCL:
    """
    CC-Ricci with structural preservation and predictive λ.
    """

    def __init__(
        self,
        model,
        device='cpu',
        lambda_low=30.0,
        lambda_high=150.0,
        num_classes=10
    ):
        self.model = model.to(device)
        self.device = device
        self.lambda_low = lambda_low
        self.lambda_high = lambda_high
        self.num_classes = num_classes

        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        self.regularizers = {}  # task_id -> regularizer
        self.task_loaders = {}
        self.lambda_used = {}
        self.similarity_scores = {}

    def predict_lambda(self, new_task_loader, task_name=""):
        """Predict λ based on task similarity."""
        if not self.task_loaders:
            return self.lambda_low, 0.0

        max_similarity = 0.0
        for prev_name, prev_loader in self.task_loaders.items():
            similarity, details = compute_task_similarity(
                self.model, prev_loader, new_task_loader, self.device
            )
            max_similarity = max(max_similarity, similarity)
            print(f"    Similarity to {prev_name}: {similarity:.3f}")

        predicted_lambda = self.lambda_low + max_similarity * (self.lambda_high - self.lambda_low)
        return predicted_lambda, max_similarity

    def after_task(self, dataloader, task_id=0, task_name=""):
        """Store structural reference."""
        print(f"Storing structural reference for {task_name}...")

        self.model.eval()
        all_embs = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                embs = self.model.get_embeddings(batch_x)
                all_embs.append(embs)
                all_labels.append(batch_y)

        all_embs = torch.cat(all_embs)
        all_labels = torch.cat(all_labels).to(self.device)

        reg = StructuralCentroidRegularizer(num_classes=self.num_classes)
        reg.set_reference(all_embs, all_labels)
        self.regularizers[task_id] = reg
        self.task_loaders[task_name] = dataloader

    def compute_structural_loss(self, embeddings, labels):
        """Compute total structural preservation loss."""
        if not self.regularizers:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        for reg in self.regularizers.values():
            loss = loss + reg(embeddings, labels)

        return loss

    def train_task(self, train_loader, test_loader, epochs=5, task_name="", verbose=True):
        print(f"\n  Predicting optimal λ for {task_name}...")
        predicted_lambda, similarity = self.predict_lambda(train_loader, task_name)
        print(f"  → Using λ = {predicted_lambda:.1f} (similarity = {similarity:.3f})")

        self.lambda_used[task_name] = predicted_lambda
        self.similarity_scores[task_name] = similarity

        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                logits, embeddings = self.model.forward_with_embeddings(batch_x)

                ce_loss = criterion(logits, batch_y)
                struct_loss = self.compute_structural_loss(embeddings, batch_y)

                loss = ce_loss + predicted_lambda * struct_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()

            if verbose:
                acc = self.evaluate(test_loader)['accuracy']
                print(f"    Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.1%}")

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


def main():
    device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    loaders = get_datasets('./data', subset_size=2000)

    print("\n" + "="*70)
    print("PREDICTIVE λ v2: STRUCTURAL PRESERVATION")
    print("="*70)
    print("\nKey fix: Preserve centroid STRUCTURE (distances/angles), not positions.")
    print("This prevents class ID collision between different datasets.\n")

    results = {}

    # ========================================
    # Test: 2-task MNIST → Fashion
    # ========================================
    print("-"*70)
    print("TEST 1: MNIST → Fashion")
    print("-"*70)

    print("\n[Structural Predictive λ]")
    model = SimpleMLP()
    learner = PredictiveLambdaStructuralCL(model, device=device, lambda_low=30, lambda_high=150)
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, task_name="MNIST", verbose=False)
    mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
    learner.after_task(loaders['mnist_train'], task_id=0, task_name="MNIST")
    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=5, task_name="Fashion", verbose=False)
    mnist_b = learner.evaluate(loaders['mnist_test'])['accuracy']
    fashion_b = learner.evaluate(loaders['fashion_test'])['accuracy']
    print(f"  MNIST: {mnist_a:.1%} → {mnist_b:.1%}, Fashion: {fashion_b:.1%}")
    print(f"  λ used: {learner.lambda_used}")
    results['mnist_fashion_struct'] = {'mnist': mnist_b, 'fashion': fashion_b, 'lambda': learner.lambda_used}

    # ========================================
    # Test: 2-task MNIST → KMNIST
    # ========================================
    print("\n" + "-"*70)
    print("TEST 2: MNIST → KMNIST")
    print("-"*70)

    print("\n[Structural Predictive λ]")
    model = SimpleMLP()
    learner = PredictiveLambdaStructuralCL(model, device=device, lambda_low=30, lambda_high=150)
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, task_name="MNIST", verbose=False)
    mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
    learner.after_task(loaders['mnist_train'], task_id=0, task_name="MNIST")
    learner.train_task(loaders['kmnist_train'], loaders['kmnist_test'], epochs=5, task_name="KMNIST", verbose=False)
    mnist_b = learner.evaluate(loaders['mnist_test'])['accuracy']
    kmnist_b = learner.evaluate(loaders['kmnist_test'])['accuracy']
    print(f"  MNIST: {mnist_a:.1%} → {mnist_b:.1%}, KMNIST: {kmnist_b:.1%}")
    print(f"  λ used: {learner.lambda_used}")
    results['mnist_kmnist_struct'] = {'mnist': mnist_b, 'kmnist': kmnist_b, 'lambda': learner.lambda_used}

    # ========================================
    # Test: 3-task sequence
    # ========================================
    print("\n" + "-"*70)
    print("TEST 3: MNIST → Fashion → KMNIST (3-task)")
    print("-"*70)

    # EWC baseline
    print("\n[EWC λ=5000]")
    model = SimpleMLP()
    learner = EWCCL(model, device=device, ewc_lambda=5000)
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
    learner.after_task(loaders['mnist_train'], task_id=0)
    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=5, verbose=False)
    learner.after_task(loaders['fashion_train'], task_id=1)
    learner.train_task(loaders['kmnist_train'], loaders['kmnist_test'], epochs=5, verbose=False)
    e_mnist = learner.evaluate(loaders['mnist_test'])['accuracy']
    e_fashion = learner.evaluate(loaders['fashion_test'])['accuracy']
    e_kmnist = learner.evaluate(loaders['kmnist_test'])['accuracy']
    e_avg = (e_mnist + e_fashion + e_kmnist) / 3
    print(f"  MNIST: {e_mnist:.1%}, Fashion: {e_fashion:.1%}, KMNIST: {e_kmnist:.1%}, Avg: {e_avg:.1%}")
    results['3task_ewc'] = {'mnist': e_mnist, 'fashion': e_fashion, 'kmnist': e_kmnist, 'avg': e_avg}

    # Structural Predictive
    print("\n[Structural Predictive λ]")
    model = SimpleMLP()
    learner = PredictiveLambdaStructuralCL(model, device=device, lambda_low=30, lambda_high=150)
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, task_name="MNIST", verbose=False)
    learner.after_task(loaders['mnist_train'], task_id=0, task_name="MNIST")
    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=5, task_name="Fashion", verbose=False)
    learner.after_task(loaders['fashion_train'], task_id=1, task_name="Fashion")
    learner.train_task(loaders['kmnist_train'], loaders['kmnist_test'], epochs=5, task_name="KMNIST", verbose=False)
    p_mnist = learner.evaluate(loaders['mnist_test'])['accuracy']
    p_fashion = learner.evaluate(loaders['fashion_test'])['accuracy']
    p_kmnist = learner.evaluate(loaders['kmnist_test'])['accuracy']
    p_avg = (p_mnist + p_fashion + p_kmnist) / 3
    print(f"  MNIST: {p_mnist:.1%}, Fashion: {p_fashion:.1%}, KMNIST: {p_kmnist:.1%}, Avg: {p_avg:.1%}")
    print(f"  λ used: {learner.lambda_used}")
    results['3task_struct'] = {'mnist': p_mnist, 'fashion': p_fashion, 'kmnist': p_kmnist, 'avg': p_avg}

    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print("\n2-Task Results:")
    print(f"  MNIST→Fashion (Structural): MNIST {results['mnist_fashion_struct']['mnist']:.1%}, Fashion {results['mnist_fashion_struct']['fashion']:.1%}")
    print(f"  MNIST→KMNIST (Structural):  MNIST {results['mnist_kmnist_struct']['mnist']:.1%}, KMNIST {results['mnist_kmnist_struct']['kmnist']:.1%}")

    print(f"\n3-Task Average:")
    print(f"  EWC:        {results['3task_ewc']['avg']:.1%}")
    print(f"  Structural: {results['3task_struct']['avg']:.1%}")

    if results['3task_struct']['avg'] > results['3task_ewc']['avg']:
        diff = results['3task_struct']['avg'] - results['3task_ewc']['avg']
        print(f"\n✓ Structural Predictive beats EWC by +{diff:.1%}!")
    elif results['3task_struct']['avg'] > results['3task_ewc']['avg'] - 0.03:
        print(f"\n~ Structural Predictive matches EWC")
    else:
        print(f"\n✗ EWC still better")


if __name__ == "__main__":
    main()
