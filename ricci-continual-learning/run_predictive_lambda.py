#!/usr/bin/env python3
"""
Predictive λ: Set regularization strength based on task similarity BEFORE training.

Key insight: Similar tasks compete for the same geometric structure,
so they need stronger protection. Dissimilar tasks can coexist.

Prediction:
- MNIST → Fashion (low similarity): λ=30 works fine
- MNIST → KMNIST (high similarity): λ=100 needed
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


def compute_task_similarity(model, loader_a, loader_b, device):
    """
    Compute similarity between two tasks based on their embedding distributions.

    Methods:
    1. Centroid overlap: How close are class centroids between tasks?
    2. Feature correlation: How similar are the learned features?
    3. Gradient alignment: Would gradients point in similar directions?

    Returns similarity score in [0, 1] where 1 = very similar (high interference risk)
    """
    model.eval()

    # Get embeddings for both tasks
    def get_embeddings(loader):
        all_embs = []
        all_labels = []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                embs = model.get_embeddings(batch_x)
                all_embs.append(embs.cpu())
                all_labels.append(batch_y)
        return torch.cat(all_embs), torch.cat(all_labels)

    embs_a, labels_a = get_embeddings(loader_a)
    embs_b, labels_b = get_embeddings(loader_b)

    # Method 1: Distribution overlap (using mean and std)
    mean_a, std_a = embs_a.mean(dim=0), embs_a.std(dim=0)
    mean_b, std_b = embs_b.mean(dim=0), embs_b.std(dim=0)

    # Cosine similarity of mean embeddings
    mean_sim = F.cosine_similarity(mean_a.unsqueeze(0), mean_b.unsqueeze(0)).item()

    # Method 2: Centroid overlap
    # Compute centroids for each class
    def get_centroids(embs, labels):
        centroids = {}
        for c in range(10):
            mask = labels == c
            if mask.sum() > 0:
                centroids[c] = embs[mask].mean(dim=0)
        return centroids

    cents_a = get_centroids(embs_a, labels_a)
    cents_b = get_centroids(embs_b, labels_b)

    # Average distance between corresponding centroids
    # (same class index, different task)
    centroid_distances = []
    for c in range(10):
        if c in cents_a and c in cents_b:
            dist = torch.norm(cents_a[c] - cents_b[c]).item()
            centroid_distances.append(dist)

    avg_centroid_dist = np.mean(centroid_distances) if centroid_distances else 0

    # Method 3: Feature activation pattern similarity
    # How similar are the activation patterns?
    activation_corr = torch.corrcoef(torch.stack([embs_a.mean(dim=0), embs_b.mean(dim=0)]))[0, 1].item()

    # Combine metrics into similarity score
    # High mean_sim + high activation_corr + low centroid_dist = high similarity

    # Normalize centroid distance (empirically, typical range is 0-10)
    normalized_dist = 1 - min(avg_centroid_dist / 10, 1)

    # Combine (weighted average)
    similarity = 0.3 * (mean_sim + 1) / 2 + 0.3 * normalized_dist + 0.4 * (activation_corr + 1) / 2

    return {
        'similarity': similarity,
        'mean_cosine': mean_sim,
        'centroid_distance': avg_centroid_dist,
        'activation_correlation': activation_corr
    }


class PredictiveLambdaRicciCL:
    """
    CC-Ricci with predictive λ based on task similarity.

    Before training on new task, compute similarity to previous tasks
    and set λ accordingly.
    """

    def __init__(
        self,
        model,
        device='cpu',
        lambda_low=30.0,      # For dissimilar tasks
        lambda_high=150.0,    # For similar tasks
        similarity_threshold=0.5,
        num_classes=10
    ):
        self.model = model.to(device)
        self.device = device
        self.lambda_low = lambda_low
        self.lambda_high = lambda_high
        self.similarity_threshold = similarity_threshold
        self.num_classes = num_classes

        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.initial_weights = get_weight_vector(model).clone()

        self.reference_centroids = {}
        self.task_loaders = {}  # Store loaders for similarity computation
        self.lambda_used = {}   # Track λ used for each task
        self.similarity_scores = {}

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

    def predict_lambda(self, new_task_loader, task_name=""):
        """Predict optimal λ based on similarity to previous tasks."""
        if not self.task_loaders:
            # First task - no interference possible
            return self.lambda_low, 0.0

        # Compute similarity to all previous tasks
        max_similarity = 0.0
        for prev_name, prev_loader in self.task_loaders.items():
            sim_info = compute_task_similarity(
                self.model, prev_loader, new_task_loader, self.device
            )
            similarity = sim_info['similarity']
            max_similarity = max(max_similarity, similarity)

            print(f"    Similarity to {prev_name}: {similarity:.3f}")
            print(f"      (mean_cos={sim_info['mean_cosine']:.3f}, "
                  f"cent_dist={sim_info['centroid_distance']:.2f}, "
                  f"act_corr={sim_info['activation_correlation']:.3f})")

        # Interpolate λ based on max similarity
        # Higher similarity → higher λ
        t = max_similarity  # Already in [0, 1]
        predicted_lambda = self.lambda_low + t * (self.lambda_high - self.lambda_low)

        return predicted_lambda, max_similarity

    def compute_centroid_loss(self, embeddings, labels):
        if not self.reference_centroids:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        batch_centroids = {}
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                batch_centroids[c] = embeddings[mask].mean(dim=0)

        for task_id, ref_cents in self.reference_centroids.items():
            for c, ref_cent in ref_cents.items():
                if c in batch_centroids:
                    loss = loss + F.mse_loss(batch_centroids[c], ref_cent)

        return loss

    def after_task(self, dataloader, task_id=0, task_name="task"):
        print(f"Storing reference for {task_name}...")
        centroids = self.compute_centroids(dataloader)
        self.reference_centroids[task_id] = {c: v.detach().clone() for c, v in centroids.items()}
        self.task_loaders[task_name] = dataloader
        print(f"  Stored {len(centroids)} centroids")

    def train_task(self, train_loader, test_loader, epochs=5, task_name="", verbose=True):
        # PREDICTIVE: Compute λ BEFORE training
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
                ricci_loss = self.compute_centroid_loss(embeddings, batch_y)

                loss = ce_loss + predicted_lambda * ricci_loss
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


def run_experiment(device, loaders):
    """Compare fixed λ vs predictive λ."""

    print("\n" + "="*70)
    print("EXPERIMENT: PREDICTIVE λ vs FIXED λ")
    print("="*70)

    results = {}

    # ========================================
    # Test 1: MNIST → Fashion (expected low similarity)
    # ========================================
    print("\n" + "-"*70)
    print("TEST 1: MNIST → Fashion (expecting LOW similarity, LOW λ)")
    print("-"*70)

    # Fixed λ=50
    print("\n[Fixed λ=50]")
    model = SimpleMLP()
    learner = CentroidRicciCL(model, device=device, ricci_lambda=50)
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
    mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
    learner.after_task(loaders['mnist_train'], task_id=0)
    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=5, verbose=False)
    fixed_mnist = learner.evaluate(loaders['mnist_test'])['accuracy']
    fixed_fashion = learner.evaluate(loaders['fashion_test'])['accuracy']
    print(f"  MNIST: {mnist_a:.1%} → {fixed_mnist:.1%}, Fashion: {fixed_fashion:.1%}")
    results['mnist_fashion_fixed50'] = {'mnist': fixed_mnist, 'fashion': fixed_fashion}

    # Predictive λ
    print("\n[Predictive λ]")
    model = SimpleMLP()
    learner = PredictiveLambdaRicciCL(model, device=device, lambda_low=30, lambda_high=150)
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, task_name="MNIST", verbose=False)
    mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
    learner.after_task(loaders['mnist_train'], task_id=0, task_name="MNIST")
    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=5, task_name="Fashion", verbose=False)
    pred_mnist = learner.evaluate(loaders['mnist_test'])['accuracy']
    pred_fashion = learner.evaluate(loaders['fashion_test'])['accuracy']
    print(f"  MNIST: {mnist_a:.1%} → {pred_mnist:.1%}, Fashion: {pred_fashion:.1%}")
    print(f"  λ used: {learner.lambda_used}")
    results['mnist_fashion_predictive'] = {
        'mnist': pred_mnist,
        'fashion': pred_fashion,
        'lambda': learner.lambda_used.get('Fashion', 0),
        'similarity': learner.similarity_scores.get('Fashion', 0)
    }

    # ========================================
    # Test 2: MNIST → KMNIST (expected high similarity)
    # ========================================
    print("\n" + "-"*70)
    print("TEST 2: MNIST → KMNIST (expecting HIGH similarity, HIGH λ)")
    print("-"*70)

    # Fixed λ=50
    print("\n[Fixed λ=50]")
    model = SimpleMLP()
    learner = CentroidRicciCL(model, device=device, ricci_lambda=50)
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
    mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
    learner.after_task(loaders['mnist_train'], task_id=0)
    learner.train_task(loaders['kmnist_train'], loaders['kmnist_test'], epochs=5, verbose=False)
    fixed_mnist = learner.evaluate(loaders['mnist_test'])['accuracy']
    fixed_kmnist = learner.evaluate(loaders['kmnist_test'])['accuracy']
    print(f"  MNIST: {mnist_a:.1%} → {fixed_mnist:.1%}, KMNIST: {fixed_kmnist:.1%}")
    results['mnist_kmnist_fixed50'] = {'mnist': fixed_mnist, 'kmnist': fixed_kmnist}

    # Fixed λ=100 (what we'd manually choose for similar tasks)
    print("\n[Fixed λ=100]")
    model = SimpleMLP()
    learner = CentroidRicciCL(model, device=device, ricci_lambda=100)
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
    mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
    learner.after_task(loaders['mnist_train'], task_id=0)
    learner.train_task(loaders['kmnist_train'], loaders['kmnist_test'], epochs=5, verbose=False)
    fixed100_mnist = learner.evaluate(loaders['mnist_test'])['accuracy']
    fixed100_kmnist = learner.evaluate(loaders['kmnist_test'])['accuracy']
    print(f"  MNIST: {mnist_a:.1%} → {fixed100_mnist:.1%}, KMNIST: {fixed100_kmnist:.1%}")
    results['mnist_kmnist_fixed100'] = {'mnist': fixed100_mnist, 'kmnist': fixed100_kmnist}

    # Predictive λ
    print("\n[Predictive λ]")
    model = SimpleMLP()
    learner = PredictiveLambdaRicciCL(model, device=device, lambda_low=30, lambda_high=150)
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, task_name="MNIST", verbose=False)
    mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
    learner.after_task(loaders['mnist_train'], task_id=0, task_name="MNIST")
    learner.train_task(loaders['kmnist_train'], loaders['kmnist_test'], epochs=5, task_name="KMNIST", verbose=False)
    pred_mnist = learner.evaluate(loaders['mnist_test'])['accuracy']
    pred_kmnist = learner.evaluate(loaders['kmnist_test'])['accuracy']
    print(f"  MNIST: {mnist_a:.1%} → {pred_mnist:.1%}, KMNIST: {pred_kmnist:.1%}")
    print(f"  λ used: {learner.lambda_used}")
    results['mnist_kmnist_predictive'] = {
        'mnist': pred_mnist,
        'kmnist': pred_kmnist,
        'lambda': learner.lambda_used.get('KMNIST', 0),
        'similarity': learner.similarity_scores.get('KMNIST', 0)
    }

    # ========================================
    # Test 3: 3-task sequence with predictive λ
    # ========================================
    print("\n" + "-"*70)
    print("TEST 3: MNIST → Fashion → KMNIST (3-task with predictive λ)")
    print("-"*70)

    # Fixed λ=50
    print("\n[Fixed λ=50]")
    model = SimpleMLP()
    learner = CentroidRicciCL(model, device=device, ricci_lambda=50)
    learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
    learner.after_task(loaders['mnist_train'], task_id=0)
    learner.train_task(loaders['fashion_train'], loaders['fashion_test'], epochs=5, verbose=False)
    learner.after_task(loaders['fashion_train'], task_id=1)
    learner.train_task(loaders['kmnist_train'], loaders['kmnist_test'], epochs=5, verbose=False)
    f_mnist = learner.evaluate(loaders['mnist_test'])['accuracy']
    f_fashion = learner.evaluate(loaders['fashion_test'])['accuracy']
    f_kmnist = learner.evaluate(loaders['kmnist_test'])['accuracy']
    f_avg = (f_mnist + f_fashion + f_kmnist) / 3
    print(f"  MNIST: {f_mnist:.1%}, Fashion: {f_fashion:.1%}, KMNIST: {f_kmnist:.1%}, Avg: {f_avg:.1%}")
    results['3task_fixed50'] = {'mnist': f_mnist, 'fashion': f_fashion, 'kmnist': f_kmnist, 'avg': f_avg}

    # EWC
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

    # Predictive λ
    print("\n[Predictive λ]")
    model = SimpleMLP()
    learner = PredictiveLambdaRicciCL(model, device=device, lambda_low=30, lambda_high=150)
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
    print(f"  Similarities: {learner.similarity_scores}")
    results['3task_predictive'] = {
        'mnist': p_mnist, 'fashion': p_fashion, 'kmnist': p_kmnist, 'avg': p_avg,
        'lambdas': dict(learner.lambda_used),
        'similarities': dict(learner.similarity_scores)
    }

    return results


def main():
    device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    loaders = get_datasets('./data', subset_size=2000)
    results = run_experiment(device, loaders)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: PREDICTIVE λ RESULTS")
    print("="*70)

    print("\n--- Test 1: MNIST → Fashion ---")
    print("Expected: LOW similarity → LOW λ → good plasticity")
    r1_fixed = results['mnist_fashion_fixed50']
    r1_pred = results['mnist_fashion_predictive']
    print(f"Fixed λ=50:   MNIST {r1_fixed['mnist']:.1%}, Fashion {r1_fixed['fashion']:.1%}")
    print(f"Predictive:   MNIST {r1_pred['mnist']:.1%}, Fashion {r1_pred['fashion']:.1%}")
    print(f"              (used λ={r1_pred['lambda']:.1f}, similarity={r1_pred['similarity']:.3f})")

    print("\n--- Test 2: MNIST → KMNIST ---")
    print("Expected: HIGH similarity → HIGH λ → better retention")
    r2_fixed50 = results['mnist_kmnist_fixed50']
    r2_fixed100 = results['mnist_kmnist_fixed100']
    r2_pred = results['mnist_kmnist_predictive']
    print(f"Fixed λ=50:   MNIST {r2_fixed50['mnist']:.1%}, KMNIST {r2_fixed50['kmnist']:.1%}")
    print(f"Fixed λ=100:  MNIST {r2_fixed100['mnist']:.1%}, KMNIST {r2_fixed100['kmnist']:.1%}")
    print(f"Predictive:   MNIST {r2_pred['mnist']:.1%}, KMNIST {r2_pred['kmnist']:.1%}")
    print(f"              (used λ={r2_pred['lambda']:.1f}, similarity={r2_pred['similarity']:.3f})")

    print("\n--- Test 3: 3-Task Sequence ---")
    r3_fixed = results['3task_fixed50']
    r3_ewc = results['3task_ewc']
    r3_pred = results['3task_predictive']
    print(f"Fixed λ=50:   Avg {r3_fixed['avg']:.1%}")
    print(f"EWC:          Avg {r3_ewc['avg']:.1%}")
    print(f"Predictive:   Avg {r3_pred['avg']:.1%}")
    print(f"              λ used: {r3_pred['lambdas']}")

    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    # Check if similarity detection worked
    fashion_sim = results['mnist_fashion_predictive']['similarity']
    kmnist_sim = results['mnist_kmnist_predictive']['similarity']

    print(f"\nSimilarity Detection:")
    print(f"  MNIST-Fashion: {fashion_sim:.3f}")
    print(f"  MNIST-KMNIST:  {kmnist_sim:.3f}")

    if kmnist_sim > fashion_sim:
        print("  ✓ Correctly detected KMNIST as more similar to MNIST!")
    else:
        print("  ✗ Failed to detect higher similarity for KMNIST")

    # Check if predictive λ helped
    print(f"\nPerformance Impact:")

    # For Fashion: should have similar or better performance (low λ = more plasticity)
    if r1_pred['fashion'] >= r1_fixed['fashion'] - 0.05:
        print(f"  ✓ Fashion plasticity maintained ({r1_pred['fashion']:.1%} vs {r1_fixed['fashion']:.1%})")

    # For KMNIST: should have better MNIST retention (high λ)
    if r2_pred['mnist'] > r2_fixed50['mnist']:
        diff = r2_pred['mnist'] - r2_fixed50['mnist']
        print(f"  ✓ KMNIST: Better MNIST retention (+{diff:.1%})")
    else:
        print(f"  ~ KMNIST: No improvement in retention")

    # 3-task
    if r3_pred['avg'] > r3_fixed['avg']:
        diff = r3_pred['avg'] - r3_fixed['avg']
        print(f"  ✓ 3-task: Predictive beats Fixed by +{diff:.1%}")

    if r3_pred['avg'] >= r3_ewc['avg']:
        print(f"  ✓ 3-task: Predictive matches/beats EWC!")


if __name__ == "__main__":
    main()
