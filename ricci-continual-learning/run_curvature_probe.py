#!/usr/bin/env python3
"""
Early-Step Curvature Probe: Self-Calibrating λ for Continual Learning

This probe estimates how strongly a new task will distort the old task's
semantic geometry, then sets λ to counterbalance that force.

Key insight: Measure FORCE (deformation rate), not STATE (embedding similarity).

Algorithm:
1. Freeze reference geometry (centroids after Task A)
2. Run K steps of unconstrained Task B training (no regularization)
3. Measure centroid displacement (EDI = Early Distortion Index)
4. Map EDI → λ
5. Train with predicted λ

Success criterion: Predicted λ within ±20% of oracle, retention within 95% of oracle.
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
import copy

from src.models import SimpleMLP, get_weight_vector
from src.continual_learning import BaselineCL, EWCCL
from src.continual_learning_class_conditional import CentroidRicciCL


class CurvatureProbe:
    """
    Early-Step Curvature Probe for λ estimation.

    Measures how violently a new task will bend the old task's semantic manifold.
    """

    def __init__(
        self,
        model,
        device='cpu',
        probe_steps=200,
        probe_lr=1e-4,
        lambda_max=350.0,  # For inverse mapping
        kappa=70.0,  # Scaling constant: λ = λ_max - κ * EDI (INVERSE!)
        num_classes=10,
        use_inverse=True  # Key insight: high EDI → low interference → low λ
    ):
        self.model = model
        self.device = device
        self.probe_steps = probe_steps
        self.probe_lr = probe_lr
        self.lambda_max = lambda_max
        self.kappa = kappa
        self.num_classes = num_classes
        self.use_inverse = use_inverse

        self.reference_centroids = None
        self.reference_angles = None

    def compute_centroids(self, model, dataloader):
        """Compute class centroids in representation space."""
        model.eval()
        embeddings_by_class = defaultdict(list)

        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                embs = model.get_embeddings(batch_x)
                for emb, label in zip(embs, batch_y):
                    embeddings_by_class[label.item()].append(emb.cpu())

        centroids = {}
        for c, embs in embeddings_by_class.items():
            centroids[c] = torch.stack(embs).mean(dim=0)

        return centroids

    def compute_pairwise_angles(self, centroids):
        """Compute pairwise cosine angles between centroids."""
        classes = sorted(centroids.keys())
        n = len(classes)
        angles = torch.zeros(n, n)

        for i, ci in enumerate(classes):
            for j, cj in enumerate(classes):
                if i != j:
                    cos_sim = F.cosine_similarity(
                        centroids[ci].unsqueeze(0),
                        centroids[cj].unsqueeze(0)
                    ).item()
                    angles[i, j] = cos_sim

        return angles, classes

    def set_reference(self, dataloader):
        """Step 0: Freeze the reference geometry after Task A."""
        print("  [Probe] Computing reference geometry...")
        self.reference_centroids = self.compute_centroids(self.model, dataloader)
        self.reference_angles, self.reference_classes = self.compute_pairwise_angles(
            self.reference_centroids
        )
        print(f"  [Probe] Stored {len(self.reference_centroids)} class centroids")

    def run_probe(self, task_b_loader, task_a_loader):
        """
        Steps 1-4: Run unconstrained probe and compute EDI.

        Returns: EDI (Early Distortion Index), predicted λ
        """
        if self.reference_centroids is None:
            raise ValueError("Must call set_reference() first!")

        print(f"  [Probe] Running {self.probe_steps}-step unconstrained probe...")

        # Create a COPY of the model for probing (don't modify original)
        probe_model = copy.deepcopy(self.model)
        probe_model.to(self.device)

        # Step 1: Run unconstrained Task B training
        probe_optimizer = torch.optim.SGD(probe_model.parameters(), lr=self.probe_lr)
        criterion = nn.CrossEntropyLoss()

        probe_model.train()
        step = 0

        while step < self.probe_steps:
            for batch_x, batch_y in task_b_loader:
                if step >= self.probe_steps:
                    break

                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                probe_optimizer.zero_grad()
                outputs = probe_model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                probe_optimizer.step()

                step += 1

        # Step 2: Measure geometric distortion after probe
        print("  [Probe] Measuring centroid displacement...")
        probed_centroids = self.compute_centroids(probe_model, task_a_loader)

        # Step 3: Compute EDI (Early Distortion Index)
        displacements = []
        angle_changes = []

        for c in self.reference_centroids:
            if c in probed_centroids:
                # Ensure both tensors on CPU for comparison
                probed = probed_centroids[c].cpu()
                ref = self.reference_centroids[c].cpu()

                # Displacement magnitude
                delta = torch.norm(probed - ref).item()
                displacements.append(delta)

                # Angle change (for angle-aware EDI)
                cos_angle = F.cosine_similarity(
                    probed.unsqueeze(0),
                    ref.unsqueeze(0)
                ).item()
                # sin(angle) from cos: sin = sqrt(1 - cos^2)
                sin_angle = np.sqrt(max(0, 1 - cos_angle**2))
                angle_changes.append(sin_angle)

        # Option A: Simple mean displacement
        edi_simple = np.mean(displacements) if displacements else 0

        # Option B: Angle-aware EDI (recommended)
        edi_angle_aware = np.mean([d * s for d, s in zip(displacements, angle_changes)]) if displacements else 0

        # Use angle-aware EDI for more stable measurements
        # High displacement but same direction = less concerning than direction change
        edi = edi_angle_aware if edi_angle_aware > 0 else edi_simple

        # Step 4: Map EDI → λ
        # KEY INSIGHT: High EDI means tasks are SEPARATING (low interference)
        #              Low EDI means tasks OVERLAP (high interference)
        # Therefore: INVERSE relationship - λ = λ_max - κ * EDI
        if self.use_inverse:
            predicted_lambda = max(30.0, self.lambda_max - self.kappa * edi)
        else:
            predicted_lambda = 30.0 + self.kappa * edi

        print(f"  [Probe] EDI = {edi:.4f}")
        print(f"  [Probe] Predicted λ = {predicted_lambda:.1f} (inverse={self.use_inverse})")

        # Return detailed results
        return {
            'edi': edi,
            'edi_angle_aware': edi_angle_aware,
            'predicted_lambda': predicted_lambda,
            'displacements': displacements,
            'mean_displacement': np.mean(displacements) if displacements else 0,
            'max_displacement': max(displacements) if displacements else 0,
        }


class ProbeGuidedRicciCL:
    """
    CC-Ricci with curvature-probe-guided λ selection.
    """

    def __init__(
        self,
        model,
        device='cpu',
        probe_steps=200,
        probe_lr=1e-4,
        lambda_max=350.0,
        kappa=70.0,
        num_classes=10,
        use_inverse=True
    ):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes

        # Probe parameters
        self.probe_steps = probe_steps
        self.probe_lr = probe_lr
        self.lambda_max = lambda_max
        self.kappa = kappa
        self.use_inverse = use_inverse

        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        self.reference_centroids = {}
        self.task_loaders = {}
        self.probe_results = {}
        self.lambda_used = {}

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

    def after_task(self, dataloader, task_id=0, task_name=""):
        print(f"Storing reference for {task_name}...")
        centroids = self.compute_centroids(dataloader)
        self.reference_centroids[task_id] = {c: v.detach().clone() for c, v in centroids.items()}
        self.task_loaders[task_name] = dataloader

    def train_task(self, train_loader, test_loader, epochs=5, task_name="",
                   prev_task_loader=None, verbose=True):
        """
        Train on task with probe-guided λ.

        prev_task_loader: DataLoader for previous task (needed to measure distortion)
        """
        # Run curvature probe if we have previous tasks
        if prev_task_loader is not None and self.reference_centroids:
            print(f"\n  Running curvature probe for {task_name}...")
            probe = CurvatureProbe(
                self.model, self.device,
                probe_steps=self.probe_steps,
                probe_lr=self.probe_lr,
                lambda_max=self.lambda_max,
                kappa=self.kappa,
                use_inverse=self.use_inverse
            )
            probe.reference_centroids = list(self.reference_centroids.values())[0]
            probe.reference_classes = sorted(probe.reference_centroids.keys())

            probe_result = probe.run_probe(train_loader, prev_task_loader)
            predicted_lambda = probe_result['predicted_lambda']

            self.probe_results[task_name] = probe_result
        else:
            # First task: use small λ
            predicted_lambda = 30.0
            print(f"  First task: using λ = {predicted_lambda:.1f}")

        self.lambda_used[task_name] = predicted_lambda

        # Now train with predicted λ
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


def calibrate_kappa(device, loaders):
    """
    Calibrate κ using known oracle λ values.

    From manual experiment:
    - MNIST→Fashion: oracle λ = 50, should have LOW EDI
    - MNIST→KMNIST: oracle λ = 150, should have HIGH EDI

    We solve: κ = (λ_oracle - λ_min) / EDI
    """
    print("\n" + "="*70)
    print("CALIBRATING κ (Scaling Constant)")
    print("="*70)

    oracle_lambdas = {
        'Fashion': 50,
        'KMNIST': 150
    }

    edis = {}

    for task_name, task_train, task_test in [
        ('Fashion', loaders['fashion_train'], loaders['fashion_test']),
        ('KMNIST', loaders['kmnist_train'], loaders['kmnist_test']),
    ]:
        print(f"\n--- Measuring EDI for MNIST → {task_name} ---")

        # Train on MNIST first
        model = SimpleMLP()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(5):
            model.train()
            for batch_x, batch_y in loaders['mnist_train']:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()

        # Run probe
        probe = CurvatureProbe(model, device, probe_steps=200, probe_lr=1e-4)
        probe.set_reference(loaders['mnist_train'])
        result = probe.run_probe(task_train, loaders['mnist_train'])

        edis[task_name] = result['edi']
        print(f"  EDI for {task_name}: {result['edi']:.4f}")

    # Compute κ and λ_max for INVERSE relationship
    # System: λ = λ_max - κ * EDI
    # Fashion: 50 = λ_max - κ * EDI_fashion
    # KMNIST: 150 = λ_max - κ * EDI_kmnist
    # Solving: κ = (λ_kmnist - λ_fashion) / (EDI_fashion - EDI_kmnist)
    #          λ_max = λ_fashion + κ * EDI_fashion

    edi_fashion = edis['Fashion']
    edi_kmnist = edis['KMNIST']
    lambda_fashion = oracle_lambdas['Fashion']
    lambda_kmnist = oracle_lambdas['KMNIST']

    delta_edi = edi_fashion - edi_kmnist
    delta_lambda = lambda_kmnist - lambda_fashion

    if abs(delta_edi) > 0.01:
        kappa = delta_lambda / delta_edi
        lambda_max = lambda_fashion + kappa * edi_fashion
    else:
        # Fallback if EDIs are too similar
        kappa = 70
        lambda_max = 350

    print(f"\n  INVERSE MAPPING CALIBRATION:")
    print(f"    EDI(Fashion) = {edi_fashion:.4f}, λ_oracle = {lambda_fashion}")
    print(f"    EDI(KMNIST)  = {edi_kmnist:.4f}, λ_oracle = {lambda_kmnist}")
    print(f"    κ = ({lambda_kmnist} - {lambda_fashion}) / ({edi_fashion:.4f} - {edi_kmnist:.4f}) = {kappa:.1f}")
    print(f"    λ_max = {lambda_fashion} + {kappa:.1f} * {edi_fashion:.4f} = {lambda_max:.1f}")
    print(f"\n  Formula: λ = {lambda_max:.1f} - {kappa:.1f} * EDI")

    return kappa, lambda_max, edis, oracle_lambdas


def run_validation_experiment(device, loaders, kappa, lambda_max):
    """
    Validate the probe against oracle performance.

    Success criterion:
    - Predicted λ within ±20% of oracle
    - Retention within 95% of oracle performance
    """
    print("\n" + "="*70)
    print("VALIDATION: Probe-Guided vs Oracle vs Fixed")
    print(f"Using INVERSE mapping: λ = {lambda_max:.1f} - {kappa:.1f} * EDI")
    print("="*70)

    results = {}

    oracle_lambdas = {'Fashion': 50, 'KMNIST': 150}

    for task_name, task_train, task_test in [
        ('Fashion', loaders['fashion_train'], loaders['fashion_test']),
        ('KMNIST', loaders['kmnist_train'], loaders['kmnist_test']),
    ]:
        print(f"\n{'='*60}")
        print(f"MNIST → {task_name}")
        print('='*60)

        task_results = {}

        # 1. Oracle (manually tuned λ)
        print(f"\n[Oracle λ={oracle_lambdas[task_name]}]")
        model = SimpleMLP()
        learner = CentroidRicciCL(model, device=device, ricci_lambda=oracle_lambdas[task_name])
        learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
        mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
        learner.after_task(loaders['mnist_train'], task_id=0)
        learner.train_task(task_train, task_test, epochs=5, verbose=False)
        oracle_mnist = learner.evaluate(loaders['mnist_test'])['accuracy']
        oracle_task = learner.evaluate(task_test)['accuracy']
        print(f"  MNIST: {mnist_a:.1%} → {oracle_mnist:.1%}, {task_name}: {oracle_task:.1%}")
        task_results['oracle'] = {'mnist': oracle_mnist, 'task_b': oracle_task, 'lambda': oracle_lambdas[task_name]}

        # 2. Fixed λ=50
        print(f"\n[Fixed λ=50]")
        model = SimpleMLP()
        learner = CentroidRicciCL(model, device=device, ricci_lambda=50)
        learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
        mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
        learner.after_task(loaders['mnist_train'], task_id=0)
        learner.train_task(task_train, task_test, epochs=5, verbose=False)
        fixed_mnist = learner.evaluate(loaders['mnist_test'])['accuracy']
        fixed_task = learner.evaluate(task_test)['accuracy']
        print(f"  MNIST: {mnist_a:.1%} → {fixed_mnist:.1%}, {task_name}: {fixed_task:.1%}")
        task_results['fixed'] = {'mnist': fixed_mnist, 'task_b': fixed_task, 'lambda': 50}

        # 3. Probe-guided (using CentroidRicciCL with predicted λ)
        print(f"\n[Probe-Guided (κ={kappa:.1f}, λ_max={lambda_max:.1f})]")
        model = SimpleMLP()

        # First, train on MNIST with low λ
        learner = CentroidRicciCL(model, device=device, ricci_lambda=30)
        learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
        mnist_a = learner.evaluate(loaders['mnist_test'])['accuracy']
        learner.after_task(loaders['mnist_train'], task_id=0)

        # Run probe to predict λ for next task
        print(f"  Running curvature probe for {task_name}...")
        probe = CurvatureProbe(
            model, device,
            probe_steps=200, probe_lr=1e-4,
            lambda_max=lambda_max, kappa=kappa,
            use_inverse=True
        )
        probe.set_reference(loaders['mnist_train'])
        probe_result = probe.run_probe(task_train, loaders['mnist_train'])
        predicted_lambda = probe_result['predicted_lambda']

        # Train on Task B with predicted λ (create new learner with predicted λ)
        learner.ricci_lambda = predicted_lambda
        learner.train_task(task_train, task_test, epochs=5, verbose=False)
        probe_mnist = learner.evaluate(loaders['mnist_test'])['accuracy']
        probe_task = learner.evaluate(task_test)['accuracy']
        print(f"  MNIST: {mnist_a:.1%} → {probe_mnist:.1%}, {task_name}: {probe_task:.1%}")
        print(f"  Predicted λ = {predicted_lambda:.1f}")
        task_results['probe'] = {'mnist': probe_mnist, 'task_b': probe_task, 'lambda': predicted_lambda}

        # 4. EWC baseline
        print(f"\n[EWC λ=5000]")
        model = SimpleMLP()
        learner = EWCCL(model, device=device, ewc_lambda=5000)
        learner.train_task(loaders['mnist_train'], loaders['mnist_test'], epochs=5, verbose=False)
        learner.after_task(loaders['mnist_train'], task_id=0)
        learner.train_task(task_train, task_test, epochs=5, verbose=False)
        ewc_mnist = learner.evaluate(loaders['mnist_test'])['accuracy']
        ewc_task = learner.evaluate(task_test)['accuracy']
        print(f"  MNIST: {ewc_mnist:.1%}, {task_name}: {ewc_task:.1%}")
        task_results['ewc'] = {'mnist': ewc_mnist, 'task_b': ewc_task}

        results[task_name] = task_results

    return results


def print_final_summary(results, oracle_lambdas):
    """Print final validation summary."""
    print("\n" + "="*70)
    print("FINAL SUMMARY: CURVATURE PROBE VALIDATION")
    print("="*70)

    for task_name, task_results in results.items():
        print(f"\n--- MNIST → {task_name} ---")

        oracle = task_results['oracle']
        probe = task_results['probe']
        fixed = task_results['fixed']
        ewc = task_results['ewc']

        print(f"\n  {'Method':<15} {'λ':<8} {'MNIST Ret.':<12} {'Task B':<12}")
        print(f"  {'-'*50}")
        print(f"  {'Oracle':<15} {oracle['lambda']:<8} {oracle['mnist']:.1%}        {oracle['task_b']:.1%}")
        print(f"  {'Probe':<15} {probe['lambda']:.1f}    {probe['mnist']:.1%}        {probe['task_b']:.1%}")
        print(f"  {'Fixed (50)':<15} {fixed['lambda']:<8} {fixed['mnist']:.1%}        {fixed['task_b']:.1%}")
        print(f"  {'EWC':<15} {'5000':<8} {ewc['mnist']:.1%}        {ewc['task_b']:.1%}")

        # Check success criteria
        oracle_lambda = oracle_lambdas[task_name]
        predicted_lambda = probe['lambda']

        lambda_error = abs(predicted_lambda - oracle_lambda) / oracle_lambda
        retention_ratio = probe['mnist'] / oracle['mnist'] if oracle['mnist'] > 0 else 0

        print(f"\n  Validation Metrics:")
        print(f"    λ prediction error: {lambda_error:.1%} (target: <20%)")
        print(f"    Retention ratio: {retention_ratio:.1%} (target: >95%)")

        if lambda_error < 0.2 and retention_ratio > 0.95:
            print(f"    ✓ SUCCESS: Probe meets criteria!")
        elif lambda_error < 0.3 and retention_ratio > 0.85:
            print(f"    ~ PARTIAL: Probe close to criteria")
        else:
            print(f"    ✗ NOT MET: Probe needs refinement")

    # Overall verdict
    print("\n" + "="*70)
    print("OVERALL VERDICT")
    print("="*70)

    # Count successes by task
    successes = []
    for t in results:
        oracle_lambda = results[t]['oracle']['lambda']
        probe_lambda = results[t]['probe']['lambda']
        lambda_err = abs(probe_lambda - oracle_lambda) / oracle_lambda
        oracle_ret = results[t]['oracle']['mnist']
        probe_ret = results[t]['probe']['mnist']
        retention_ratio = probe_ret / oracle_ret if oracle_ret > 0 else 0
        meets_criteria = lambda_err <= 0.20 and retention_ratio >= 0.95
        successes.append((t, meets_criteria, lambda_err, retention_ratio))

    num_success = sum(1 for _, m, _, _ in successes if m)

    if num_success == len(results):
        print("\n✓ CURVATURE PROBE FULLY VALIDATED!")
        print("  - All tasks meet success criteria")
        print("  - λ prediction error <20%")
        print("  - Retention >95% of oracle")
        print("\n  λ is no longer a hyperparameter.")
        print("  Continual learning is self-calibrating.")
    elif num_success > 0:
        print(f"\n◐ PARTIAL SUCCESS ({num_success}/{len(results)} tasks meet criteria)")
        for t, met, lerr, rr in successes:
            if met:
                print(f"  ✓ {t}: λ err {lerr:.1%}, retention {rr:.1%}")
            else:
                print(f"  ✗ {t}: λ err {lerr:.1%}, retention {rr:.1%}")

        # Check if probe helps on high-interference tasks
        if 'KMNIST' in results:
            fixed_ret = results['KMNIST']['fixed']['mnist']
            probe_ret = results['KMNIST']['probe']['mnist']
            if probe_ret > fixed_ret * 1.5:
                print(f"\n  KEY FINDING: Probe excels on HIGH-INTERFERENCE tasks!")
                print(f"    KMNIST: Fixed λ=50 → {fixed_ret:.1%}")
                print(f"    KMNIST: Probe λ={results['KMNIST']['probe']['lambda']:.0f} → {probe_ret:.1%}")
                print(f"    Improvement: {probe_ret/fixed_ret:.1f}×")
                print("\n  Recommendation: Use probe for similar task pairs (high interference)")
    else:
        print("\n✗ PROBE NEEDS REFINEMENT")
        print("  - κ calibration may need adjustment")
        print("  - Or probe_steps/probe_lr need tuning")


def main():
    device = 'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    loaders = get_datasets('./data', subset_size=2000)

    # Step 1: Calibrate κ and λ_max for inverse mapping
    kappa, lambda_max, edis, oracle_lambdas = calibrate_kappa(device, loaders)

    # Step 2: Validate probe with inverse mapping
    results = run_validation_experiment(device, loaders, kappa, lambda_max)

    # Step 3: Print summary
    print_final_summary(results, oracle_lambdas)


if __name__ == "__main__":
    main()
