"""
Continual Learning Methods for the Ricci-REWA Experiment

This module implements three approaches to catastrophic forgetting:
1. Baseline: Standard training (no protection)
2. EWC: Elastic Weight Consolidation (protect important weights)
3. Ricci-Reg: Ricci Regularization (protect geometry)

The key hypothesis: Ricci-Reg will outperform EWC because it preserves
the geometric structure of learned representations rather than the
specific weight values.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Callable
from copy import deepcopy
import numpy as np
from tqdm import tqdm

from .ricci_curvature import (
    RicciCurvatureRegularizer,
    DifferentiableRicciLoss,
    compute_ricci_on_embeddings
)
from .models import get_weight_vector, compute_weight_distance


class ContinualLearner:
    """Base class for continual learning methods."""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        lr: float = 1e-3,
        weight_decay: float = 0.0
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.task_count = 0
        self.initial_weights = get_weight_vector(model).clone()

        # Metrics storage
        self.training_history = []

    def train_epoch(
        self,
        dataloader: DataLoader,
        task_id: int = 0
    ) -> Dict[str, float]:
        """Train for one epoch. Override in subclasses for regularization."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total
        }

    def evaluate(
        self,
        dataloader: DataLoader,
        task_id: int = 0
    ) -> Dict[str, float]:
        """Evaluate on a dataset."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = F.cross_entropy(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        return {
            'loss': total_loss / len(dataloader),
            'accuracy': correct / total
        }

    def train_task(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        task_id: int = 0,
        verbose: bool = True
    ) -> List[Dict]:
        """Train on a complete task."""
        history = []

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, task_id)

            if val_loader:
                val_metrics = self.evaluate(val_loader, task_id)
                train_metrics['val_loss'] = val_metrics['loss']
                train_metrics['val_accuracy'] = val_metrics['accuracy']

            train_metrics['epoch'] = epoch
            train_metrics['task_id'] = task_id
            history.append(train_metrics)

            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs} - "
                msg += f"Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}"
                if val_loader:
                    msg += f" - Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"
                print(msg)

        self.task_count += 1
        self.training_history.extend(history)

        return history

    def after_task(self, dataloader: DataLoader, task_id: int = 0):
        """Called after completing training on a task. Override in subclasses."""
        pass

    def get_weight_change(self) -> float:
        """Compute weight change from initialization."""
        current_weights = get_weight_vector(self.model)
        return torch.norm(current_weights - self.initial_weights).item()


class BaselineCL(ContinualLearner):
    """
    Baseline continual learning: standard training with no protection.

    Expected behavior: catastrophic forgetting - after training on Task B,
    performance on Task A collapses to chance.
    """

    def __init__(self, model: nn.Module, device: str = 'cpu', **kwargs):
        super().__init__(model, device, **kwargs)


class EWCCL(ContinualLearner):
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    Protects important weights by penalizing changes to them.
    Importance is measured by the Fisher Information Matrix.

    Key limitation: operates in weight space, constraining capacity.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        ewc_lambda: float = 1000.0,
        fisher_samples: int = 1000,
        **kwargs
    ):
        super().__init__(model, device, **kwargs)

        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples

        # Storage for Fisher information and optimal parameters
        self.fisher_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        self.optimal_params_dict: Dict[int, Dict[str, torch.Tensor]] = {}

    def compute_fisher(
        self,
        dataloader: DataLoader,
        task_id: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix (diagonal approximation).

        Fisher = E[grad(log p(y|x))^2] - measures parameter importance.
        """
        self.model.eval()

        fisher = {}
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param.data)

        # Sample from the dataloader
        samples_processed = 0
        for data, target in dataloader:
            if samples_processed >= self.fisher_samples:
                break

            data, target = data.to(self.device), target.to(self.device)
            batch_size = data.size(0)

            self.model.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) * batch_size

            samples_processed += batch_size

        # Normalize
        for name in fisher:
            fisher[name] /= samples_processed

        return fisher

    def ewc_loss(self) -> torch.Tensor:
        """
        Compute EWC penalty for all previous tasks.

        L_ewc = sum_i sum_j F_j * (θ_j - θ*_j)^2

        where F is Fisher information and θ* are optimal params from previous task.
        """
        loss = torch.tensor(0.0, device=self.device)

        for task_id in self.fisher_dict:
            for name, param in self.model.named_parameters():
                fisher = self.fisher_dict[task_id][name]
                optimal = self.optimal_params_dict[task_id][name]

                loss += (fisher * (param - optimal).pow(2)).sum()

        return loss

    def train_epoch(
        self,
        dataloader: DataLoader,
        task_id: int = 0
    ) -> Dict[str, float]:
        """Train with EWC regularization."""
        self.model.train()
        total_loss = 0
        total_ewc_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)

            ce_loss = F.cross_entropy(output, target)
            ewc_penalty = self.ewc_loss()

            loss = ce_loss + self.ewc_lambda * ewc_penalty

            loss.backward()
            self.optimizer.step()

            total_loss += ce_loss.item()
            total_ewc_loss += ewc_penalty.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        return {
            'loss': total_loss / len(dataloader),
            'ewc_loss': total_ewc_loss / len(dataloader),
            'accuracy': correct / total
        }

    def after_task(self, dataloader: DataLoader, task_id: int = 0):
        """Compute Fisher information after task completion."""
        print(f"Computing Fisher information for task {task_id}...")

        # Store optimal parameters
        self.optimal_params_dict[task_id] = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

        # Compute Fisher
        self.fisher_dict[task_id] = self.compute_fisher(dataloader, task_id)

        print(f"Fisher computation complete.")


class RicciRegCL(ContinualLearner):
    """
    Ricci Regularization Continual Learning.

    Instead of protecting weights (EWC), we protect the local geometry
    of the learned representations as measured by Ricci curvature.

    Key hypothesis: this allows weights to change freely while preserving
    the functional properties encoded in the geometry.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        ricci_lambda: float = 1.0,
        k_neighbors: int = 10,
        curvature_samples: int = 200,
        use_differentiable: bool = True,
        **kwargs
    ):
        super().__init__(model, device, **kwargs)

        self.ricci_lambda = ricci_lambda
        self.k_neighbors = k_neighbors
        self.curvature_samples = curvature_samples
        self.use_differentiable = use_differentiable

        # Curvature regularizers for each task
        if use_differentiable:
            self.curvature_regularizers: Dict[int, DifferentiableRicciLoss] = {}
        else:
            self.curvature_regularizers: Dict[int, RicciCurvatureRegularizer] = {}

        # Store reference embeddings for each task
        self.reference_embeddings: Dict[int, torch.Tensor] = {}

        # Store reference curvature tensors for analysis
        self.reference_curvatures: Dict[int, np.ndarray] = {}

    def compute_ricci_loss(
        self,
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute curvature preservation loss for all previous tasks.
        """
        if not self.curvature_regularizers:
            return torch.tensor(0.0, device=self.device)

        total_loss = torch.tensor(0.0, device=self.device)

        for task_id, regularizer in self.curvature_regularizers.items():
            if self.use_differentiable:
                loss = regularizer(embeddings)
            else:
                loss = regularizer.compute_loss(embeddings)
            total_loss += loss

        return total_loss

    def train_epoch(
        self,
        dataloader: DataLoader,
        task_id: int = 0
    ) -> Dict[str, float]:
        """Train with Ricci curvature regularization."""
        self.model.train()
        total_loss = 0
        total_ricci_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass with embeddings
            logits, embeddings = self.model.forward_with_embeddings(data)

            ce_loss = F.cross_entropy(logits, target)
            ricci_penalty = self.compute_ricci_loss(embeddings)

            loss = ce_loss + self.ricci_lambda * ricci_penalty

            loss.backward()
            self.optimizer.step()

            total_loss += ce_loss.item()
            total_ricci_loss += ricci_penalty.item() if isinstance(ricci_penalty, torch.Tensor) else ricci_penalty
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        return {
            'loss': total_loss / len(dataloader),
            'ricci_loss': total_ricci_loss / len(dataloader),
            'accuracy': correct / total
        }

    def after_task(self, dataloader: DataLoader, task_id: int = 0):
        """Store reference curvature after task completion."""
        print(f"Computing reference curvature for task {task_id}...")

        self.model.eval()

        # Collect embeddings from the dataset
        all_embeddings = []
        samples = 0

        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                embeddings = self.model.get_embeddings(data)
                all_embeddings.append(embeddings)

                samples += data.size(0)
                if samples >= self.curvature_samples * 10:
                    break

        all_embeddings = torch.cat(all_embeddings, dim=0)

        # Sample if needed
        if all_embeddings.shape[0] > self.curvature_samples:
            indices = torch.randperm(all_embeddings.shape[0])[:self.curvature_samples]
            sampled_embeddings = all_embeddings[indices]
        else:
            sampled_embeddings = all_embeddings

        # Store reference embeddings
        self.reference_embeddings[task_id] = sampled_embeddings.detach().clone()

        # Create regularizer
        if self.use_differentiable:
            regularizer = DifferentiableRicciLoss(k_neighbors=self.k_neighbors)
            regularizer.set_reference(sampled_embeddings)
        else:
            regularizer = RicciCurvatureRegularizer(
                k_neighbors=self.k_neighbors,
                sample_size=self.curvature_samples
            )
            regularizer.set_reference(sampled_embeddings)

        self.curvature_regularizers[task_id] = regularizer

        # Also compute and store the actual Ricci tensor for analysis
        R, _ = compute_ricci_on_embeddings(
            sampled_embeddings,
            k_neighbors=self.k_neighbors,
            sample_size=min(200, self.curvature_samples)
        )
        self.reference_curvatures[task_id] = R

        print(f"Reference curvature stored for task {task_id}.")

    def get_current_curvature(
        self,
        dataloader: DataLoader
    ) -> np.ndarray:
        """Compute current curvature for analysis."""
        self.model.eval()

        all_embeddings = []
        samples = 0

        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                embeddings = self.model.get_embeddings(data)
                all_embeddings.append(embeddings)

                samples += data.size(0)
                if samples >= self.curvature_samples:
                    break

        all_embeddings = torch.cat(all_embeddings, dim=0)

        if all_embeddings.shape[0] > self.curvature_samples:
            indices = torch.randperm(all_embeddings.shape[0])[:self.curvature_samples]
            all_embeddings = all_embeddings[indices]

        R, _ = compute_ricci_on_embeddings(
            all_embeddings,
            k_neighbors=self.k_neighbors,
            sample_size=min(200, self.curvature_samples)
        )

        return R

    def curvature_distance(
        self,
        dataloader: DataLoader,
        task_id: int = 0
    ) -> float:
        """Compute how much curvature has changed from reference."""
        if task_id not in self.reference_curvatures:
            return 0.0

        current_R = self.get_current_curvature(dataloader)
        reference_R = self.reference_curvatures[task_id]

        # Match sizes
        min_size = min(current_R.shape[0], reference_R.shape[0])
        current_R = current_R[:min_size, :min_size]
        reference_R = reference_R[:min_size, :min_size]

        # Frobenius norm of difference
        return np.linalg.norm(current_R - reference_R, 'fro')


class HybridCL(ContinualLearner):
    """
    Hybrid method: combine EWC and Ricci regularization.

    This tests whether the methods are complementary or if
    Ricci regularization subsumes EWC.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        ewc_lambda: float = 500.0,
        ricci_lambda: float = 0.5,
        fisher_samples: int = 1000,
        k_neighbors: int = 10,
        curvature_samples: int = 200,
        **kwargs
    ):
        super().__init__(model, device, **kwargs)

        self.ewc_lambda = ewc_lambda
        self.ricci_lambda = ricci_lambda
        self.fisher_samples = fisher_samples
        self.k_neighbors = k_neighbors
        self.curvature_samples = curvature_samples

        # EWC components
        self.fisher_dict: Dict[int, Dict[str, torch.Tensor]] = {}
        self.optimal_params_dict: Dict[int, Dict[str, torch.Tensor]] = {}

        # Ricci components
        self.curvature_regularizers: Dict[int, DifferentiableRicciLoss] = {}
        self.reference_embeddings: Dict[int, torch.Tensor] = {}

    def ewc_loss(self) -> torch.Tensor:
        """Compute EWC penalty."""
        loss = torch.tensor(0.0, device=self.device)

        for task_id in self.fisher_dict:
            for name, param in self.model.named_parameters():
                fisher = self.fisher_dict[task_id][name]
                optimal = self.optimal_params_dict[task_id][name]
                loss += (fisher * (param - optimal).pow(2)).sum()

        return loss

    def ricci_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute Ricci curvature loss."""
        if not self.curvature_regularizers:
            return torch.tensor(0.0, device=self.device)

        total_loss = torch.tensor(0.0, device=self.device)

        for task_id, regularizer in self.curvature_regularizers.items():
            total_loss += regularizer(embeddings)

        return total_loss

    def train_epoch(
        self,
        dataloader: DataLoader,
        task_id: int = 0
    ) -> Dict[str, float]:
        """Train with both EWC and Ricci regularization."""
        self.model.train()
        total_loss = 0
        total_ewc_loss = 0
        total_ricci_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            logits, embeddings = self.model.forward_with_embeddings(data)

            ce_loss = F.cross_entropy(logits, target)
            ewc_penalty = self.ewc_loss()
            ricci_penalty = self.ricci_loss(embeddings)

            loss = ce_loss + self.ewc_lambda * ewc_penalty + self.ricci_lambda * ricci_penalty

            loss.backward()
            self.optimizer.step()

            total_loss += ce_loss.item()
            total_ewc_loss += ewc_penalty.item()
            total_ricci_loss += ricci_penalty.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        return {
            'loss': total_loss / len(dataloader),
            'ewc_loss': total_ewc_loss / len(dataloader),
            'ricci_loss': total_ricci_loss / len(dataloader),
            'accuracy': correct / total
        }

    def after_task(self, dataloader: DataLoader, task_id: int = 0):
        """Store both Fisher information and reference curvature."""
        print(f"Computing Fisher information and reference curvature for task {task_id}...")

        # EWC: store optimal params and compute Fisher
        self.optimal_params_dict[task_id] = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
        }

        self.model.eval()
        fisher = {}
        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param.data)

        samples_processed = 0
        for data, target in dataloader:
            if samples_processed >= self.fisher_samples:
                break
            data, target = data.to(self.device), target.to(self.device)
            self.model.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) * data.size(0)

            samples_processed += data.size(0)

        for name in fisher:
            fisher[name] /= samples_processed

        self.fisher_dict[task_id] = fisher

        # Ricci: store reference embeddings and curvature
        all_embeddings = []
        samples = 0

        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                embeddings = self.model.get_embeddings(data)
                all_embeddings.append(embeddings)
                samples += data.size(0)
                if samples >= self.curvature_samples:
                    break

        all_embeddings = torch.cat(all_embeddings, dim=0)

        if all_embeddings.shape[0] > self.curvature_samples:
            indices = torch.randperm(all_embeddings.shape[0])[:self.curvature_samples]
            sampled_embeddings = all_embeddings[indices]
        else:
            sampled_embeddings = all_embeddings

        self.reference_embeddings[task_id] = sampled_embeddings.detach().clone()

        regularizer = DifferentiableRicciLoss(k_neighbors=self.k_neighbors)
        regularizer.set_reference(sampled_embeddings)
        self.curvature_regularizers[task_id] = regularizer

        print(f"Task {task_id} consolidation complete.")


if __name__ == "__main__":
    from .models import SimpleMLP

    print("Testing continual learning methods...")

    # Create a simple model
    model = SimpleMLP()

    # Create dummy data
    X = torch.randn(100, 784)
    y = torch.randint(0, 10, (100,))
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    # Test Baseline
    print("\nTesting Baseline...")
    baseline = BaselineCL(SimpleMLP())
    metrics = baseline.train_epoch(loader)
    print(f"Baseline metrics: {metrics}")

    # Test EWC
    print("\nTesting EWC...")
    ewc = EWCCL(SimpleMLP(), ewc_lambda=1000)
    metrics = ewc.train_epoch(loader)
    print(f"EWC metrics: {metrics}")
    ewc.after_task(loader, task_id=0)
    metrics = ewc.train_epoch(loader)  # Should now have EWC penalty
    print(f"EWC with penalty: {metrics}")

    # Test Ricci
    print("\nTesting RicciReg...")
    ricci = RicciRegCL(SimpleMLP(), ricci_lambda=0.1)
    metrics = ricci.train_epoch(loader)
    print(f"Ricci metrics: {metrics}")
    ricci.after_task(loader, task_id=0)
    metrics = ricci.train_epoch(loader)
    print(f"Ricci with penalty: {metrics}")

    print("\nAll continual learning tests passed!")
