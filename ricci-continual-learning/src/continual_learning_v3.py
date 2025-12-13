"""
Continual Learning with Focused Curvature Preservation

Uses scale-invariant curvature measures that should allow:
1. Absolute distances to change freely
2. Weights to change freely
3. But local curvature (shape) to be preserved
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional

from .curvature_focused import FocusedCurvatureRegularizer, CurvatureOnlyRegularizer
from .models import get_weight_vector


class FocusedRicciCL:
    """
    Continual Learning with focused curvature preservation.

    Key difference from previous versions:
    - Only preserves SHAPE (curvature), not SIZE (distances)
    - This should allow more plasticity while preserving stability
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        ricci_lambda: float = 50.0,
        k_neighbors: int = 15,
        n_samples: int = 300
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.ricci_lambda = ricci_lambda
        self.k_neighbors = k_neighbors
        self.n_samples = n_samples

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.task_count = 0
        self.initial_weights = get_weight_vector(model).clone()

        # Curvature regularizers for each task
        self.regularizers: Dict[int, FocusedCurvatureRegularizer] = {}
        self.reference_embeddings: Dict[int, torch.Tensor] = {}

        self.training_history = []

    def compute_curvature_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute focused curvature preservation loss."""
        if not self.regularizers:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for task_id, regularizer in self.regularizers.items():
            loss = regularizer(embeddings)
            total_loss = total_loss + loss

        return total_loss

    def train_epoch(
        self,
        dataloader: DataLoader,
        task_id: int = 0
    ) -> Dict[str, float]:
        """Train with focused curvature regularization."""
        self.model.train()
        total_loss = 0
        total_ce_loss = 0
        total_curv_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            logits, embeddings = self.model.forward_with_embeddings(data)

            ce_loss = F.cross_entropy(logits, target)
            curv_loss = self.compute_curvature_loss(embeddings)

            loss = ce_loss + self.ricci_lambda * curv_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_curv_loss += curv_loss.item() if isinstance(curv_loss, torch.Tensor) else curv_loss
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        return {
            'loss': total_ce_loss / len(dataloader),
            'curv_loss': total_curv_loss / len(dataloader),
            'total_loss': total_loss / len(dataloader),
            'accuracy': correct / total
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate accuracy."""
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
        """Train on a task."""
        history = []

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, task_id)

            if val_loader:
                val_metrics = self.evaluate(val_loader)
                train_metrics['val_loss'] = val_metrics['loss']
                train_metrics['val_accuracy'] = val_metrics['accuracy']

            history.append(train_metrics)

            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs} - "
                msg += f"Loss: {train_metrics['loss']:.4f}, "
                msg += f"Curv: {train_metrics['curv_loss']:.4f}, "
                msg += f"Acc: {train_metrics['accuracy']:.4f}"
                if val_loader:
                    msg += f" - Val: {val_metrics['accuracy']:.4f}"
                print(msg)

        self.task_count += 1
        return history

    def after_task(self, dataloader: DataLoader, task_id: int = 0):
        """Store reference curvature after task."""
        print(f"Computing reference curvature (focused) for task {task_id}...")

        self.model.eval()

        all_embeddings = []
        samples = 0

        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                embeddings = self.model.get_embeddings(data)
                all_embeddings.append(embeddings)
                samples += data.size(0)
                if samples >= self.n_samples * 2:
                    break

        all_embeddings = torch.cat(all_embeddings, dim=0)

        if all_embeddings.shape[0] > self.n_samples:
            indices = torch.randperm(all_embeddings.shape[0])[:self.n_samples]
            sampled_embeddings = all_embeddings[indices]
        else:
            sampled_embeddings = all_embeddings

        self.reference_embeddings[task_id] = sampled_embeddings.detach().clone()

        regularizer = FocusedCurvatureRegularizer(
            k_neighbors=self.k_neighbors,
            n_samples=self.n_samples
        )
        regularizer.set_reference(sampled_embeddings)
        self.regularizers[task_id] = regularizer

        print(f"Reference curvature stored for task {task_id}.")

    def get_weight_change(self) -> float:
        """Get weight change from initialization."""
        current_weights = get_weight_vector(self.model)
        return torch.norm(current_weights - self.initial_weights).item()


class AngleOnlyRicciCL:
    """
    Even more focused: only preserve angular relationships.

    This is the most scale-invariant version, preserving only
    the directions between neighbors, not distances at all.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        ricci_lambda: float = 50.0,
        k_neighbors: int = 15,
        n_samples: int = 300
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.ricci_lambda = ricci_lambda
        self.k_neighbors = k_neighbors
        self.n_samples = n_samples

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.task_count = 0
        self.initial_weights = get_weight_vector(model).clone()

        self.regularizers: Dict[int, CurvatureOnlyRegularizer] = {}
        self.training_history = []

    def compute_curvature_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute angle-based curvature loss."""
        if not self.regularizers:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for task_id, regularizer in self.regularizers.items():
            loss = regularizer(embeddings)
            total_loss = total_loss + loss

        return total_loss

    def train_epoch(self, dataloader: DataLoader, task_id: int = 0) -> Dict[str, float]:
        """Train epoch."""
        self.model.train()
        total_loss = 0
        total_ce_loss = 0
        total_curv_loss = 0
        correct = 0
        total = 0

        for data, target in dataloader:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            logits, embeddings = self.model.forward_with_embeddings(data)

            ce_loss = F.cross_entropy(logits, target)
            curv_loss = self.compute_curvature_loss(embeddings)

            loss = ce_loss + self.ricci_lambda * curv_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_curv_loss += curv_loss.item()
            pred = logits.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

        return {
            'loss': total_ce_loss / len(dataloader),
            'curv_loss': total_curv_loss / len(dataloader),
            'accuracy': correct / total
        }

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        return {'accuracy': correct / total}

    def train_task(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        task_id: int = 0,
        verbose: bool = True
    ) -> List[Dict]:
        """Train task."""
        history = []

        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, task_id)

            if val_loader:
                val_metrics = self.evaluate(val_loader)
                train_metrics['val_accuracy'] = val_metrics['accuracy']

            history.append(train_metrics)

            if verbose:
                msg = f"Epoch {epoch + 1}/{epochs} - "
                msg += f"Loss: {train_metrics['loss']:.4f}, "
                msg += f"Curv: {train_metrics['curv_loss']:.4f}, "
                msg += f"Acc: {train_metrics['accuracy']:.4f}"
                if val_loader:
                    msg += f" - Val: {val_metrics['accuracy']:.4f}"
                print(msg)

        self.task_count += 1
        return history

    def after_task(self, dataloader: DataLoader, task_id: int = 0):
        """Store reference."""
        print(f"Computing angular reference for task {task_id}...")

        self.model.eval()

        all_embeddings = []
        samples = 0

        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                embeddings = self.model.get_embeddings(data)
                all_embeddings.append(embeddings)
                samples += data.size(0)
                if samples >= self.n_samples * 2:
                    break

        all_embeddings = torch.cat(all_embeddings, dim=0)

        if all_embeddings.shape[0] > self.n_samples:
            indices = torch.randperm(all_embeddings.shape[0])[:self.n_samples]
            sampled_embeddings = all_embeddings[indices]
        else:
            sampled_embeddings = all_embeddings

        regularizer = CurvatureOnlyRegularizer(k_neighbors=self.k_neighbors)
        regularizer.set_reference(sampled_embeddings)
        self.regularizers[task_id] = regularizer

        print(f"Reference stored for task {task_id}.")

    def get_weight_change(self) -> float:
        """Get weight change."""
        current_weights = get_weight_vector(self.model)
        return torch.norm(current_weights - self.initial_weights).item()
