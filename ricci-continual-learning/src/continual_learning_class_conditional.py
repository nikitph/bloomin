"""
Continual Learning with Class-Conditional Curvature Preservation

This preserves the geometric structure of each class separately,
which should be more directly related to classification performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple

from .class_conditional_curvature import (
    ClassConditionalCurvature,
    ClassCentroidRegularizer,
    PrototypeRegularizer
)
from .models import get_weight_vector


class ClassConditionalRicciCL:
    """
    Continual Learning with class-conditional curvature preservation.

    Key idea: Instead of preserving global embedding curvature,
    preserve the structure WITHIN each class. This is more directly
    related to classification boundaries.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        ricci_lambda: float = 10.0,
        num_classes: int = 10,
        preserve_centroids: bool = True,
        preserve_spread: bool = True,
        preserve_local: bool = True
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.ricci_lambda = ricci_lambda
        self.num_classes = num_classes

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.task_count = 0
        self.initial_weights = get_weight_vector(model).clone()

        # Class-conditional regularizers for each task
        self.regularizers: Dict[int, ClassConditionalCurvature] = {}

        self.training_history = []

    def compute_curvature_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute class-conditional curvature loss."""
        if not self.regularizers:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for task_id, regularizer in self.regularizers.items():
            loss = regularizer(embeddings, labels)
            total_loss = total_loss + loss

        return total_loss

    def train_epoch(
        self,
        dataloader: DataLoader,
        task_id: int = 0
    ) -> Dict[str, float]:
        """Train with class-conditional curvature regularization."""
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
            curv_loss = self.compute_curvature_loss(embeddings, target)

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
        """Evaluate accuracy."""
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
        """Train on a task."""
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
        """Store reference class structure after task."""
        print(f"Computing class-conditional reference for task {task_id}...")

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

        # Create and store regularizer
        regularizer = ClassConditionalCurvature(num_classes=self.num_classes)
        regularizer.set_reference(all_embeddings, all_labels)
        self.regularizers[task_id] = regularizer

        print(f"Class-conditional reference stored for task {task_id}.")

    def get_weight_change(self) -> float:
        """Get weight change from initialization."""
        current_weights = get_weight_vector(self.model)
        return torch.norm(current_weights - self.initial_weights).item()


class CentroidRicciCL:
    """
    Simpler version: just preserve class centroid structure.

    This is faster and may be sufficient for preventing forgetting.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        ricci_lambda: float = 10.0,
        num_classes: int = 10
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.ricci_lambda = ricci_lambda
        self.num_classes = num_classes

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.task_count = 0
        self.initial_weights = get_weight_vector(model).clone()

        self.regularizers: Dict[int, ClassCentroidRegularizer] = {}
        self.training_history = []

    def compute_curvature_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute centroid structure loss."""
        if not self.regularizers:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for task_id, regularizer in self.regularizers.items():
            loss = regularizer(embeddings, labels)
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
            curv_loss = self.compute_curvature_loss(embeddings, target)

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
        print(f"Computing centroid reference for task {task_id}...")

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

        regularizer = ClassCentroidRegularizer(num_classes=self.num_classes)
        regularizer.set_reference(all_embeddings, all_labels)
        self.regularizers[task_id] = regularizer

        print(f"Centroid reference stored for task {task_id}.")

    def get_weight_change(self) -> float:
        """Get weight change."""
        current_weights = get_weight_vector(self.model)
        return torch.norm(current_weights - self.initial_weights).item()


class PrototypeRicciCL:
    """
    Prototype-based continual learning.

    Store class prototypes and regularize samples to:
    1. Stay close to their own prototype
    2. Stay far from other prototypes
    3. Preserve prototype structure
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        ricci_lambda: float = 10.0,
        num_classes: int = 10,
        attraction_weight: float = 1.0,
        repulsion_weight: float = 0.3,
        structure_weight: float = 1.0
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.ricci_lambda = ricci_lambda
        self.num_classes = num_classes

        self.attraction_weight = attraction_weight
        self.repulsion_weight = repulsion_weight
        self.structure_weight = structure_weight

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        self.task_count = 0
        self.initial_weights = get_weight_vector(model).clone()

        self.regularizers: Dict[int, PrototypeRegularizer] = {}
        self.training_history = []

    def compute_curvature_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute prototype-based loss."""
        if not self.regularizers:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for task_id, regularizer in self.regularizers.items():
            loss = regularizer(embeddings, labels)
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
            curv_loss = self.compute_curvature_loss(embeddings, target)

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
                msg += f"Proto: {train_metrics['curv_loss']:.4f}, "
                msg += f"Acc: {train_metrics['accuracy']:.4f}"
                if val_loader:
                    msg += f" - Val: {val_metrics['accuracy']:.4f}"
                print(msg)

        self.task_count += 1
        return history

    def after_task(self, dataloader: DataLoader, task_id: int = 0):
        """Store prototypes."""
        print(f"Computing prototypes for task {task_id}...")

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

        regularizer = PrototypeRegularizer(
            num_classes=self.num_classes,
            attraction_weight=self.attraction_weight,
            repulsion_weight=self.repulsion_weight,
            structure_weight=self.structure_weight
        )
        regularizer.set_reference(all_embeddings, all_labels)
        self.regularizers[task_id] = regularizer

        print(f"Prototypes stored for task {task_id}.")

    def get_weight_change(self) -> float:
        """Get weight change."""
        current_weights = get_weight_vector(self.model)
        return torch.norm(current_weights - self.initial_weights).item()
