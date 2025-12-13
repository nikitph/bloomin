"""
Neural Network Models for Continual Learning Experiments

These models are designed to expose intermediate embeddings for
Ricci curvature analysis while maintaining competitive performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class SimpleMLP(nn.Module):
    """
    Simple MLP for MNIST/FashionMNIST classification.

    Architecture designed to create meaningful intermediate representations
    for curvature analysis.
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        num_classes: int = 10,
        dropout: float = 0.2
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_dim, num_classes)

        # For curvature analysis - stores the embedding dimension
        self.embedding_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        x = x.view(x.size(0), -1)  # Flatten
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the penultimate layer embeddings (before classifier).

        These embeddings are used for Ricci curvature computation.
        """
        x = x.view(x.size(0), -1)
        embeddings = self.features(x)
        return embeddings

    def forward_with_embeddings(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass that also returns embeddings.

        Returns:
            logits: Classification logits
            embeddings: Penultimate layer activations
        """
        x = x.view(x.size(0), -1)
        embeddings = self.features(x)
        logits = self.classifier(embeddings)
        return logits, embeddings


class ConvNet(nn.Module):
    """
    Convolutional network for image classification.

    More powerful than MLP, with convolutional feature extraction
    and a learned embedding space.
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 10,
        embedding_dim: int = 128
    ):
        super().__init__()

        self.embedding_dim = embedding_dim

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Calculate flattened size after convolutions
        # For 28x28 input: 28->14->7->3 (after 3 pools)
        self.flat_size = 128 * 3 * 3

        # Embedding layer
        self.fc1 = nn.Linear(self.flat_size, embedding_dim)
        self.fc1_bn = nn.BatchNorm1d(embedding_dim)

        # Classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass."""
        # Ensure proper shape
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            # Assume flattened 28x28
            x = x.view(-1, 1, 28, 28)

        # Conv layers
        x = self.pool(F.relu(self.conv1(x)))  # 28->14
        x = self.pool(F.relu(self.conv2(x)))  # 14->7
        x = self.pool(F.relu(self.conv3(x)))  # 7->3
        x = self.dropout(x)

        # Flatten and embed
        x = x.view(-1, self.flat_size)
        x = F.relu(self.fc1_bn(self.fc1(x)))

        # Classify
        x = self.classifier(x)
        return x

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings from the embedding layer."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            x = x.view(-1, 1, 28, 28)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)

        x = x.view(-1, self.flat_size)
        embeddings = F.relu(self.fc1_bn(self.fc1(x)))

        return embeddings

    def forward_with_embeddings(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both logits and embeddings."""
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() == 2:
            x = x.view(-1, 1, 28, 28)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)

        x = x.view(-1, self.flat_size)
        embeddings = F.relu(self.fc1_bn(self.fc1(x)))
        logits = self.classifier(embeddings)

        return logits, embeddings


class MultiHeadClassifier(nn.Module):
    """
    Model with shared feature extractor and task-specific heads.

    This allows fair comparison: all methods use the same feature
    extraction, differing only in how they handle continual learning.
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_tasks: int = 2,
        num_classes_per_task: int = 10
    ):
        super().__init__()

        self.base_model = base_model
        self.num_tasks = num_tasks

        # Remove the original classifier from base model
        embedding_dim = base_model.embedding_dim

        # Create task-specific heads
        self.heads = nn.ModuleList([
            nn.Linear(embedding_dim, num_classes_per_task)
            for _ in range(num_tasks)
        ])

    def forward(
        self,
        x: torch.Tensor,
        task_id: int = 0
    ) -> torch.Tensor:
        """Forward pass for a specific task."""
        embeddings = self.base_model.get_embeddings(x)
        logits = self.heads[task_id](embeddings)
        return logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get embeddings from base model."""
        return self.base_model.get_embeddings(x)

    def forward_with_embeddings(
        self,
        x: torch.Tensor,
        task_id: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both logits and embeddings."""
        embeddings = self.base_model.get_embeddings(x)
        logits = self.heads[task_id](embeddings)
        return logits, embeddings


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_weight_vector(model: nn.Module) -> torch.Tensor:
    """Flatten all model parameters into a single vector."""
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def set_weight_vector(model: nn.Module, weight_vector: torch.Tensor):
    """Set model parameters from a flattened vector."""
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(weight_vector[offset:offset + numel].view_as(p))
        offset += numel


def compute_weight_distance(
    model1: nn.Module,
    model2: nn.Module
) -> float:
    """Compute L2 distance between two models' parameters."""
    w1 = get_weight_vector(model1)
    w2 = get_weight_vector(model2)
    return torch.norm(w1 - w2).item()


if __name__ == "__main__":
    # Test models
    print("Testing models...")

    # Test MLP
    mlp = SimpleMLP()
    x = torch.randn(32, 784)

    logits = mlp(x)
    print(f"MLP output shape: {logits.shape}")

    embeddings = mlp.get_embeddings(x)
    print(f"MLP embeddings shape: {embeddings.shape}")

    logits2, emb2 = mlp.forward_with_embeddings(x)
    print(f"MLP forward_with_embeddings: logits {logits2.shape}, emb {emb2.shape}")

    print(f"MLP parameters: {count_parameters(mlp):,}")

    # Test ConvNet
    conv = ConvNet()
    x_img = torch.randn(32, 1, 28, 28)

    logits = conv(x_img)
    print(f"\nConvNet output shape: {logits.shape}")

    embeddings = conv.get_embeddings(x_img)
    print(f"ConvNet embeddings shape: {embeddings.shape}")

    print(f"ConvNet parameters: {count_parameters(conv):,}")

    # Test MultiHead
    multi = MultiHeadClassifier(SimpleMLP(), num_tasks=2)
    x = torch.randn(32, 784)

    for task_id in range(2):
        logits = multi(x, task_id=task_id)
        print(f"\nMultiHead task {task_id} output shape: {logits.shape}")

    print(f"MultiHead parameters: {count_parameters(multi):,}")

    # Test weight distance
    mlp1 = SimpleMLP()
    mlp2 = SimpleMLP()
    dist = compute_weight_distance(mlp1, mlp2)
    print(f"\nWeight distance between random MLPs: {dist:.4f}")

    print("\nAll model tests passed!")
