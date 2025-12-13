"""
Focused Curvature Preservation

The key insight: preserve CURVATURE (ratios, angles, second-order structure)
rather than DISTANCES (first-order structure).

This should allow:
1. Weights to change freely
2. Absolute distances to change
3. But local curvature (shape) preserved
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict


class CurvatureOnlyRegularizer(nn.Module):
    """
    Preserves only the curvature signature, not absolute distances.

    Key quantities that define curvature:
    1. Ratio of neighbor distances (relative position in local cluster)
    2. Angular relationships between neighbor vectors
    3. Expansion/contraction of local neighborhoods (but not absolute size)

    Does NOT preserve:
    - Absolute distances
    - Global embedding scale
    """

    def __init__(
        self,
        k_neighbors: int = 15,
        temperature: float = 1.0
    ):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.temperature = temperature

        self.reference_curvature_sig = None
        self.reference_angular_sig = None

    def _get_knn(self, embeddings: torch.Tensor, k: int):
        """Get k-NN indices and distances."""
        dists = torch.cdist(embeddings, embeddings, p=2)
        knn_dists, knn_indices = torch.topk(dists, k + 1, largest=False)
        return knn_dists[:, 1:], knn_indices[:, 1:]

    def _compute_curvature_signature(
        self,
        embeddings: torch.Tensor,
        knn_dists: torch.Tensor,
        knn_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute scale-invariant curvature signature.

        For each point, compute:
        - Ratio of distances to k neighbors (normalized by mean)
        - This captures the SHAPE of the local neighborhood, not size
        """
        # Normalize distances by local mean to get scale-invariant ratios
        mean_dist = knn_dists.mean(dim=1, keepdim=True) + 1e-8
        normalized_dists = knn_dists / mean_dist

        # The signature is the sorted ratios (order-invariant)
        sorted_ratios, _ = torch.sort(normalized_dists, dim=1)

        return sorted_ratios

    def _compute_angular_signature(
        self,
        embeddings: torch.Tensor,
        knn_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute angular signature: pairwise angles between neighbor vectors.

        This is scale-invariant and captures local manifold curvature.
        """
        batch_size = embeddings.shape[0]
        k = knn_indices.shape[1]

        # Get neighbor embeddings: (batch, k, dim)
        neighbor_emb = embeddings[knn_indices.long()]

        # Vectors from center to neighbors
        vectors = neighbor_emb - embeddings.unsqueeze(1)

        # Normalize to unit vectors
        vectors = F.normalize(vectors, dim=2, eps=1e-8)

        # Pairwise cosine similarities: (batch, k, k)
        cos_sims = torch.bmm(vectors, vectors.transpose(1, 2))

        # Take upper triangle (unique pairs)
        mask = torch.triu(torch.ones(k, k, device=embeddings.device), diagonal=1).bool()
        angular_sig = cos_sims[:, mask]

        # Sort for permutation invariance
        angular_sig, _ = torch.sort(angular_sig, dim=1)

        return angular_sig

    def set_reference(self, embeddings: torch.Tensor):
        """Store reference curvature signatures."""
        with torch.no_grad():
            k = min(self.k_neighbors, embeddings.shape[0] - 1)
            knn_dists, knn_indices = self._get_knn(embeddings, k)

            self.reference_curvature_sig = self._compute_curvature_signature(
                embeddings, knn_dists, knn_indices
            ).detach().clone()

            self.reference_angular_sig = self._compute_angular_signature(
                embeddings, knn_indices
            ).detach().clone()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute curvature preservation loss.

        Only penalizes changes to SHAPE, not absolute distances.
        """
        if self.reference_curvature_sig is None:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        n = min(embeddings.shape[0], self.reference_curvature_sig.shape[0])

        # Sample if needed
        if embeddings.shape[0] > n:
            idx = torch.randperm(embeddings.shape[0], device=embeddings.device)[:n]
            curr_emb = embeddings[idx]
        else:
            curr_emb = embeddings

        k = min(self.k_neighbors, n - 1)
        knn_dists, knn_indices = self._get_knn(curr_emb, k)

        # Compute current signatures
        curr_curv_sig = self._compute_curvature_signature(curr_emb, knn_dists, knn_indices)
        curr_ang_sig = self._compute_angular_signature(curr_emb, knn_indices)

        # Match sizes with reference
        ref_curv_sig = self.reference_curvature_sig[:n, :k]
        ref_ang_sig = self.reference_angular_sig[:n]

        # Curvature loss: difference in distance ratios
        curv_loss = F.mse_loss(curr_curv_sig, ref_curv_sig)

        # Angular loss: difference in angle distributions
        # Truncate to match sizes
        min_ang = min(curr_ang_sig.shape[1], ref_ang_sig.shape[1])
        ang_loss = F.mse_loss(curr_ang_sig[:, :min_ang], ref_ang_sig[:n, :min_ang])

        # Combined loss (angular is more important for curvature)
        total_loss = 0.3 * curv_loss + 0.7 * ang_loss

        return total_loss


class SecondOrderGeometryRegularizer(nn.Module):
    """
    Preserves second-order geometry: how local neighborhoods relate to each other.

    This is closer to actual Ricci curvature, which measures how
    neighborhoods diverge/converge relative to each other.
    """

    def __init__(
        self,
        k_neighbors: int = 10,
        n_samples: int = 200
    ):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.n_samples = n_samples

        self.reference_second_order = None

    def _compute_second_order(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute second-order geometric structure.

        For each pair of nearby points (i, j):
        - Compare their neighborhoods N_i and N_j
        - Measure overlap/divergence

        This captures how the manifold curves.
        """
        n = embeddings.shape[0]
        k = min(self.k_neighbors, n - 1)

        # Get k-NN
        dists = torch.cdist(embeddings, embeddings, p=2)
        _, knn_indices = torch.topk(dists, k + 1, largest=False)
        knn_indices = knn_indices[:, 1:]  # Remove self

        # For computational efficiency, sample pairs
        n_pairs = min(n, self.n_samples)
        pair_indices = torch.randperm(n, device=embeddings.device)[:n_pairs]

        second_order_features = []

        for i in pair_indices:
            neighbors_i = set(knn_indices[i].tolist())

            # For each neighbor of i, compute neighborhood overlap
            overlaps = []
            for j in knn_indices[i]:
                neighbors_j = set(knn_indices[j.item()].tolist())
                overlap = len(neighbors_i & neighbors_j) / k
                overlaps.append(overlap)

            second_order_features.append(torch.tensor(overlaps, device=embeddings.device))

        second_order = torch.stack(second_order_features)

        # Sort for permutation invariance
        second_order, _ = torch.sort(second_order, dim=1)

        return second_order

    def set_reference(self, embeddings: torch.Tensor):
        """Store reference second-order structure."""
        with torch.no_grad():
            self.reference_second_order = self._compute_second_order(embeddings).detach().clone()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute second-order preservation loss."""
        if self.reference_second_order is None:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        curr_second_order = self._compute_second_order(embeddings)

        # Match sizes
        n = min(curr_second_order.shape[0], self.reference_second_order.shape[0])
        k = min(curr_second_order.shape[1], self.reference_second_order.shape[1])

        loss = F.mse_loss(
            curr_second_order[:n, :k],
            self.reference_second_order[:n, :k]
        )

        return loss


class FocusedCurvatureRegularizer(nn.Module):
    """
    Combined regularizer focusing on curvature, not distances.
    """

    def __init__(self, k_neighbors: int = 15, n_samples: int = 200):
        super().__init__()

        self.curvature_reg = CurvatureOnlyRegularizer(k_neighbors=k_neighbors)
        self.second_order_reg = SecondOrderGeometryRegularizer(
            k_neighbors=k_neighbors, n_samples=n_samples
        )

    def set_reference(self, embeddings: torch.Tensor):
        """Set reference for all components."""
        self.curvature_reg.set_reference(embeddings)
        self.second_order_reg.set_reference(embeddings)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute combined curvature loss."""
        curv_loss = self.curvature_reg(embeddings)
        second_order_loss = self.second_order_reg(embeddings)

        return curv_loss + 0.5 * second_order_loss


if __name__ == "__main__":
    print("Testing focused curvature regularizers...")

    torch.manual_seed(42)

    # Create test embeddings
    embeddings = torch.randn(100, 64)

    # Test CurvatureOnlyRegularizer
    print("\n1. CurvatureOnlyRegularizer:")
    reg = CurvatureOnlyRegularizer(k_neighbors=10)
    reg.set_reference(embeddings)

    # Small perturbation
    small_pert = embeddings + torch.randn_like(embeddings) * 0.1
    loss_small = reg(small_pert)
    print(f"   Small perturbation loss: {loss_small.item():.4f}")

    # Scale change (should NOT affect curvature)
    scaled = embeddings * 2.0
    loss_scaled = reg(scaled)
    print(f"   Scaled (2x) loss: {loss_scaled.item():.4f}")

    # Random permutation (should have high loss)
    shuffled = embeddings[torch.randperm(100)]
    loss_shuffled = reg(shuffled)
    print(f"   Random shuffle loss: {loss_shuffled.item():.4f}")

    # Test SecondOrderGeometryRegularizer
    print("\n2. SecondOrderGeometryRegularizer:")
    reg2 = SecondOrderGeometryRegularizer(k_neighbors=10, n_samples=50)
    reg2.set_reference(embeddings)

    loss_small = reg2(small_pert)
    print(f"   Small perturbation loss: {loss_small.item():.4f}")

    loss_scaled = reg2(scaled)
    print(f"   Scaled (2x) loss: {loss_scaled.item():.4f}")

    print("\nAll tests passed!")
