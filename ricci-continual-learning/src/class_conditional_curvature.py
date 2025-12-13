"""
Class-Conditional Curvature Preservation

Key insight: Classification performance depends on:
1. Within-class cohesion (samples of same class should cluster)
2. Between-class separation (different classes should be far apart)
3. Class-specific geometry (the "shape" of each class cluster)

Instead of preserving global curvature, we preserve:
- The curvature WITHIN each class (intra-class geometry)
- The relative positions of class centroids (inter-class structure)
- The angular relationships between classes

This is more directly related to classification than global curvature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple


class ClassConditionalCurvature(nn.Module):
    """
    Preserves the geometric structure of each class separately.

    For each class c:
    - Store the centroid μ_c
    - Store the within-class covariance structure
    - Store the k-NN distance distribution (local curvature proxy)
    - Store angular relationships between samples

    During training, regularize to preserve these class-specific structures.
    """

    def __init__(
        self,
        num_classes: int = 10,
        k_neighbors: int = 10,
        preserve_centroids: bool = True,
        preserve_spread: bool = True,
        preserve_local_structure: bool = True,
        centroid_weight: float = 1.0,
        spread_weight: float = 0.5,
        local_weight: float = 0.5
    ):
        super().__init__()

        self.num_classes = num_classes
        self.k_neighbors = k_neighbors
        self.preserve_centroids = preserve_centroids
        self.preserve_spread = preserve_spread
        self.preserve_local_structure = preserve_local_structure

        self.centroid_weight = centroid_weight
        self.spread_weight = spread_weight
        self.local_weight = local_weight

        # Reference structures for each class
        self.class_centroids: Optional[torch.Tensor] = None  # (num_classes, dim)
        self.class_spreads: Optional[torch.Tensor] = None    # (num_classes,) - avg distance to centroid
        self.class_covariances: Optional[Dict[int, torch.Tensor]] = None  # Per-class covariance
        self.class_local_dists: Optional[Dict[int, torch.Tensor]] = None  # Per-class k-NN distances

        # Inter-class structure
        self.centroid_distances: Optional[torch.Tensor] = None  # (num_classes, num_classes)
        self.centroid_angles: Optional[torch.Tensor] = None     # Angular relationships

    def _compute_class_statistics(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict, Dict]:
        """
        Compute per-class statistics.

        Returns:
            centroids: (num_classes, dim)
            spreads: (num_classes,)
            covariances: dict of (dim, dim) matrices
            local_dists: dict of k-NN distance distributions
        """
        device = embeddings.device
        dim = embeddings.shape[1]

        centroids = torch.zeros(self.num_classes, dim, device=device)
        spreads = torch.zeros(self.num_classes, device=device)
        covariances = {}
        local_dists = {}

        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() < 2:
                continue

            class_emb = embeddings[mask]
            n_c = class_emb.shape[0]

            # Centroid
            centroid = class_emb.mean(dim=0)
            centroids[c] = centroid

            # Spread (mean distance to centroid)
            dists_to_centroid = torch.norm(class_emb - centroid, dim=1)
            spreads[c] = dists_to_centroid.mean()

            # Covariance (for Mahalanobis-like structure)
            centered = class_emb - centroid
            cov = (centered.T @ centered) / (n_c - 1 + 1e-8)
            covariances[c] = cov

            # Local k-NN distances within class
            if n_c > self.k_neighbors:
                pairwise_dists = torch.cdist(class_emb, class_emb, p=2)
                k = min(self.k_neighbors, n_c - 1)
                knn_dists, _ = torch.topk(pairwise_dists, k + 1, largest=False)
                knn_dists = knn_dists[:, 1:]  # Remove self

                # Store normalized distance distribution (scale-invariant)
                mean_knn = knn_dists.mean(dim=1, keepdim=True) + 1e-8
                local_dists[c] = (knn_dists / mean_knn).mean(dim=0)  # Average over samples

        return centroids, spreads, covariances, local_dists

    def _compute_inter_class_structure(
        self,
        centroids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute structure between class centroids.

        Returns:
            distances: (num_classes, num_classes) pairwise distances
            angles: Angular signature of centroid arrangement
        """
        # Pairwise distances between centroids
        distances = torch.cdist(centroids, centroids, p=2)

        # Angular structure: cosine similarities between centroid vectors
        # (relative to global centroid)
        global_centroid = centroids.mean(dim=0, keepdim=True)
        centered = centroids - global_centroid
        normalized = F.normalize(centered, dim=1, eps=1e-8)
        angles = normalized @ normalized.T

        return distances, angles

    def set_reference(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Store reference class-conditional structure from Task A.
        """
        with torch.no_grad():
            # Compute per-class statistics
            centroids, spreads, covariances, local_dists = self._compute_class_statistics(
                embeddings, labels
            )

            self.class_centroids = centroids.detach().clone()
            self.class_spreads = spreads.detach().clone()
            self.class_covariances = {k: v.detach().clone() for k, v in covariances.items()}
            self.class_local_dists = {k: v.detach().clone() for k, v in local_dists.items()}

            # Compute inter-class structure
            distances, angles = self._compute_inter_class_structure(centroids)
            self.centroid_distances = distances.detach().clone()
            self.centroid_angles = angles.detach().clone()

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute class-conditional curvature preservation loss.

        If labels are provided, use them for per-class computation.
        If not, use soft assignment based on distance to stored centroids.
        """
        if self.class_centroids is None:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        device = embeddings.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        if labels is not None:
            # Compute current class statistics
            curr_centroids, curr_spreads, _, curr_local_dists = self._compute_class_statistics(
                embeddings, labels
            )

            # === Centroid Preservation Loss ===
            if self.preserve_centroids:
                # Preserve relative positions of centroids (not absolute)
                curr_dists, curr_angles = self._compute_inter_class_structure(curr_centroids)

                # Distance structure preservation (normalized)
                ref_dist_norm = self.centroid_distances / (self.centroid_distances.mean() + 1e-8)
                curr_dist_norm = curr_dists / (curr_dists.mean() + 1e-8)
                centroid_loss = F.mse_loss(curr_dist_norm, ref_dist_norm)

                # Angular structure preservation
                angle_loss = F.mse_loss(curr_angles, self.centroid_angles)

                total_loss = total_loss + self.centroid_weight * (centroid_loss + angle_loss)

            # === Spread Preservation Loss ===
            if self.preserve_spread:
                # Preserve relative spreads (scale-invariant)
                ref_spread_norm = self.class_spreads / (self.class_spreads.mean() + 1e-8)
                curr_spread_norm = curr_spreads / (curr_spreads.mean() + 1e-8)
                spread_loss = F.mse_loss(curr_spread_norm, ref_spread_norm)

                total_loss = total_loss + self.spread_weight * spread_loss

            # === Local Structure Preservation Loss ===
            if self.preserve_local_structure:
                local_loss = torch.tensor(0.0, device=device)
                n_classes_with_local = 0

                for c in self.class_local_dists:
                    if c in curr_local_dists:
                        ref_local = self.class_local_dists[c]
                        curr_local = curr_local_dists[c]

                        # Match sizes
                        min_k = min(ref_local.shape[0], curr_local.shape[0])
                        local_loss = local_loss + F.mse_loss(
                            curr_local[:min_k], ref_local[:min_k]
                        )
                        n_classes_with_local += 1

                if n_classes_with_local > 0:
                    local_loss = local_loss / n_classes_with_local
                    total_loss = total_loss + self.local_weight * local_loss

        else:
            # No labels provided - use soft assignment
            # Compute distances to reference centroids
            dists_to_centroids = torch.cdist(embeddings, self.class_centroids, p=2)

            # Soft assignment (softmax over negative distances)
            soft_assignments = F.softmax(-dists_to_centroids, dim=1)

            # Soft centroid computation
            curr_centroids = soft_assignments.T @ embeddings  # (num_classes, dim)
            curr_centroids = curr_centroids / (soft_assignments.sum(dim=0, keepdim=True).T + 1e-8)

            # Preserve inter-class structure
            curr_dists, curr_angles = self._compute_inter_class_structure(curr_centroids)

            ref_dist_norm = self.centroid_distances / (self.centroid_distances.mean() + 1e-8)
            curr_dist_norm = curr_dists / (curr_dists.mean() + 1e-8)

            centroid_loss = F.mse_loss(curr_dist_norm, ref_dist_norm)
            angle_loss = F.mse_loss(curr_angles, self.centroid_angles)

            total_loss = total_loss + self.centroid_weight * (centroid_loss + angle_loss)

        return total_loss


class ClassCentroidRegularizer(nn.Module):
    """
    Simpler version: just preserve class centroid structure.

    This preserves:
    1. Relative distances between class centroids
    2. Angular arrangement of classes
    3. Relative class spreads (how "tight" each class is)

    More efficient and may be sufficient for preventing forgetting.
    """

    def __init__(
        self,
        num_classes: int = 10,
        distance_weight: float = 1.0,
        angle_weight: float = 1.0,
        spread_weight: float = 0.5
    ):
        super().__init__()

        self.num_classes = num_classes
        self.distance_weight = distance_weight
        self.angle_weight = angle_weight
        self.spread_weight = spread_weight

        self.ref_centroids = None
        self.ref_spreads = None
        self.ref_centroid_dists = None
        self.ref_centroid_angles = None

    def set_reference(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Store reference class structure."""
        with torch.no_grad():
            device = embeddings.device
            dim = embeddings.shape[1]

            centroids = torch.zeros(self.num_classes, dim, device=device)
            spreads = torch.zeros(self.num_classes, device=device)

            for c in range(self.num_classes):
                mask = labels == c
                if mask.sum() > 0:
                    class_emb = embeddings[mask]
                    centroid = class_emb.mean(dim=0)
                    centroids[c] = centroid
                    spreads[c] = torch.norm(class_emb - centroid, dim=1).mean()

            self.ref_centroids = centroids.detach().clone()
            self.ref_spreads = spreads.detach().clone()
            self.ref_centroid_dists = torch.cdist(centroids, centroids, p=2).detach().clone()

            # Angular structure
            global_centroid = centroids.mean(dim=0, keepdim=True)
            centered = centroids - global_centroid
            normalized = F.normalize(centered, dim=1, eps=1e-8)
            self.ref_centroid_angles = (normalized @ normalized.T).detach().clone()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute class structure preservation loss."""
        if self.ref_centroids is None:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        device = embeddings.device
        dim = embeddings.shape[1]

        # Compute current class statistics
        curr_centroids = torch.zeros(self.num_classes, dim, device=device)
        curr_spreads = torch.zeros(self.num_classes, device=device)

        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                class_emb = embeddings[mask]
                centroid = class_emb.mean(dim=0)
                curr_centroids[c] = centroid
                curr_spreads[c] = torch.norm(class_emb - centroid, dim=1).mean()

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # === Distance Structure Loss ===
        curr_dists = torch.cdist(curr_centroids, curr_centroids, p=2)

        # Normalize for scale invariance
        ref_dist_norm = self.ref_centroid_dists / (self.ref_centroid_dists.mean() + 1e-8)
        curr_dist_norm = curr_dists / (curr_dists.mean() + 1e-8)

        dist_loss = F.mse_loss(curr_dist_norm, ref_dist_norm)
        total_loss = total_loss + self.distance_weight * dist_loss

        # === Angular Structure Loss ===
        global_centroid = curr_centroids.mean(dim=0, keepdim=True)
        centered = curr_centroids - global_centroid
        normalized = F.normalize(centered, dim=1, eps=1e-8)
        curr_angles = normalized @ normalized.T

        angle_loss = F.mse_loss(curr_angles, self.ref_centroid_angles)
        total_loss = total_loss + self.angle_weight * angle_loss

        # === Spread Preservation Loss ===
        ref_spread_norm = self.ref_spreads / (self.ref_spreads.mean() + 1e-8)
        curr_spread_norm = curr_spreads / (curr_spreads.mean() + 1e-8)

        spread_loss = F.mse_loss(curr_spread_norm, ref_spread_norm)
        total_loss = total_loss + self.spread_weight * spread_loss

        return total_loss


class PrototypeRegularizer(nn.Module):
    """
    Prototype-based regularization.

    Store class prototypes (centroids) and regularize:
    1. Samples of class c should be close to prototype c
    2. Samples of class c should be far from prototypes of other classes
    3. Relative distances between prototypes should be preserved

    This is related to prototypical networks and has theoretical grounding
    in metric learning.
    """

    def __init__(
        self,
        num_classes: int = 10,
        attraction_weight: float = 1.0,
        repulsion_weight: float = 0.5,
        structure_weight: float = 1.0,
        temperature: float = 1.0
    ):
        super().__init__()

        self.num_classes = num_classes
        self.attraction_weight = attraction_weight
        self.repulsion_weight = repulsion_weight
        self.structure_weight = structure_weight
        self.temperature = temperature

        self.prototypes = None
        self.prototype_dists = None

    def set_reference(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """Compute and store class prototypes."""
        with torch.no_grad():
            device = embeddings.device
            dim = embeddings.shape[1]

            prototypes = torch.zeros(self.num_classes, dim, device=device)

            for c in range(self.num_classes):
                mask = labels == c
                if mask.sum() > 0:
                    prototypes[c] = embeddings[mask].mean(dim=0)

            self.prototypes = prototypes.detach().clone()
            self.prototype_dists = torch.cdist(prototypes, prototypes, p=2).detach().clone()

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute prototype-based regularization loss.

        This encourages:
        1. Samples to stay close to their class prototype
        2. Samples to stay far from other class prototypes
        3. Prototype structure to be preserved
        """
        if self.prototypes is None:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        device = embeddings.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # Distances from each sample to each prototype
        dists_to_prototypes = torch.cdist(embeddings, self.prototypes, p=2)

        # === Attraction Loss ===
        # Samples should be close to their own class prototype
        if self.attraction_weight > 0:
            attraction_loss = torch.tensor(0.0, device=device)
            for c in range(self.num_classes):
                mask = labels == c
                if mask.sum() > 0:
                    # Distance to own prototype
                    own_dists = dists_to_prototypes[mask, c]
                    attraction_loss = attraction_loss + own_dists.mean()

            attraction_loss = attraction_loss / self.num_classes
            total_loss = total_loss + self.attraction_weight * attraction_loss

        # === Repulsion Loss ===
        # Samples should be far from other class prototypes
        if self.repulsion_weight > 0:
            repulsion_loss = torch.tensor(0.0, device=device)
            for c in range(self.num_classes):
                mask = labels == c
                if mask.sum() > 0:
                    # Distances to other prototypes
                    other_dists = dists_to_prototypes[mask]
                    other_dists[:, c] = float('inf')  # Exclude own class

                    # Negative log of min distance (want to maximize min distance)
                    min_other_dist = other_dists.min(dim=1)[0]
                    repulsion_loss = repulsion_loss - torch.log(min_other_dist + 1e-8).mean()

            repulsion_loss = repulsion_loss / self.num_classes
            total_loss = total_loss + self.repulsion_weight * repulsion_loss

        # === Structure Loss ===
        # Preserve relative distances between prototypes
        if self.structure_weight > 0:
            # Compute current prototype positions (from current embeddings)
            curr_prototypes = torch.zeros_like(self.prototypes)
            for c in range(self.num_classes):
                mask = labels == c
                if mask.sum() > 0:
                    curr_prototypes[c] = embeddings[mask].mean(dim=0)

            curr_dists = torch.cdist(curr_prototypes, curr_prototypes, p=2)

            # Normalize for scale invariance
            ref_norm = self.prototype_dists / (self.prototype_dists.mean() + 1e-8)
            curr_norm = curr_dists / (curr_dists.mean() + 1e-8)

            structure_loss = F.mse_loss(curr_norm, ref_norm)
            total_loss = total_loss + self.structure_weight * structure_loss

        return total_loss


if __name__ == "__main__":
    print("Testing class-conditional curvature preservation...")

    torch.manual_seed(42)

    # Create synthetic class data
    num_samples = 500
    num_classes = 10
    dim = 64

    # Generate class clusters
    embeddings = []
    labels = []

    for c in range(num_classes):
        # Each class is a cluster at a different location
        center = torch.randn(dim) * 5
        samples = torch.randn(num_samples // num_classes, dim) * 0.5 + center
        embeddings.append(samples)
        labels.extend([c] * (num_samples // num_classes))

    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.tensor(labels)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")

    # Test ClassConditionalCurvature
    print("\n1. Testing ClassConditionalCurvature...")
    ccc = ClassConditionalCurvature(num_classes=num_classes)
    ccc.set_reference(embeddings, labels)

    # Small perturbation
    perturbed = embeddings + torch.randn_like(embeddings) * 0.1
    loss_small = ccc(perturbed, labels)
    print(f"   Small perturbation loss: {loss_small.item():.4f}")

    # Large perturbation
    perturbed_large = embeddings + torch.randn_like(embeddings) * 1.0
    loss_large = ccc(perturbed_large, labels)
    print(f"   Large perturbation loss: {loss_large.item():.4f}")

    # Shuffle labels (should have high loss)
    shuffled_labels = labels[torch.randperm(len(labels))]
    loss_shuffled = ccc(embeddings, shuffled_labels)
    print(f"   Shuffled labels loss: {loss_shuffled.item():.4f}")

    # Test ClassCentroidRegularizer
    print("\n2. Testing ClassCentroidRegularizer...")
    ccr = ClassCentroidRegularizer(num_classes=num_classes)
    ccr.set_reference(embeddings, labels)

    loss_small = ccr(perturbed, labels)
    print(f"   Small perturbation loss: {loss_small.item():.4f}")

    loss_large = ccr(perturbed_large, labels)
    print(f"   Large perturbation loss: {loss_large.item():.4f}")

    # Test PrototypeRegularizer
    print("\n3. Testing PrototypeRegularizer...")
    pr = PrototypeRegularizer(num_classes=num_classes)
    pr.set_reference(embeddings, labels)

    loss_small = pr(perturbed, labels)
    print(f"   Small perturbation loss: {loss_small.item():.4f}")

    loss_large = pr(perturbed_large, labels)
    print(f"   Large perturbation loss: {loss_large.item():.4f}")

    print("\nAll tests passed!")
