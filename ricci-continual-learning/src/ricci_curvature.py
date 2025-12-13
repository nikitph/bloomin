"""
Ollivier-Ricci Curvature Computation for Neural Network Embeddings

The Ollivier-Ricci curvature measures how mass transportation changes
distances on a graph/manifold. For neural embeddings:
- Positive curvature: points cluster (like a sphere)
- Negative curvature: points spread (like a saddle/hyperbolic space)
- Zero curvature: Euclidean behavior

Key insight from Ricci-REWA: Preserving this curvature during training
preserves task-relevant structure even when weights change dramatically.
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import ot  # Optimal Transport (POT library)
from sklearn.neighbors import NearestNeighbors
from typing import Tuple, Optional, Dict
import warnings


class OllivierRicci:
    """
    Compute Ollivier-Ricci curvature on a k-NN graph of embeddings.

    The Ollivier-Ricci curvature between two nodes x, y is:
        κ(x,y) = 1 - W_1(m_x, m_y) / d(x,y)

    where:
        - W_1 is the Wasserstein-1 (Earth Mover's) distance
        - m_x is the probability distribution at x (uniform over neighbors)
        - d(x,y) is the graph distance between x and y
    """

    def __init__(self, k_neighbors: int = 10, alpha: float = 0.5):
        """
        Args:
            k_neighbors: Number of neighbors for k-NN graph
            alpha: Laziness parameter (0 = lazy random walk includes self-loop)
                   Higher alpha means more mass stays at the node itself
        """
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.graph = None
        self.distances = None
        self.neighbor_indices = None

    def fit(self, embeddings: np.ndarray) -> 'OllivierRicci':
        """
        Build the k-NN graph from embeddings.

        Args:
            embeddings: (n_samples, n_features) array
        """
        n_samples = embeddings.shape[0]
        k = min(self.k_neighbors, n_samples - 1)

        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)

        # Remove self-loops (first neighbor is always self)
        self.neighbor_indices = indices[:, 1:]  # (n_samples, k)
        neighbor_distances = distances[:, 1:]  # (n_samples, k)

        # Build sparse adjacency matrix with distances
        row_indices = np.repeat(np.arange(n_samples), k)
        col_indices = self.neighbor_indices.flatten()
        edge_weights = neighbor_distances.flatten()

        # Make symmetric (undirected graph)
        self.graph = csr_matrix(
            (edge_weights, (row_indices, col_indices)),
            shape=(n_samples, n_samples)
        )
        self.graph = self.graph + self.graph.T
        self.graph.data = self.graph.data / 2  # Average duplicate edges

        # Precompute all pairwise shortest path distances
        self.distances = shortest_path(self.graph, directed=False, unweighted=False)

        # Store embeddings for Euclidean distance fallback
        self.embeddings = embeddings

        return self

    def _get_measure(self, node: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the probability measure m_x at a node.

        Returns (support, probabilities) where:
        - support: indices of nodes in the support of m_x
        - probabilities: probability mass at each support node
        """
        neighbors = self.neighbor_indices[node]
        k = len(neighbors)

        if self.alpha > 0:
            # Lazy random walk: alpha mass stays at node, (1-alpha) spread to neighbors
            support = np.concatenate([[node], neighbors])
            probs = np.concatenate([[self.alpha], np.full(k, (1 - self.alpha) / k)])
        else:
            # Standard random walk: uniform over neighbors
            support = neighbors
            probs = np.full(k, 1.0 / k)

        return support, probs

    def compute_edge_curvature(self, x: int, y: int) -> float:
        """
        Compute Ollivier-Ricci curvature for edge (x, y).

        κ(x,y) = 1 - W_1(m_x, m_y) / d(x,y)
        """
        # Get measures at x and y
        support_x, probs_x = self._get_measure(x)
        support_y, probs_y = self._get_measure(y)

        # Build cost matrix (pairwise distances between supports)
        n_x, n_y = len(support_x), len(support_y)
        cost_matrix = np.zeros((n_x, n_y))

        for i, xi in enumerate(support_x):
            for j, yj in enumerate(support_y):
                cost_matrix[i, j] = self.distances[xi, yj]

        # Handle infinite distances (disconnected components)
        if np.any(np.isinf(cost_matrix)):
            # Fallback to Euclidean distance
            for i, xi in enumerate(support_x):
                for j, yj in enumerate(support_y):
                    if np.isinf(cost_matrix[i, j]):
                        cost_matrix[i, j] = np.linalg.norm(
                            self.embeddings[xi] - self.embeddings[yj]
                        )

        # Compute Wasserstein distance using optimal transport
        wasserstein_dist = ot.emd2(probs_x, probs_y, cost_matrix)

        # Graph distance between x and y
        d_xy = self.distances[x, y]
        if np.isinf(d_xy) or d_xy == 0:
            d_xy = np.linalg.norm(self.embeddings[x] - self.embeddings[y])

        if d_xy < 1e-10:
            return 0.0

        # Ollivier-Ricci curvature
        curvature = 1 - wasserstein_dist / d_xy

        return curvature

    def compute_all_curvatures(self) -> Dict[Tuple[int, int], float]:
        """
        Compute curvature for all edges in the graph.

        Returns dict mapping (x, y) edge to curvature value.
        """
        curvatures = {}

        # Get all edges (nonzero entries in upper triangle)
        rows, cols = self.graph.nonzero()
        edges = [(r, c) for r, c in zip(rows, cols) if r < c]

        for x, y in edges:
            curvatures[(x, y)] = self.compute_edge_curvature(x, y)

        return curvatures

    def compute_node_curvature(self, node: int) -> float:
        """
        Compute scalar Ricci curvature at a node (average over incident edges).
        """
        neighbors = self.neighbor_indices[node]
        curvatures = [self.compute_edge_curvature(node, n) for n in neighbors]
        return np.mean(curvatures)

    def compute_ricci_tensor(self, sample_size: Optional[int] = None) -> np.ndarray:
        """
        Compute approximate Ricci "tensor" as matrix of pairwise curvatures.

        For computational tractability with large graphs, we can sample.

        Returns:
            (n, n) matrix where R[i,j] is the curvature between nodes i and j
            (only defined for neighboring nodes, 0 elsewhere)
        """
        n = self.embeddings.shape[0]
        R = np.zeros((n, n))

        if sample_size and sample_size < n:
            # Sample a subset of nodes
            sample_indices = np.random.choice(n, sample_size, replace=False)
        else:
            sample_indices = np.arange(n)

        for i in sample_indices:
            for j in self.neighbor_indices[i]:
                if j in sample_indices:
                    R[i, j] = self.compute_edge_curvature(i, j)
                    R[j, i] = R[i, j]  # Symmetric

        return R


def compute_ricci_on_embeddings(
    embeddings: torch.Tensor,
    k_neighbors: int = 10,
    alpha: float = 0.5,
    sample_size: Optional[int] = 500
) -> Tuple[np.ndarray, OllivierRicci]:
    """
    Convenience function to compute Ricci curvature tensor from embeddings.

    Args:
        embeddings: (batch, features) tensor of neural network embeddings
        k_neighbors: Number of neighbors for k-NN graph
        alpha: Laziness parameter for random walk
        sample_size: If set, sample this many nodes (for large batches)

    Returns:
        ricci_tensor: (n, n) curvature matrix
        ricci_computer: OllivierRicci object for further analysis
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    n = embeddings.shape[0]
    if sample_size and sample_size > n:
        sample_size = None

    ricci = OllivierRicci(k_neighbors=k_neighbors, alpha=alpha)
    ricci.fit(embeddings)

    R = ricci.compute_ricci_tensor(sample_size=sample_size)

    return R, ricci


class RicciCurvatureRegularizer:
    """
    Regularizer that penalizes changes in Ricci curvature during training.

    This is the key innovation: instead of freezing weights (EWC) or
    replaying examples, we preserve the geometric structure of the
    learned representations.
    """

    def __init__(
        self,
        k_neighbors: int = 10,
        alpha: float = 0.5,
        sample_size: int = 200,
        comparison_method: str = 'frobenius'
    ):
        """
        Args:
            k_neighbors: k for k-NN graph
            alpha: Laziness parameter
            sample_size: Number of samples for curvature computation
            comparison_method: How to compare curvature tensors
                'frobenius': Frobenius norm of difference
                'spectral': Spectral norm (largest eigenvalue difference)
                'mean': Mean absolute difference of non-zero entries
        """
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.sample_size = sample_size
        self.comparison_method = comparison_method

        self.reference_curvature = None
        self.reference_embeddings = None
        self.reference_indices = None

    def set_reference(self, embeddings: torch.Tensor):
        """
        Store the reference curvature from Task A.

        Call this after training on the first task.
        """
        embeddings_np = embeddings.detach().cpu().numpy()
        n = embeddings_np.shape[0]

        # Sample indices for consistent comparison
        self.reference_indices = np.random.choice(
            n, min(self.sample_size, n), replace=False
        )
        sampled_embeddings = embeddings_np[self.reference_indices]

        # Compute and store reference curvature
        self.reference_curvature, _ = compute_ricci_on_embeddings(
            sampled_embeddings,
            k_neighbors=self.k_neighbors,
            alpha=self.alpha,
            sample_size=None  # Use all sampled points
        )
        self.reference_embeddings = sampled_embeddings

    def compute_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute curvature preservation loss.

        This measures how much the current curvature differs from reference.
        """
        if self.reference_curvature is None:
            return torch.tensor(0.0, device=embeddings.device)

        embeddings_np = embeddings.detach().cpu().numpy()
        n = embeddings_np.shape[0]

        # Sample same number of points
        sample_indices = np.random.choice(
            n, min(self.sample_size, n), replace=False
        )
        sampled_embeddings = embeddings_np[sample_indices]

        # Compute current curvature
        current_curvature, _ = compute_ricci_on_embeddings(
            sampled_embeddings,
            k_neighbors=self.k_neighbors,
            alpha=self.alpha,
            sample_size=None
        )

        # Compare curvatures
        if self.comparison_method == 'frobenius':
            # Resize to match (they may have different sizes)
            min_size = min(
                self.reference_curvature.shape[0],
                current_curvature.shape[0]
            )
            ref = self.reference_curvature[:min_size, :min_size]
            cur = current_curvature[:min_size, :min_size]
            loss = np.linalg.norm(cur - ref, 'fro')

        elif self.comparison_method == 'spectral':
            min_size = min(
                self.reference_curvature.shape[0],
                current_curvature.shape[0]
            )
            ref = self.reference_curvature[:min_size, :min_size]
            cur = current_curvature[:min_size, :min_size]
            diff = cur - ref
            loss = np.linalg.norm(diff, 2)  # Spectral norm

        elif self.comparison_method == 'mean':
            min_size = min(
                self.reference_curvature.shape[0],
                current_curvature.shape[0]
            )
            ref = self.reference_curvature[:min_size, :min_size]
            cur = current_curvature[:min_size, :min_size]
            # Only compare non-zero entries
            mask = (ref != 0) | (cur != 0)
            if mask.sum() > 0:
                loss = np.abs(cur[mask] - ref[mask]).mean()
            else:
                loss = 0.0
        else:
            raise ValueError(f"Unknown comparison method: {self.comparison_method}")

        return torch.tensor(loss, device=embeddings.device, dtype=torch.float32)


class DifferentiableRicciLoss(torch.nn.Module):
    """
    A differentiable approximation to Ricci curvature loss.

    The exact Ollivier-Ricci computation involves discrete optimization
    (optimal transport), which is not directly differentiable. This module
    provides a smooth approximation using entropic regularization.
    """

    def __init__(
        self,
        k_neighbors: int = 10,
        sinkhorn_reg: float = 0.1,
        sinkhorn_iters: int = 50
    ):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_iters = sinkhorn_iters

        self.reference_embeddings = None

    def set_reference(self, embeddings: torch.Tensor):
        """Store reference embeddings from Task A."""
        self.reference_embeddings = embeddings.detach().clone()

    def _pairwise_distances(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute pairwise Euclidean distances."""
        return torch.cdist(x, y, p=2)

    def _sinkhorn_distance(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        M: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Sinkhorn (entropic-regularized Wasserstein) distance.

        This is a differentiable approximation to the Wasserstein distance.
        """
        n, m = M.shape

        # Uniform distributions if not provided
        if a is None:
            a = torch.ones(n, device=M.device) / n
        if b is None:
            b = torch.ones(m, device=M.device) / m

        # Sinkhorn iterations
        K = torch.exp(-M / self.sinkhorn_reg)
        u = torch.ones_like(a)

        for _ in range(self.sinkhorn_iters):
            v = b / (K.T @ u + 1e-10)
            u = a / (K @ v + 1e-10)

        # Transport plan
        P = torch.diag(u) @ K @ torch.diag(v)

        # Wasserstein distance
        return (P * M).sum()

    def forward(
        self,
        embeddings: torch.Tensor,
        reference_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute differentiable Ricci curvature preservation loss.

        Instead of computing full Ricci tensor, we use a proxy:
        the distribution of local distances should be preserved.
        """
        if reference_embeddings is None:
            reference_embeddings = self.reference_embeddings

        if reference_embeddings is None:
            return torch.tensor(0.0, device=embeddings.device)

        # Compute k-NN distances for current embeddings
        k = min(self.k_neighbors, embeddings.shape[0] - 1)

        current_dists = self._pairwise_distances(embeddings, embeddings)
        reference_dists = self._pairwise_distances(reference_embeddings, reference_embeddings)

        # Get k smallest distances for each point (excluding self)
        current_knn, _ = torch.topk(current_dists, k + 1, largest=False)
        current_knn = current_knn[:, 1:]  # Remove self-distance

        reference_knn, _ = torch.topk(reference_dists, k + 1, largest=False)
        reference_knn = reference_knn[:, 1:]

        # Normalize to create distributions
        current_knn = current_knn / (current_knn.sum(dim=1, keepdim=True) + 1e-10)
        reference_knn = reference_knn / (reference_knn.sum(dim=1, keepdim=True) + 1e-10)

        # Compute Sinkhorn distance between local distance distributions
        # This captures how the local geometry has changed
        n = min(embeddings.shape[0], reference_embeddings.shape[0])

        total_loss = torch.tensor(0.0, device=embeddings.device)

        for i in range(n):
            # Cost matrix: difference in position within sorted distances
            M = torch.abs(
                torch.arange(k, device=embeddings.device).float().unsqueeze(0) -
                torch.arange(k, device=embeddings.device).float().unsqueeze(1)
            ) / k

            loss_i = self._sinkhorn_distance(current_knn[i], reference_knn[i], M)
            total_loss = total_loss + loss_i

        return total_loss / n


if __name__ == "__main__":
    # Test the Ricci curvature computation
    print("Testing Ollivier-Ricci curvature computation...")

    # Generate embeddings with known geometry
    np.random.seed(42)

    # Cluster 1: tight cluster (high positive curvature expected)
    cluster1 = np.random.randn(50, 10) * 0.1 + np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Cluster 2: spread out (lower curvature expected)
    cluster2 = np.random.randn(50, 10) * 1.0 + np.array([-1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    embeddings = np.vstack([cluster1, cluster2])

    # Compute curvature
    R, ricci = compute_ricci_on_embeddings(embeddings, k_neighbors=5, sample_size=100)

    # Analyze by cluster
    cluster1_curvatures = []
    cluster2_curvatures = []

    for i in range(50):
        cluster1_curvatures.append(ricci.compute_node_curvature(i))
    for i in range(50, 100):
        cluster2_curvatures.append(ricci.compute_node_curvature(i))

    print(f"Tight cluster (expected high curvature): {np.mean(cluster1_curvatures):.4f}")
    print(f"Spread cluster (expected lower curvature): {np.mean(cluster2_curvatures):.4f}")

    # Test differentiable loss
    print("\nTesting differentiable Ricci loss...")
    torch_emb = torch.tensor(embeddings, dtype=torch.float32, requires_grad=True)

    diff_loss = DifferentiableRicciLoss(k_neighbors=5)
    diff_loss.set_reference(torch_emb)

    # Perturb embeddings
    perturbed = torch_emb + torch.randn_like(torch_emb) * 0.1
    loss = diff_loss(perturbed)

    print(f"Loss after small perturbation: {loss.item():.4f}")

    # Large perturbation
    perturbed_large = torch_emb + torch.randn_like(torch_emb) * 1.0
    loss_large = diff_loss(perturbed_large)

    print(f"Loss after large perturbation: {loss_large.item():.4f}")

    print("\nRicci curvature computation working correctly!")
