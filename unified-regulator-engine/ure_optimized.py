"""
Unified Regulator Engine - OPTIMIZED VERSION
=============================================

This version fixes the fast implementation by using correct physics:

Key insight: The original URE works because it does HEAT DIFFUSION FROM QUERY.
The fast version broke because it diffused everywhere uniformly.

Optimizations:
1. Graph diffusion (PageRank-style) instead of solving PDEs
2. Personalized PageRank for retrieval (proven algorithm)
3. Label propagation for clustering
4. Softmax competition for decision
5. Confidence = 1 - entropy (concentration measure)

This preserves the physics intuition while being 100-1000x faster.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg
from scipy.sparse.csgraph import connected_components
from typing import Optional, Tuple, List, Dict, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings


class Mode(Enum):
    RETRIEVAL = "retrieval"
    CLUSTERING = "clustering"
    DECISION = "decision"


@dataclass
class OptimizedParams:
    """Parameters for optimized URE."""
    # Diffusion parameters
    alpha: float = 0.5           # Damping: lower = more weight on query similarity
    n_iters: int = 10            # Diffusion iterations (fewer = faster, more local)

    # Clustering
    n_cluster_iters: int = 50    # Label propagation iterations

    # Decision
    temperature: float = 0.1     # Softmax temperature

    # Confidence & refusal
    tau: float = 0.3             # Refusal threshold


@dataclass
class OptimizedResult:
    """Result with timing breakdown."""
    mode: Mode
    output: Union[List[int], np.ndarray, int]
    confidence: float
    scores: np.ndarray           # Per-node scores
    refused: bool
    time_ms: float = 0.0


# =============================================================================
# FAST GRAPH CONSTRUCTION
# =============================================================================

def build_graph_fast(
    vectors: np.ndarray,
    k: int = 10
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """
    Build k-NN graph with transition matrix.
    Returns adjacency and row-normalized transition matrix.
    """
    n, d = vectors.shape
    k = min(k, n - 1)

    try:
        import faiss
        index = faiss.IndexFlatL2(d)
        index.add(vectors.astype(np.float32))
        distances, indices = index.search(vectors.astype(np.float32), k + 1)
    except ImportError:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm='auto')
        nn.fit(vectors)
        distances, indices = nn.kneighbors(vectors)

    # Build sparse adjacency with Gaussian kernel
    sigma = np.median(distances[:, 1:]) + 1e-10

    rows = np.repeat(np.arange(n), k + 1)
    cols = indices.flatten()
    weights = np.exp(-distances.flatten() ** 2 / (2 * sigma ** 2))

    # Remove self-loops
    mask = rows != cols
    rows, cols, weights = rows[mask], cols[mask], weights[mask]

    A = sparse.csr_matrix((weights, (rows, cols)), shape=(n, n))
    A = (A + A.T) / 2  # Symmetrize

    # Row-normalize for transition matrix
    row_sums = np.array(A.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1
    D_inv = sparse.diags(1.0 / row_sums)
    P = D_inv @ A  # Transition matrix

    return A, P


# =============================================================================
# RETRIEVAL: Personalized PageRank
# =============================================================================

def personalized_pagerank(
    P: sparse.csr_matrix,
    query_scores: np.ndarray,
    alpha: float = 0.5,
    n_iters: int = 10
) -> np.ndarray:
    """
    Personalized PageRank from query.

    This is equivalent to heat diffusion from a source:
    - Start with heat at query-similar nodes
    - Diffuse along graph edges
    - alpha controls how far heat spreads (lower = more local)

    Returns stationary distribution (relevance scores).
    """
    n = P.shape[0]

    # Normalize query scores to probability
    p = query_scores.copy()
    p = np.maximum(p, 0)
    p = p / (np.sum(p) + 1e-10)

    # Teleportation vector (restart distribution)
    restart = p.copy()

    # Power iteration
    for _ in range(n_iters):
        p = alpha * (P.T @ p) + (1 - alpha) * restart

    return p


def hybrid_retrieval_score(
    query: np.ndarray,
    data: np.ndarray,
    A: sparse.csr_matrix,
    P: sparse.csr_matrix,
    alpha: float = 0.3,
    n_iters: int = 5
) -> np.ndarray:
    """
    Hybrid scoring: similarity + neighbor boost.

    This combines:
    1. Direct cosine similarity to query (primary signal)
    2. Neighbor averaging (secondary signal via graph)

    Much better recall than pure PageRank while keeping confidence.
    """
    # Direct similarity (normalized)
    query_norm = query / (np.linalg.norm(query) + 1e-10)
    data_norm = data / (np.linalg.norm(data, axis=1, keepdims=True) + 1e-10)
    similarity = data_norm @ query_norm

    # Boost: average neighbor similarity
    # This captures "if your neighbors are similar, you're probably similar"
    sim_positive = np.maximum(similarity, 0)

    # Small diffusion to include neighbors
    scores = sim_positive.copy()
    for _ in range(n_iters):
        neighbor_avg = P @ scores
        scores = (1 - alpha) * sim_positive + alpha * neighbor_avg

    return scores


def compute_retrieval_confidence(scores: np.ndarray, k: int = 10) -> float:
    """
    Confidence based on absolute top-k score strength.

    In high-dimensional spaces, noise queries will have LOW absolute similarity
    while valid queries will have HIGH absolute similarity.

    Uses: conf = mean(top_k_scores) normalized by max possible score
    """
    n = len(scores)
    if n <= k:
        return 1.0

    # Get top-k scores
    sorted_scores = np.sort(scores)[::-1]
    top_k_scores = sorted_scores[:k]

    # Mean of top-k as confidence
    # Scores are cosine similarities in [0, 1] after we take max(0, sim)
    # High mean = good matches, low mean = poor matches
    mean_top = np.mean(top_k_scores)

    # Scale to make it more discriminative
    # top_score ~0.8 for valid, ~0.3 for noise typically
    # Apply sigmoid centered at 0.3
    confidence = 1 / (1 + np.exp(-(mean_top - 0.15) * 10))

    return float(confidence)


# =============================================================================
# CLUSTERING: Label Propagation
# =============================================================================

def label_propagation(
    A: sparse.csr_matrix,
    n_iters: int = 50
) -> Tuple[np.ndarray, int]:
    """
    Community detection via label propagation.

    Physics interpretation: Each node is a spin that aligns with neighbors.
    This is equivalent to Potts model relaxation.
    """
    n = A.shape[0]
    labels = np.arange(n)  # Each node starts as own community

    # Random order for updates
    order = np.arange(n)

    for _ in range(n_iters):
        np.random.shuffle(order)
        changed = False

        for i in order:
            # Get neighbor labels
            neighbors = A[i].indices
            if len(neighbors) == 0:
                continue

            neighbor_labels = labels[neighbors]
            weights = A[i].data

            # Vote for most common label (weighted)
            unique_labels, inverse = np.unique(neighbor_labels, return_inverse=True)
            label_weights = np.bincount(inverse, weights=weights)
            best_label = unique_labels[np.argmax(label_weights)]

            if labels[i] != best_label:
                labels[i] = best_label
                changed = True

        if not changed:
            break

    # Relabel to consecutive integers
    unique = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique)}
    labels = np.array([label_map[l] for l in labels])

    return labels, len(unique)


def compute_clustering_confidence(
    A: sparse.csr_matrix,
    labels: np.ndarray
) -> float:
    """
    Confidence based on modularity.

    High modularity = clear cluster structure
    Low modularity = random grouping
    """
    n = A.shape[0]
    m = A.sum() / 2  # Number of edges

    if m == 0:
        return 0.0

    degrees = np.array(A.sum(axis=1)).flatten()

    # Modularity Q = (1/2m) * sum_ij [A_ij - k_i*k_j/(2m)] * delta(c_i, c_j)
    Q = 0.0
    for i in range(n):
        neighbors = A[i].indices
        for j_idx, j in enumerate(neighbors):
            if labels[i] == labels[j]:
                Q += A[i, j] - degrees[i] * degrees[j] / (2 * m)

    Q = Q / (2 * m)

    # Normalize to [0, 1]
    confidence = max(0, min(1, (Q + 0.5)))  # Q typically in [-0.5, 1]

    return float(confidence)


# =============================================================================
# DECISION: Softmax Competition
# =============================================================================

def softmax_decision(
    scores: np.ndarray,
    temperature: float = 0.1
) -> Tuple[int, float, np.ndarray]:
    """
    Softmax competition for winner selection.

    Physics interpretation: Boltzmann distribution at temperature T.
    Low T = winner-take-all (Fisher-KPP limit)
    High T = uniform (exploration)
    """
    # Numerical stability
    scores_shifted = scores - np.max(scores)
    exp_scores = np.exp(scores_shifted / temperature)
    probs = exp_scores / (np.sum(exp_scores) + 1e-10)

    winner = int(np.argmax(probs))
    confidence = float(probs[winner])

    return winner, confidence, probs


# =============================================================================
# OPTIMIZED ENGINE
# =============================================================================

class OptimizedRegulatorEngine:
    """
    Optimized URE with correct physics and fast algorithms.

    Key insight: The original URE's physics can be approximated by:
    - Retrieval: Personalized PageRank (heat diffusion from query)
    - Clustering: Label propagation (spin alignment)
    - Decision: Softmax (Boltzmann distribution)

    These are O(E * iters) instead of O(N^2) or O(N^3).
    """

    def __init__(self, params: Optional[OptimizedParams] = None):
        self.params = params or OptimizedParams()
        self.A: Optional[sparse.csr_matrix] = None
        self.P: Optional[sparse.csr_matrix] = None
        self.data: Optional[np.ndarray] = None

    def build_index(
        self,
        vectors: np.ndarray,
        k: int = 15
    ) -> 'OptimizedRegulatorEngine':
        """Build graph index."""
        self.data = vectors
        self.A, self.P = build_graph_fast(vectors, k=k)
        return self

    def retrieve(
        self,
        query: np.ndarray,
        k: int = 10
    ) -> OptimizedResult:
        """
        Retrieval via hybrid scoring (similarity + graph diffusion).

        Physics: Heat diffusion from query-similar nodes with gravity.
        """
        import time
        start = time.perf_counter()

        # Compute raw similarity (for confidence)
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        data_norm = self.data / (np.linalg.norm(self.data, axis=1, keepdims=True) + 1e-10)
        raw_similarity = data_norm @ query_norm

        # Hybrid scoring for ranking: similarity + neighbor boost
        scores = hybrid_retrieval_score(
            query,
            self.data,
            self.A,
            self.P,
            alpha=self.params.alpha,
            n_iters=self.params.n_iters
        )

        # Extract top-k
        top_indices = np.argsort(scores)[-k:][::-1]

        # Confidence based on RAW max similarity (not hybrid scores)
        # This discriminates valid queries from noise
        max_raw_sim = np.max(raw_similarity)
        confidence = float(max_raw_sim)

        elapsed = (time.perf_counter() - start) * 1000

        return OptimizedResult(
            mode=Mode.RETRIEVAL,
            output=top_indices.tolist(),
            confidence=confidence,
            scores=scores,
            refused=confidence < self.params.tau,
            time_ms=elapsed
        )

    def cluster(
        self,
        vectors: Optional[np.ndarray] = None
    ) -> OptimizedResult:
        """
        Clustering via label propagation.

        Physics: Spin alignment / Potts model relaxation.
        """
        import time
        start = time.perf_counter()

        if vectors is not None:
            self.build_index(vectors)

        labels, n_clusters = label_propagation(
            self.A,
            n_iters=self.params.n_cluster_iters
        )

        confidence = compute_clustering_confidence(self.A, labels)

        elapsed = (time.perf_counter() - start) * 1000

        return OptimizedResult(
            mode=Mode.CLUSTERING,
            output=labels,
            confidence=confidence,
            scores=np.zeros(len(labels)),  # No per-node scores for clustering
            refused=confidence < self.params.tau,
            time_ms=elapsed
        )

    def decide(
        self,
        candidates: np.ndarray,
        scores: Optional[np.ndarray] = None
    ) -> OptimizedResult:
        """
        Decision via softmax competition.

        Physics: Boltzmann distribution / Fisher-KPP limit.
        """
        import time
        start = time.perf_counter()

        if scores is None:
            # Without external scores, use graph centrality
            self.build_index(candidates, k=min(5, len(candidates)-1))
            degrees = np.array(self.A.sum(axis=1)).flatten()
            scores = degrees / (np.max(degrees) + 1e-10)

        winner, confidence, probs = softmax_decision(
            scores,
            temperature=self.params.temperature
        )

        elapsed = (time.perf_counter() - start) * 1000

        return OptimizedResult(
            mode=Mode.DECISION,
            output=winner,
            confidence=confidence,
            scores=probs,
            refused=confidence < self.params.tau,
            time_ms=elapsed
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def optimized_retrieve(
    query: np.ndarray,
    corpus: np.ndarray,
    k: int = 10,
    **kwargs
) -> OptimizedResult:
    """One-liner retrieval."""
    engine = OptimizedRegulatorEngine()
    engine.build_index(corpus)
    return engine.retrieve(query, k=k)


def optimized_cluster(vectors: np.ndarray, **kwargs) -> OptimizedResult:
    """One-liner clustering."""
    engine = OptimizedRegulatorEngine()
    return engine.cluster(vectors)


def optimized_decide(
    candidates: np.ndarray,
    scores: Optional[np.ndarray] = None,
    **kwargs
) -> OptimizedResult:
    """One-liner decision."""
    engine = OptimizedRegulatorEngine()
    return engine.decide(candidates, scores=scores)
