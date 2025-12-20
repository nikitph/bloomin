"""
Unified Regulator Engine (URE)
==============================

A thermodynamically regulated computation engine where inference is relaxation
into stable attractors.

Core Principle:
    Computation is controlled energy flow on a manifold, regulated by dissipation.

Architecture:
    Exploration → Selection → Commitment

This engine unifies retrieval, clustering, and decision-making as different
configurations of the same dynamical system.

Task Mapping:
    | Task       | Exploration     | Selection      | Commitment      |
    |------------|-----------------|----------------|-----------------|
    | Retrieval  | Wave            | Telegrapher    | Poisson basins  |
    | Clustering | Fokker-Planck   | Cahn-Hilliard  | Phase domains   |
    | Decision   | Schrödinger     | Fisher-KPP     | Winner          |

Author: URE Implementation
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, eigsh
from scipy.ndimage import label
from typing import Literal, Optional, Tuple, List, Dict, Union, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import warnings


# =============================================================================
# TYPE DEFINITIONS
# =============================================================================

class Mode(Enum):
    """Operating modes for the regulator engine."""
    RETRIEVAL = "retrieval"
    CLUSTERING = "clustering"
    DECISION = "decision"


@dataclass
class RegulatorParams:
    """Parameters for the regulator engine."""
    # Time evolution
    T_explore: int = 50          # Exploration timesteps
    T_select: int = 50           # Selection timesteps
    dt: float = 0.01             # Time step

    # Exploration parameters
    D: float = 0.1               # Diffusion coefficient (Fokker-Planck)

    # Selection parameters
    gamma: float = 0.1           # Damping (Telegrapher)
    epsilon: float = 0.01        # Interface width (Cahn-Hilliard)

    # Confidence/refusal
    tau: float = 0.3             # Refusal threshold

    # Numerical stability
    normalize_every: int = 10    # Normalize field every N steps
    min_field_value: float = 1e-10


@dataclass
class RegulatorState:
    """Unified state for the regulator engine."""
    # Graph structure
    n_nodes: int
    L: sparse.csr_matrix         # Laplacian matrix
    adjacency: sparse.csr_matrix # Adjacency matrix

    # Core fields
    psi: np.ndarray              # Exploratory field (wave/probability/amplitude)
    u: np.ndarray                # Stabilized field (energy/phase/activation)
    u_t: np.ndarray              # Time derivative of u (for Telegrapher)
    rho: np.ndarray              # Committed mass (sources)
    phi: np.ndarray              # Potential/decision landscape

    # Optional task-specific fields
    v_drift: Optional[np.ndarray] = None    # Drift field (Fokker-Planck)
    V_loss: Optional[np.ndarray] = None     # Energy/loss landscape (Schrödinger)

    # Metadata
    node_data: Optional[np.ndarray] = None  # Original embeddings at nodes
    node_labels: Optional[np.ndarray] = None


@dataclass
class RegulatorResult:
    """Result from the regulator engine."""
    mode: Mode
    output: Union[List[int], np.ndarray, int]  # basins/clusters/decision
    confidence: float
    confidence_per_item: np.ndarray
    refused: bool
    metadata: Dict = field(default_factory=dict)


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_knn_graph(
    vectors: np.ndarray,
    k: int = 10,
    metric: str = "euclidean",
    include_self: bool = False
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    Build k-nearest neighbor graph from vectors.

    Args:
        vectors: [N, D] array of embeddings
        k: Number of neighbors
        metric: Distance metric
        include_self: Whether to include self-loops

    Returns:
        adjacency: Sparse adjacency matrix (weighted by similarity)
        laplacian: Graph Laplacian
    """
    from sklearn.neighbors import NearestNeighbors

    n = vectors.shape[0]
    k_actual = min(k + (0 if include_self else 1), n)

    nn = NearestNeighbors(n_neighbors=k_actual, metric=metric)
    nn.fit(vectors)
    distances, indices = nn.kneighbors(vectors)

    # Build sparse adjacency matrix with Gaussian kernel weights
    rows, cols, data = [], [], []
    sigma = np.median(distances[:, 1:])  # Median distance for kernel width

    for i in range(n):
        for j_idx in range(k_actual):
            j = indices[i, j_idx]
            if not include_self and i == j:
                continue
            d = distances[i, j_idx]
            w = np.exp(-d**2 / (2 * sigma**2))
            rows.append(i)
            cols.append(j)
            data.append(w)

    adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    # Symmetrize
    adjacency = (adjacency + adjacency.T) / 2

    # Build Laplacian: L = D - A
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    D = sparse.diags(degrees)
    laplacian = D - adjacency

    return adjacency, laplacian


def build_grid_graph(
    height: int,
    width: int,
    connectivity: int = 4
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    Build a regular grid graph.

    Args:
        height: Grid height
        width: Grid width
        connectivity: 4 or 8 connectivity

    Returns:
        adjacency, laplacian
    """
    n = height * width
    rows, cols, data = [], [], []

    def idx(i, j):
        return i * width + j

    for i in range(height):
        for j in range(width):
            current = idx(i, j)

            # 4-connectivity
            neighbors = []
            if i > 0: neighbors.append(idx(i-1, j))
            if i < height-1: neighbors.append(idx(i+1, j))
            if j > 0: neighbors.append(idx(i, j-1))
            if j < width-1: neighbors.append(idx(i, j+1))

            # 8-connectivity (diagonals)
            if connectivity == 8:
                if i > 0 and j > 0: neighbors.append(idx(i-1, j-1))
                if i > 0 and j < width-1: neighbors.append(idx(i-1, j+1))
                if i < height-1 and j > 0: neighbors.append(idx(i+1, j-1))
                if i < height-1 and j < width-1: neighbors.append(idx(i+1, j+1))

            for neighbor in neighbors:
                rows.append(current)
                cols.append(neighbor)
                data.append(1.0)

    adjacency = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    degrees = np.array(adjacency.sum(axis=1)).flatten()
    D = sparse.diags(degrees)
    laplacian = D - adjacency

    return adjacency, laplacian


def build_epsilon_graph(
    vectors: np.ndarray,
    epsilon: float,
    metric: str = "euclidean"
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """
    Build epsilon-neighborhood graph.

    Args:
        vectors: [N, D] array
        epsilon: Neighborhood radius
        metric: Distance metric

    Returns:
        adjacency, laplacian
    """
    from sklearn.neighbors import radius_neighbors_graph

    adjacency = radius_neighbors_graph(
        vectors,
        radius=epsilon,
        mode='connectivity',
        metric=metric
    )
    adjacency = (adjacency + adjacency.T).astype(float)
    adjacency.data = np.ones_like(adjacency.data)

    degrees = np.array(adjacency.sum(axis=1)).flatten()
    D = sparse.diags(degrees)
    laplacian = D - adjacency

    return adjacency, laplacian


# =============================================================================
# OPERATOR LIBRARY
# =============================================================================

def op_wave(psi: np.ndarray, L: sparse.csr_matrix) -> np.ndarray:
    """
    Wave operator: ∂ψ/∂t = L·ψ

    Diffuses the exploratory field across the graph.
    Used in retrieval exploration phase.
    """
    return L @ psi


def op_telegrapher(
    u: np.ndarray,
    u_t: np.ndarray,
    L: sparse.csr_matrix,
    gamma: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Telegrapher operator: ∂²u/∂t² + γ·∂u/∂t = L·u

    Damped wave equation for selection phase in retrieval.
    Returns (du, du_t) for symplectic integration.

    Args:
        u: Current field
        u_t: Time derivative of field
        L: Laplacian
        gamma: Damping coefficient

    Returns:
        du: Update to u
        du_tt: Update to u_t (acceleration)
    """
    du = u_t
    du_tt = L @ u - gamma * u_t
    return du, du_tt


def op_fokker_planck(
    p: np.ndarray,
    v_drift: np.ndarray,
    L: sparse.csr_matrix,
    A: sparse.csr_matrix,
    D: float
) -> np.ndarray:
    """
    Fokker-Planck operator: ∂p/∂t = -∇·(v·p) + D·∇²p

    Advection-diffusion for clustering exploration.

    Args:
        p: Probability density
        v_drift: Drift velocity at each node
        L: Laplacian
        A: Adjacency matrix
        D: Diffusion coefficient

    Returns:
        dp: Update to probability density
    """
    # Diffusion term: D * L @ p
    diffusion = D * (L @ p)

    # Advection term: approximate divergence of (v * p) on graph
    # ∇·(vp) ≈ sum over neighbors of (v_j * p_j - v_i * p_i) * w_ij / degree_i
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees = np.maximum(degrees, 1e-10)

    # Compute flux: (v * p) at each node
    flux = v_drift * p

    # Divergence approximation
    advection = A @ flux - degrees * flux
    advection = advection / degrees

    return -advection + diffusion


def op_schrodinger_imag(
    psi: np.ndarray,
    V_loss: np.ndarray,
    L: sparse.csr_matrix
) -> np.ndarray:
    """
    Imaginary-time Schrödinger operator: ∂ψ/∂t = L·ψ - V·ψ

    Exponential decay in high-potential regions.
    Used for decision exploration phase.

    Args:
        psi: Wave amplitude
        V_loss: Potential/loss landscape
        L: Laplacian

    Returns:
        dpsi: Update to wave amplitude
    """
    return L @ psi - V_loss * psi


def op_cahn_hilliard(
    u: np.ndarray,
    L: sparse.csr_matrix,
    epsilon: float
) -> np.ndarray:
    """
    Cahn-Hilliard operator: ∂u/∂t = ∇²μ, μ = u³ - u - ε²∇²u

    Phase separation dynamics for clustering selection.
    Drives field toward ±1 phases.

    Args:
        u: Order parameter field
        L: Laplacian
        epsilon: Interface width parameter

    Returns:
        du: Update to order parameter
    """
    # Chemical potential: μ = u³ - u - ε²·L·u
    mu = u**3 - u - epsilon**2 * (L @ u)

    # Evolution: ∂u/∂t = L·μ (conserved dynamics)
    return L @ mu


def op_fisher_kpp(
    u: np.ndarray,
    L: sparse.csr_matrix
) -> np.ndarray:
    """
    Fisher-KPP operator: ∂u/∂t = L·u + u·(1-u)

    Reaction-diffusion for decision selection.
    Drives toward winner-take-all (u=0 or u=1).

    Args:
        u: Activation field
        L: Laplacian

    Returns:
        du: Update to activation
    """
    return L @ u + u * (1 - u)


def op_poisson(
    rho: np.ndarray,
    L: sparse.csr_matrix
) -> np.ndarray:
    """
    Poisson solver: L·φ = -ρ

    Computes potential landscape from mass distribution.
    Used for basin extraction in retrieval commitment.

    Args:
        rho: Source/mass distribution
        L: Laplacian

    Returns:
        phi: Potential field
    """
    # Regularize Laplacian (add small diagonal for invertibility)
    n = L.shape[0]
    L_reg = L + sparse.eye(n) * 1e-6

    # Solve L·φ = -ρ
    phi = spsolve(L_reg, -rho)

    return phi


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_field(
    field: np.ndarray,
    min_value: float = 1e-10,
    mode: str = "l1"
) -> np.ndarray:
    """Normalize field to sum to 1 (probability) or unit norm."""
    field = np.maximum(field, min_value)

    if mode == "l1":
        return field / (np.sum(field) + 1e-10)
    elif mode == "l2":
        return field / (np.linalg.norm(field) + 1e-10)
    elif mode == "softmax":
        field = field - np.max(field)  # Numerical stability
        exp_field = np.exp(field)
        return exp_field / (np.sum(exp_field) + 1e-10)
    else:
        return field


def map_to_energy(psi: np.ndarray) -> np.ndarray:
    """Map exploratory field to energy (|ψ|²)."""
    return np.abs(psi)**2


def extract_mass(u: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """Extract committed mass from stabilized field."""
    return np.maximum(u - threshold, 0)


def extract_basins(
    phi: np.ndarray,
    L: sparse.csr_matrix,
    A: sparse.csr_matrix
) -> Tuple[np.ndarray, int]:
    """
    Extract basins of attraction from potential landscape.

    Uses gradient descent to find local minima, then assigns
    each node to its basin.

    Args:
        phi: Potential field
        L: Laplacian
        A: Adjacency matrix

    Returns:
        basin_labels: Basin assignment for each node
        n_basins: Number of basins found
    """
    n = len(phi)
    basin_labels = np.zeros(n, dtype=int)

    # Find local minima by checking if lower than all neighbors
    A_dense = A.toarray() if sparse.issparse(A) else A

    local_minima = []
    for i in range(n):
        neighbors = np.where(A_dense[i] > 0)[0]
        if len(neighbors) == 0:
            # Isolated node is its own minimum
            local_minima.append(i)
        elif phi[i] <= np.min(phi[neighbors]):
            local_minima.append(i)

    # Assign basin labels to minima
    for idx, minimum in enumerate(local_minima):
        basin_labels[minimum] = idx + 1

    # Gradient descent from each node to find its basin
    for i in range(n):
        if basin_labels[i] > 0:
            continue

        current = i
        visited = set()

        while basin_labels[current] == 0:
            if current in visited:
                # Cycle detected, break
                break
            visited.add(current)

            neighbors = np.where(A_dense[current] > 0)[0]
            if len(neighbors) == 0:
                break

            # Move to lowest neighbor
            next_node = neighbors[np.argmin(phi[neighbors])]
            if phi[next_node] >= phi[current]:
                # At a local minimum that wasn't detected
                basin_labels[current] = len(local_minima) + 1
                local_minima.append(current)
                break
            current = next_node

        # Assign all visited nodes to the basin
        if basin_labels[current] > 0:
            for v in visited:
                basin_labels[v] = basin_labels[current]

    return basin_labels, len(local_minima)


def extract_phase_domains(
    u: np.ndarray,
    A: sparse.csr_matrix,
    threshold: float = 0.0
) -> Tuple[np.ndarray, int]:
    """
    Extract connected components based on phase (sign of u).

    Args:
        u: Order parameter field
        A: Adjacency matrix
        threshold: Threshold for phase assignment

    Returns:
        cluster_labels: Cluster assignment for each node
        n_clusters: Number of clusters found
    """
    # Create binary phase assignment
    phase = (u > threshold).astype(int)

    # Find connected components within each phase
    A_dense = A.toarray() if sparse.issparse(A) else A
    n = len(u)

    cluster_labels = np.zeros(n, dtype=int)
    current_label = 0
    visited = set()

    for start in range(n):
        if start in visited:
            continue

        current_label += 1
        stack = [start]
        start_phase = phase[start]

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            cluster_labels[node] = current_label

            # Add neighbors with same phase
            neighbors = np.where(A_dense[node] > 0)[0]
            for neighbor in neighbors:
                if neighbor not in visited and phase[neighbor] == start_phase:
                    stack.append(neighbor)

    return cluster_labels, current_label


# =============================================================================
# CONFIDENCE COMPUTATION
# =============================================================================

def compute_confidence_basins(
    rho: np.ndarray,
    basin_labels: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute confidence for basin-based retrieval.

    confidence_i = mass_i / total_mass

    Args:
        rho: Mass distribution
        basin_labels: Basin assignments

    Returns:
        max_confidence: Confidence of top basin
        confidence_per_basin: Confidence for each basin
    """
    unique_basins = np.unique(basin_labels[basin_labels > 0])
    if len(unique_basins) == 0:
        return 0.0, np.array([])

    total_mass = np.sum(rho) + 1e-10
    confidence_per_basin = np.zeros(len(unique_basins))

    for i, basin in enumerate(unique_basins):
        mask = basin_labels == basin
        basin_mass = np.sum(rho[mask])
        confidence_per_basin[i] = basin_mass / total_mass

    return np.max(confidence_per_basin), confidence_per_basin


def compute_confidence_clusters(
    u: np.ndarray,
    cluster_labels: np.ndarray
) -> Tuple[float, np.ndarray]:
    """
    Compute confidence for clustering.

    confidence_i = |phase_mass_i| / total_mass

    Args:
        u: Order parameter field
        cluster_labels: Cluster assignments

    Returns:
        max_confidence: Confidence of top cluster
        confidence_per_cluster: Confidence for each cluster
    """
    unique_clusters = np.unique(cluster_labels[cluster_labels > 0])
    if len(unique_clusters) == 0:
        return 0.0, np.array([])

    # Use absolute value of u as mass
    mass = np.abs(u)
    total_mass = np.sum(mass) + 1e-10
    confidence_per_cluster = np.zeros(len(unique_clusters))

    for i, cluster in enumerate(unique_clusters):
        mask = cluster_labels == cluster
        cluster_mass = np.sum(mass[mask])
        confidence_per_cluster[i] = cluster_mass / total_mass

    return np.max(confidence_per_cluster), confidence_per_cluster


def compute_confidence_decision(u: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Compute confidence for decision-making.

    confidence = u_winner / sum(u)

    Args:
        u: Activation field

    Returns:
        confidence: Confidence in winner
        confidence_per_node: Normalized activations
    """
    u_positive = np.maximum(u, 0)
    total = np.sum(u_positive) + 1e-10
    confidence_per_node = u_positive / total

    return np.max(confidence_per_node), confidence_per_node


# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_state(
    n_nodes: int,
    adjacency: sparse.csr_matrix,
    laplacian: sparse.csr_matrix,
    init_condition: Union[np.ndarray, str] = "uniform",
    node_data: Optional[np.ndarray] = None,
    v_drift: Optional[np.ndarray] = None,
    V_loss: Optional[np.ndarray] = None
) -> RegulatorState:
    """
    Initialize regulator state.

    Args:
        n_nodes: Number of nodes in graph
        adjacency: Adjacency matrix
        laplacian: Laplacian matrix
        init_condition: Initial field ("uniform", "random", or array)
        node_data: Original embeddings (optional)
        v_drift: Drift field for Fokker-Planck (optional)
        V_loss: Loss landscape for Schrödinger (optional)

    Returns:
        Initialized RegulatorState
    """
    # Initialize exploratory field
    if isinstance(init_condition, str):
        if init_condition == "uniform":
            psi = np.ones(n_nodes) / n_nodes
        elif init_condition == "random":
            psi = np.random.rand(n_nodes)
            psi = psi / np.sum(psi)
        else:
            raise ValueError(f"Unknown init_condition: {init_condition}")
    else:
        psi = init_condition.copy()
        psi = psi / (np.sum(psi) + 1e-10)

    return RegulatorState(
        n_nodes=n_nodes,
        L=laplacian,
        adjacency=adjacency,
        psi=psi,
        u=np.zeros(n_nodes),
        u_t=np.zeros(n_nodes),
        rho=np.zeros(n_nodes),
        phi=np.zeros(n_nodes),
        v_drift=v_drift,
        V_loss=V_loss,
        node_data=node_data
    )


def initialize_from_query(
    query: np.ndarray,
    node_data: np.ndarray,
    adjacency: sparse.csr_matrix,
    laplacian: sparse.csr_matrix,
    metric: str = "cosine"
) -> RegulatorState:
    """
    Initialize state from a query vector.

    Sets initial field based on similarity to query.

    Args:
        query: Query embedding
        node_data: Node embeddings
        adjacency: Adjacency matrix
        laplacian: Laplacian matrix
        metric: Similarity metric

    Returns:
        Initialized RegulatorState
    """
    n_nodes = node_data.shape[0]

    # Compute similarity to query
    if metric == "cosine":
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        node_norms = node_data / (np.linalg.norm(node_data, axis=1, keepdims=True) + 1e-10)
        similarity = node_norms @ query_norm
    elif metric == "euclidean":
        distances = np.linalg.norm(node_data - query, axis=1)
        similarity = 1 / (1 + distances)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Convert to probability-like initial condition
    similarity = np.maximum(similarity, 0)
    psi = similarity / (np.sum(similarity) + 1e-10)

    return initialize_state(
        n_nodes=n_nodes,
        adjacency=adjacency,
        laplacian=laplacian,
        init_condition=psi,
        node_data=node_data
    )


# =============================================================================
# MASTER REGULATOR LOOP
# =============================================================================

def regulator(
    mode: Mode,
    state: RegulatorState,
    params: RegulatorParams = None
) -> RegulatorResult:
    """
    The Master Regulator Loop.

    This is the unification: one loop with interchangeable operators,
    shared state, and shared confidence semantics.

    Phase 1: Exploration
        - Retrieval:  Wave equation (diffusion on graph)
        - Clustering: Fokker-Planck (drift + diffusion)
        - Decision:   Schrödinger (decay in high-loss regions)

    Phase 2: Selection (Dissipation)
        - Retrieval:  Telegrapher (damped wave)
        - Clustering: Cahn-Hilliard (phase separation)
        - Decision:   Fisher-KPP (reaction-diffusion)

    Phase 3: Commitment
        - Retrieval:  Poisson basins
        - Clustering: Phase domains
        - Decision:   Winner selection

    Args:
        mode: Operating mode (retrieval, clustering, decision)
        state: Initialized regulator state
        params: Regulator parameters

    Returns:
        RegulatorResult with output, confidence, and metadata
    """
    if params is None:
        params = RegulatorParams()

    psi = state.psi.copy()
    L = state.L
    A = state.adjacency
    dt = params.dt

    # =========================================================================
    # PHASE 1: EXPLORATION
    # =========================================================================

    for t in range(params.T_explore):
        if mode == Mode.RETRIEVAL:
            # Wave equation
            dpsi = op_wave(psi, L)
            psi = psi + dt * dpsi

        elif mode == Mode.CLUSTERING:
            # Fokker-Planck
            if state.v_drift is None:
                # Default: no drift (pure diffusion)
                v_drift = np.zeros(state.n_nodes)
            else:
                v_drift = state.v_drift
            dpsi = op_fokker_planck(psi, v_drift, L, A, params.D)
            psi = psi + dt * dpsi

        elif mode == Mode.DECISION:
            # Imaginary-time Schrödinger
            if state.V_loss is None:
                # Default: uniform potential
                V_loss = np.zeros(state.n_nodes)
            else:
                V_loss = state.V_loss
            dpsi = op_schrodinger_imag(psi, V_loss, L)
            psi = psi + dt * dpsi

        # Normalize periodically for stability
        if (t + 1) % params.normalize_every == 0:
            psi = normalize_field(psi, params.min_field_value, mode="l1")

    # Final normalization
    psi = normalize_field(psi, params.min_field_value, mode="l1")

    # =========================================================================
    # PHASE 2: SELECTION (Dissipation)
    # =========================================================================

    # Map exploratory field to energy
    u = map_to_energy(psi)
    u_t = np.zeros(state.n_nodes)  # Initial velocity

    for t in range(params.T_select):
        if mode == Mode.RETRIEVAL:
            # Telegrapher equation
            du, du_t = op_telegrapher(u, u_t, L, params.gamma)
            u = u + dt * du
            u_t = u_t + dt * du_t

        elif mode == Mode.CLUSTERING:
            # Cahn-Hilliard
            du = op_cahn_hilliard(u, L, params.epsilon)
            u = u + dt * du
            # Clamp to reasonable range
            u = np.clip(u, -2, 2)

        elif mode == Mode.DECISION:
            # Fisher-KPP
            du = op_fisher_kpp(u, L)
            u = u + dt * du
            # Clamp to [0, 1] for stability
            u = np.clip(u, 0, 1)

    # =========================================================================
    # PHASE 3: COMMITMENT
    # =========================================================================

    rho = extract_mass(u)

    if mode == Mode.RETRIEVAL:
        # Solve Poisson for potential landscape
        phi = op_poisson(rho, L)

        # Extract basins from potential
        basin_labels, n_basins = extract_basins(phi, L, A)

        # Compute confidence
        max_conf, conf_per_basin = compute_confidence_basins(rho, basin_labels)

        # Check refusal
        refused = max_conf < params.tau

        # Get top basin nodes
        if n_basins > 0:
            top_basin = np.argmax(conf_per_basin) + 1
            output = list(np.where(basin_labels == top_basin)[0])
        else:
            output = []

        return RegulatorResult(
            mode=mode,
            output=output,
            confidence=max_conf,
            confidence_per_item=conf_per_basin,
            refused=refused,
            metadata={
                "n_basins": n_basins,
                "basin_labels": basin_labels,
                "phi": phi,
                "rho": rho
            }
        )

    elif mode == Mode.CLUSTERING:
        # Extract phase domains
        cluster_labels, n_clusters = extract_phase_domains(u, A)

        # Compute confidence
        max_conf, conf_per_cluster = compute_confidence_clusters(u, cluster_labels)

        # Check refusal
        refused = max_conf < params.tau

        return RegulatorResult(
            mode=mode,
            output=cluster_labels,
            confidence=max_conf,
            confidence_per_item=conf_per_cluster,
            refused=refused,
            metadata={
                "n_clusters": n_clusters,
                "u": u
            }
        )

    elif mode == Mode.DECISION:
        # Winner selection
        decision = int(np.argmax(u))

        # Compute confidence
        max_conf, conf_per_node = compute_confidence_decision(u)

        # Check refusal
        refused = max_conf < params.tau

        return RegulatorResult(
            mode=mode,
            output=decision,
            confidence=max_conf,
            confidence_per_item=conf_per_node,
            refused=refused,
            metadata={
                "u": u,
                "top_k": np.argsort(u)[-10:][::-1].tolist()
            }
        )

    raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# HIGH-LEVEL API
# =============================================================================

class UnifiedRegulatorEngine:
    """
    High-level API for the Unified Regulator Engine.

    This class provides a clean interface for:
    - Building graphs from data
    - Running retrieval, clustering, and decision tasks
    - Cascading multiple modes
    """

    def __init__(
        self,
        params: Optional[RegulatorParams] = None
    ):
        """
        Initialize the engine.

        Args:
            params: Default parameters (can be overridden per-call)
        """
        self.params = params or RegulatorParams()
        self.state: Optional[RegulatorState] = None
        self.adjacency: Optional[sparse.csr_matrix] = None
        self.laplacian: Optional[sparse.csr_matrix] = None
        self.node_data: Optional[np.ndarray] = None

    def build_index(
        self,
        vectors: np.ndarray,
        k: int = 10,
        graph_type: str = "knn"
    ) -> 'UnifiedRegulatorEngine':
        """
        Build the graph index from vectors.

        Args:
            vectors: [N, D] embedding vectors
            k: Number of neighbors for kNN graph
            graph_type: "knn" or "epsilon"

        Returns:
            self (for chaining)
        """
        self.node_data = vectors

        if graph_type == "knn":
            self.adjacency, self.laplacian = build_knn_graph(vectors, k=k)
        elif graph_type == "epsilon":
            # Auto-compute epsilon from median distance
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=2)
            nn.fit(vectors)
            distances, _ = nn.kneighbors(vectors)
            epsilon = np.median(distances[:, 1]) * 2
            self.adjacency, self.laplacian = build_epsilon_graph(vectors, epsilon)
        else:
            raise ValueError(f"Unknown graph_type: {graph_type}")

        return self

    def retrieve(
        self,
        query: np.ndarray,
        k: int = 10,
        params: Optional[RegulatorParams] = None
    ) -> RegulatorResult:
        """
        Retrieval mode: find relevant items for a query.

        Args:
            query: Query embedding
            k: Number of results to return
            params: Override parameters

        Returns:
            RegulatorResult with retrieved indices
        """
        if self.node_data is None:
            raise ValueError("Must call build_index first")

        # Initialize from query
        state = initialize_from_query(
            query, self.node_data,
            self.adjacency, self.laplacian
        )

        # Run regulator
        result = regulator(
            Mode.RETRIEVAL,
            state,
            params or self.params
        )

        # Truncate to k results
        if not result.refused and len(result.output) > k:
            # Sort by mass within basin
            rho = result.metadata["rho"]
            basin_nodes = result.output
            sorted_nodes = sorted(basin_nodes, key=lambda i: -rho[i])
            result.output = sorted_nodes[:k]

        return result

    def cluster(
        self,
        vectors: Optional[np.ndarray] = None,
        params: Optional[RegulatorParams] = None
    ) -> RegulatorResult:
        """
        Clustering mode: find natural clusters in data.

        Args:
            vectors: Optional new vectors (uses indexed if None)
            params: Override parameters

        Returns:
            RegulatorResult with cluster labels
        """
        if vectors is not None:
            self.build_index(vectors)

        if self.node_data is None:
            raise ValueError("Must provide vectors or call build_index first")

        # Initialize uniform
        state = initialize_state(
            n_nodes=self.node_data.shape[0],
            adjacency=self.adjacency,
            laplacian=self.laplacian,
            init_condition="uniform",
            node_data=self.node_data
        )

        return regulator(Mode.CLUSTERING, state, params or self.params)

    def decide(
        self,
        candidates: np.ndarray,
        V_loss: Optional[np.ndarray] = None,
        params: Optional[RegulatorParams] = None
    ) -> RegulatorResult:
        """
        Decision mode: select best candidate.

        Args:
            candidates: [N, D] candidate embeddings
            V_loss: Optional loss/cost for each candidate
            params: Override parameters

        Returns:
            RegulatorResult with winning index
        """
        # Build graph over candidates
        self.build_index(candidates, k=min(10, len(candidates)-1))

        # Initialize uniform
        state = initialize_state(
            n_nodes=candidates.shape[0],
            adjacency=self.adjacency,
            laplacian=self.laplacian,
            init_condition="uniform",
            node_data=candidates,
            V_loss=V_loss
        )

        return regulator(Mode.DECISION, state, params or self.params)

    def cascade(
        self,
        query: np.ndarray,
        mode_sequence: List[Mode],
        params: Optional[RegulatorParams] = None
    ) -> List[RegulatorResult]:
        """
        Run cascaded modes (e.g., coarse retrieval → fine decision).

        Args:
            query: Initial query
            mode_sequence: Sequence of modes to run
            params: Override parameters

        Returns:
            List of results from each stage
        """
        results = []
        current_data = self.node_data

        for mode in mode_sequence:
            if mode == Mode.RETRIEVAL:
                state = initialize_from_query(
                    query, current_data,
                    self.adjacency, self.laplacian
                )
                result = regulator(mode, state, params or self.params)

                if not result.refused:
                    # Narrow down to retrieved nodes
                    current_data = current_data[result.output]
                    self.build_index(current_data)

            elif mode == Mode.CLUSTERING:
                state = initialize_state(
                    n_nodes=current_data.shape[0],
                    adjacency=self.adjacency,
                    laplacian=self.laplacian,
                    init_condition="uniform",
                    node_data=current_data
                )
                result = regulator(mode, state, params or self.params)

            elif mode == Mode.DECISION:
                state = initialize_state(
                    n_nodes=current_data.shape[0],
                    adjacency=self.adjacency,
                    laplacian=self.laplacian,
                    init_condition="uniform",
                    node_data=current_data
                )
                result = regulator(mode, state, params or self.params)

            results.append(result)

            if result.refused:
                break

        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_retrieve(
    query: np.ndarray,
    corpus: np.ndarray,
    k: int = 10,
    **kwargs
) -> RegulatorResult:
    """One-liner retrieval."""
    engine = UnifiedRegulatorEngine()
    engine.build_index(corpus)
    return engine.retrieve(query, k=k)


def quick_cluster(
    vectors: np.ndarray,
    **kwargs
) -> RegulatorResult:
    """One-liner clustering."""
    engine = UnifiedRegulatorEngine()
    return engine.cluster(vectors)


def quick_decide(
    candidates: np.ndarray,
    costs: Optional[np.ndarray] = None,
    **kwargs
) -> RegulatorResult:
    """One-liner decision."""
    engine = UnifiedRegulatorEngine()
    return engine.decide(candidates, V_loss=costs)
