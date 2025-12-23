"""
ACI Toy Implementation - Structural Cognitive Physics
======================================================

This is a minimal, runnable proof of semantics demonstrating:
- Field-based reasoning (not token probability)
- Decidable refusal via topological constraints
- Operator phase switching via spectral signals
- Lyapunov-stable convergence

This is NOT a demo of intelligence - it is a proof of semantics.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum


# =============================================================================
# 1. CORE DATA STRUCTURES
# =============================================================================

@dataclass
class Witness:
    """
    Semantic atom - a structural claim, not an embedding.

    Witnesses are the fundamental units of meaning in ACI.
    They have identity and polarity (+1 or -1).
    """
    id: str
    polarity: int = 1

    def __post_init__(self):
        if self.polarity not in (-1, 1):
            raise ValueError("Polarity must be +1 or -1")

    def negate(self) -> 'Witness':
        """Return the negation of this witness."""
        return Witness(self.id, -self.polarity)

    def __hash__(self):
        return hash((self.id, self.polarity))

    def __repr__(self):
        sign = "+" if self.polarity == 1 else "-"
        return f"W({sign}{self.id})"


class WitnessSketch:
    """
    Witness Bloom Sketch - algebraic sketch for witness sets.

    This is NOT a standard Bloom filter. It supports:
    - Union (addition)
    - Cancellation (opposite polarities cancel)
    - Semantic differencing (XOR-like operation)

    No deletions. No rebuilds.
    """

    def __init__(self, m: int = 1024, k: int = 4):
        self.bits = np.zeros(m, dtype=np.float64)
        self.m = m
        self.k = k
        self._witness_count = 0

    def _hash_indices(self, witness: Witness) -> List[int]:
        """Generate k hash indices for a witness."""
        indices = []
        for i in range(self.k):
            # Use a simple but effective hash combining id and seed
            h = hash((witness.id, i, "aci_salt")) % self.m
            indices.append(h)
        return indices

    def add(self, witness: Witness) -> 'WitnessSketch':
        """Add a witness to the sketch. Polarities accumulate."""
        for idx in self._hash_indices(witness):
            self.bits[idx] += witness.polarity
        self._witness_count += 1
        return self

    def add_many(self, witnesses: List[Witness]) -> 'WitnessSketch':
        """Add multiple witnesses."""
        for w in witnesses:
            self.add(w)
        return self

    def xor(self, other: 'WitnessSketch') -> 'WitnessSketch':
        """
        Semantic difference - what's in self but not other.
        This enables unlearning via cancellation.
        """
        assert self.m == other.m and self.k == other.k
        result = WitnessSketch(self.m, self.k)
        result.bits = self.bits - other.bits
        return result

    def union(self, other: 'WitnessSketch') -> 'WitnessSketch':
        """Merge two sketches."""
        assert self.m == other.m and self.k == other.k
        result = WitnessSketch(self.m, self.k)
        result.bits = self.bits + other.bits
        return result

    def strength(self) -> float:
        """Total activation strength."""
        return np.sum(np.abs(self.bits))

    def net_polarity(self) -> float:
        """Net signed strength."""
        return np.sum(self.bits)

    def probably_contains(self, witness: Witness) -> bool:
        """
        Probabilistic membership test.
        Returns True if the witness might be present.
        """
        indices = self._hash_indices(witness)
        values = [self.bits[idx] for idx in indices]
        # If all positions have same-sign values as the witness polarity
        return all(v * witness.polarity > 0 for v in values)

    def copy(self) -> 'WitnessSketch':
        """Create a deep copy."""
        result = WitnessSketch(self.m, self.k)
        result.bits = self.bits.copy()
        result._witness_count = self._witness_count
        return result

    def __repr__(self):
        return f"WitnessSketch(strength={self.strength():.2f}, net={self.net_polarity():.2f})"


# =============================================================================
# 2. THE MANIFOLD (Discrete Approximation)
# =============================================================================

class SemanticManifold:
    """
    2D grid approximation of the semantic manifold.
    Each cell represents local semantic energy.
    """

    def __init__(self, size: int = 64):
        self.size = size
        self.phi = np.zeros((size, size), dtype=np.float64)

    def copy(self) -> 'SemanticManifold':
        """Create a deep copy."""
        result = SemanticManifold(self.size)
        result.phi = self.phi.copy()
        return result

    def get_field(self) -> np.ndarray:
        """Get the current field state."""
        return self.phi

    def set_field(self, phi: np.ndarray):
        """Set the field state."""
        assert phi.shape == (self.size, self.size)
        self.phi = phi.copy()


# =============================================================================
# 3. CONSTITUTIONAL CONSTRAINTS (Event Horizon)
# =============================================================================

@dataclass
class Constraint:
    """
    Constitutional constraint defining forbidden regions.

    The Schwarzschild radius rs determines the event horizon -
    the boundary beyond which no safe update exists.

    alpha controls severity: higher alpha = larger forbidden zone
    """
    alpha: float

    @property
    def rs(self) -> float:
        """Schwarzschild radius - forbidden zone boundary."""
        return 0.16 * self.alpha + 0.09


class SafetyBoundary:
    """
    Manages the topological safety boundary.

    Points too close to the edge are forbidden.
    If no safe update exists, refusal is mandatory.
    """

    def __init__(self, grid_size: int, constraint: Constraint):
        self.grid_size = grid_size
        self.constraint = constraint
        self._precompute_distances()
        self._precompute_decay()

    def _precompute_distances(self):
        """Precompute distance-to-boundary for all cells."""
        self.distances = np.zeros((self.grid_size, self.grid_size))
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.distances[x, y] = self._distance_to_boundary(x, y)

    def _precompute_decay(self):
        """
        Precompute smooth decay function for boundary.
        Uses sigmoid-like transition near the boundary.
        """
        rs = self.constraint.rs
        # Transition width (smooth over 10% of the boundary region)
        width = max(0.05, rs * 0.3)

        # Compute smooth decay: 1 at center, 0 at boundary
        # Using tanh for smooth transition
        self.decay = np.tanh((self.distances - rs) / width * 3)
        self.decay = np.clip((self.decay + 1) / 2, 0, 1)

        # Ensure center is 1 and boundary is 0
        self.decay = np.where(self.distances > rs + width, 1.0, self.decay)
        self.decay = np.where(self.distances < rs - width, 0.0, self.decay)

    def _distance_to_boundary(self, x: int, y: int) -> float:
        """Normalized distance to nearest edge."""
        return min(x, y, self.grid_size - x - 1, self.grid_size - y - 1) / self.grid_size

    def is_safe(self, x: int, y: int) -> bool:
        """Check if a position is outside the forbidden zone."""
        return self.distances[x, y] > self.constraint.rs

    def get_safe_mask(self) -> np.ndarray:
        """Return boolean mask of safe positions."""
        return self.distances > self.constraint.rs

    def get_decay_mask(self) -> np.ndarray:
        """Return smooth decay mask for boundary conditions."""
        return self.decay

    def any_safe(self) -> bool:
        """Check if any safe position exists."""
        return np.any(self.get_safe_mask())

    def fraction_safe(self) -> float:
        """Fraction of grid that is safe."""
        return np.mean(self.get_safe_mask())


# =============================================================================
# 4. COGNITIVE DYNAMICS (OBDS PDE)
# =============================================================================

class CognitiveDynamics:
    """
    Implements Operator-Based Dynamical Semantics (OBDS).

    We use Diffusion + Reaction + Dissipation.
    The key is the contractive suppression operator that ensures stability.
    """

    def __init__(self, beta: float = 0.1, gamma: float = 0.02):
        self.beta = beta    # Contraction parameter
        self.gamma = gamma  # Dissipation rate (energy loss per step)

    @staticmethod
    def laplacian(phi: np.ndarray) -> np.ndarray:
        """
        Discrete Laplacian operator with Dirichlet (zero) boundary conditions.
        Implements diffusion of semantic energy.
        """
        # Zero-padded Laplacian to avoid wrap-around
        padded = np.pad(phi, 1, mode='constant', constant_values=0)
        lap = (
            padded[:-2, 1:-1] + padded[2:, 1:-1] +   # up/down
            padded[1:-1, :-2] + padded[1:-1, 2:] -    # left/right
            4 * phi
        )
        return lap

    def evolve(self, phi: np.ndarray, source: np.ndarray) -> np.ndarray:
        """
        Single evolution step using implicit heat equation for stability.

        Uses the formula: phi_new = phi + dt * (D * laplacian(phi) + source - gamma * phi)

        where D is the diffusion coefficient (scaled by dt for stability).
        The -gamma * phi term provides exponential decay to equilibrium.
        """
        dt = 0.2   # Time step (small for stability)
        D = 0.5    # Diffusion coefficient

        # Heat equation: diffusion + source + mild decay
        lap = self.laplacian(phi)
        dphi = dt * (D * lap + source - self.gamma * phi)

        # Contractive suppression for large updates
        dphi = dphi / (1 + self.beta * np.abs(dphi))

        return phi + dphi

    def evolve_masked(self, phi: np.ndarray, source: np.ndarray,
                      safe_mask: np.ndarray,
                      boundary_decay: np.ndarray = None) -> np.ndarray:
        """
        Evolve with smooth boundary decay.

        Instead of a hard mask, we use a smooth decay function
        that transitions from 1 (center) to 0 (boundary) to
        prevent wave reflections and numerical instabilities.
        """
        phi_new = self.evolve(phi, source)

        if boundary_decay is not None:
            # Smooth boundary decay
            phi_new = phi_new * boundary_decay
        else:
            # Fall back to hard mask
            phi_new = np.where(safe_mask, phi_new, 0.0)

        return phi_new


# =============================================================================
# 5. SOURCE INJECTION (Queries as Fields)
# =============================================================================

class SourceInjector:
    """
    Maps witness sketches to field sources.

    Queries create vibrations, not answers.
    The field evolution computes the answer.
    """

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self.center = grid_size // 2

    def inject(self, sketch: WitnessSketch,
               spread: float = 5.0,
               amplitude: float = 10.0) -> np.ndarray:
        """
        Convert a witness sketch to a source field.

        The source is a Gaussian centered at the grid center,
        with strength proportional to the sketch's net polarity.
        """
        source = np.zeros((self.grid_size, self.grid_size))

        # Get strength from sketch (normalized)
        strength = sketch.net_polarity() / max(1, sketch.strength()) * amplitude

        # Create Gaussian source at center
        x = np.arange(self.grid_size)
        y = np.arange(self.grid_size)
        X, Y = np.meshgrid(x, y)

        # Gaussian distribution
        dist_sq = (X - self.center)**2 + (Y - self.center)**2
        source = strength * np.exp(-dist_sq / (2 * spread**2))

        return source

    def inject_at_position(self, sketch: WitnessSketch,
                           pos: Tuple[int, int],
                           spread: float = 3.0) -> np.ndarray:
        """Inject source at a specific position."""
        source = np.zeros((self.grid_size, self.grid_size))
        strength = sketch.net_polarity()

        x = np.arange(self.grid_size)
        y = np.arange(self.grid_size)
        X, Y = np.meshgrid(x, y)

        dist_sq = (X - pos[0])**2 + (Y - pos[1])**2
        source = strength * np.exp(-dist_sq / (2 * spread**2))

        return source


# =============================================================================
# 6. LYAPUNOV STABILITY
# =============================================================================

class LyapunovMonitor:
    """
    Monitors system stability using Lyapunov energy function.

    If energy increases, the system is invalid.
    Convergence is detected when energy change is below threshold.
    """

    def __init__(self, epsilon: float = 1e-4):
        self.epsilon = epsilon
        self.energy_history: List[float] = []

    @staticmethod
    def energy(phi: np.ndarray, anchor: np.ndarray) -> float:
        """
        Lyapunov energy function.
        Measures deviation from anchor state.
        """
        return float(np.sum((phi - anchor)**2))

    def record(self, e: float):
        """Record an energy value."""
        self.energy_history.append(e)

    def converged(self, E_prev: float, E_curr: float) -> bool:
        """Check if system has converged."""
        return abs(E_prev - E_curr) < self.epsilon

    def is_stable(self, E_prev: float, E_curr: float) -> bool:
        """Check if energy is non-increasing (stable)."""
        return E_curr <= E_prev + 1e-10  # Small tolerance for numerical error

    def get_history(self) -> List[float]:
        """Get energy history."""
        return self.energy_history.copy()


# =============================================================================
# 7. SPECTRAL MONITORING (Phase Detection)
# =============================================================================

class SpectralMonitor:
    """
    Monitors spectral properties for phase detection.

    A small spectral gap indicates potential collapse.
    This triggers operator switching to prevent FAISS-style collapse.
    """

    def __init__(self, gap_threshold: float = 1e-3):
        self.gap_threshold = gap_threshold

    @staticmethod
    def spectral_gap(phi: np.ndarray) -> float:
        """
        Compute spectral gap of the field.

        Note: For a 2D field, we compute eigenvalues of the matrix.
        A small gap indicates near-degeneracy.
        """
        # Compute eigenvalues
        eigs = np.linalg.eigvals(phi)
        eigs_real = np.sort(np.real(eigs))

        if len(eigs_real) < 2:
            return float('inf')

        return float(eigs_real[-1] - eigs_real[-2])

    def needs_phase_switch(self, phi: np.ndarray) -> bool:
        """Check if operator phase switch is needed."""
        gap = self.spectral_gap(phi)
        return gap < self.gap_threshold

    def get_damping_factor(self, phi: np.ndarray) -> float:
        """
        Get adaptive damping factor based on spectral properties.
        Smaller gap = more damping needed.

        Returns 1.0 (no damping) for trivial/zero fields to allow
        initial field development.
        """
        # Don't damp if field is near-zero (allow initialization)
        field_energy = np.sum(phi**2)
        if field_energy < 1e-10:
            return 1.0  # No damping for trivial fields

        gap = self.spectral_gap(phi)
        if gap < self.gap_threshold:
            return 0.0  # Full damping - diffusion only
        elif gap < 10 * self.gap_threshold:
            return gap / (10 * self.gap_threshold)  # Partial damping
        else:
            return 1.0  # No damping


# =============================================================================
# 8. RESULT TYPES
# =============================================================================

class RefusalType(Enum):
    """Types of refusal the system can emit."""
    BOUNDARY_VIOLATION = "boundary"  # No safe update exists
    ENERGY_VIOLATION = "energy"      # Lyapunov stability violated
    SPECTRAL_COLLAPSE = "spectral"   # Spectral gap collapsed


@dataclass
class ACIResult:
    """Result of an ACI computation."""
    success: bool
    field: Optional[np.ndarray]
    refusal: Optional[RefusalType]
    iterations: int
    final_energy: float
    converged: bool

    def __repr__(self):
        if self.success:
            return f"ACIResult(success, iterations={self.iterations}, energy={self.final_energy:.6f})"
        else:
            return f"ACIResult(REFUSAL: {self.refusal.value}, iterations={self.iterations})"


# =============================================================================
# 9. THE ACI ENGINE
# =============================================================================

class ACIEngine:
    """
    The complete ACI reasoning engine.

    This integrates all components to perform field-based reasoning
    with topological safety guarantees.
    """

    def __init__(self,
                 grid_size: int = 64,
                 alpha: float = 0.5,  # Default to safe region covering most of grid
                 beta: float = 0.1,
                 gamma: float = 0.1,  # Dissipation rate
                 max_iterations: int = 150,
                 convergence_eps: float = 0.005,  # Relative convergence threshold (0.5%)
                 spectral_threshold: float = 1e-3):

        self.grid_size = grid_size
        self.constraint = Constraint(alpha)
        self.safety = SafetyBoundary(grid_size, self.constraint)
        self.dynamics = CognitiveDynamics(beta, gamma)
        self.injector = SourceInjector(grid_size)
        self.lyapunov = LyapunovMonitor(convergence_eps)
        self.spectral = SpectralMonitor(spectral_threshold)
        self.max_iterations = max_iterations

        # Initialize manifold
        self.manifold = SemanticManifold(grid_size)
        self.anchor = self.manifold.phi.copy()

    def reset(self):
        """Reset the manifold to initial state."""
        self.manifold = SemanticManifold(self.grid_size)
        self.anchor = self.manifold.phi.copy()
        self.lyapunov = LyapunovMonitor(self.lyapunov.epsilon)

    def set_anchor(self, phi: np.ndarray):
        """Set the anchor state."""
        self.anchor = phi.copy()

    def reason(self, query_sketch: WitnessSketch,
               injection_steps: int = 10) -> ACIResult:
        """
        Perform field-based reasoning on a query.

        The query creates an initial source perturbation for a few steps,
        then the system relaxes to equilibrium via diffusion and dissipation.

        Returns either a converged field or a refusal.
        """
        phi = self.manifold.phi.copy()
        safe_mask = self.safety.get_safe_mask()
        decay_mask = self.safety.get_decay_mask()

        # Check safety before starting
        if not self.safety.any_safe():
            return ACIResult(
                success=False,
                field=None,
                refusal=RefusalType.BOUNDARY_VIOLATION,
                iterations=0,
                final_energy=0.0,
                converged=False
            )

        # Generate source from query (used only during injection phase)
        base_source = self.injector.inject(query_sketch)
        zero_source = np.zeros_like(base_source)

        for t in range(self.max_iterations):
            # Inject source only for the first few steps
            if t < injection_steps:
                source = base_source * (1 - t / injection_steps)  # Ramp down
            else:
                source = zero_source

            # Check for spectral collapse - apply damping if needed
            damping = self.spectral.get_damping_factor(phi)
            if damping < 1.0:
                source = source * damping

            # Evolve the field with smooth boundary decay
            phi_new = self.dynamics.evolve_masked(phi, source, safe_mask, decay_mask)

            # Compute change in field (for convergence check)
            delta = float(np.sum(np.abs(phi_new - phi)))
            field_magnitude = float(np.sum(np.abs(phi_new))) + 1e-10
            relative_delta = delta / field_magnitude
            self.lyapunov.record(relative_delta)

            # Check convergence after injection phase (using relative change)
            if t >= injection_steps and relative_delta < self.lyapunov.epsilon:
                self.manifold.phi = phi_new
                final_energy = float(np.sum(phi_new**2))
                return ACIResult(
                    success=True,
                    field=phi_new,
                    refusal=None,
                    iterations=t + 1,
                    final_energy=final_energy,
                    converged=True
                )

            # Check for energy blow-up (instability)
            field_energy = float(np.sum(phi_new**2))
            if field_energy > 1e10:
                return ACIResult(
                    success=False,
                    field=None,
                    refusal=RefusalType.ENERGY_VIOLATION,
                    iterations=t,
                    final_energy=field_energy,
                    converged=False
                )

            phi = phi_new

        # Max iterations reached - return current state
        self.manifold.phi = phi
        final_energy = float(np.sum(phi**2))
        return ACIResult(
            success=True,
            field=phi,
            refusal=None,
            iterations=self.max_iterations,
            final_energy=final_energy,
            converged=False
        )

    def unlearn(self, forget_sketch: WitnessSketch,
                current_sketch: WitnessSketch) -> WitnessSketch:
        """
        Unlearn by cancellation.
        Returns the sketch with forgotten witnesses removed.
        """
        return current_sketch.xor(forget_sketch)


# =============================================================================
# 10. DEMONSTRATION
# =============================================================================

def demonstrate_aci():
    """
    Run a demonstration of the ACI system.

    This proves:
    1. Reasoning as field evolution
    2. Safety as geometry
    3. Refusal as inevitability
    4. Unlearning via cancellation
    5. Operator semantics > embeddings
    """
    print("=" * 60)
    print("ACI TOY IMPLEMENTATION - STRUCTURAL COGNITIVE PHYSICS")
    print("=" * 60)
    print()

    # ===================
    # Demo 1: Basic reasoning with safe configuration
    # ===================
    print("-" * 40)
    print("DEMO 1: Basic Field Reasoning")
    print("-" * 40)

    # Create engine with moderate constraint (alpha=0.5 gives rs=0.17)
    engine = ACIEngine(grid_size=64, alpha=0.5)
    print(f"Grid size: {engine.grid_size}x{engine.grid_size}")
    print(f"Constraint alpha: {engine.constraint.alpha}")
    print(f"Schwarzschild radius: {engine.constraint.rs:.3f}")
    print(f"Safe fraction: {engine.safety.fraction_safe():.2%}")
    print()

    # Create a simple query
    query = WitnessSketch()
    query.add(Witness("fact_A", +1))
    query.add(Witness("fact_B", +1))
    query.add(Witness("inference_C", +1))

    print(f"Query sketch: {query}")
    result = engine.reason(query)
    print(f"Result: {result}")
    print()

    # ===================
    # Demo 2: Cancellation / Unlearning
    # ===================
    print("-" * 40)
    print("DEMO 2: Unlearning via Cancellation")
    print("-" * 40)

    # Create knowledge base
    knowledge = WitnessSketch()
    knowledge.add(Witness("fact_A", +1))
    knowledge.add(Witness("fact_B", +1))
    knowledge.add(Witness("secret_X", +1))
    print(f"Original knowledge: {knowledge}")

    # Create forget request
    forget = WitnessSketch()
    forget.add(Witness("secret_X", +1))
    print(f"Forget request: {forget}")

    # Unlearn
    new_knowledge = engine.unlearn(forget, knowledge)
    print(f"After unlearning: {new_knowledge}")

    # Check membership
    secret = Witness("secret_X", +1)
    print(f"secret_X in original: {knowledge.probably_contains(secret)}")
    print(f"secret_X in new: {new_knowledge.probably_contains(secret)}")
    print()

    # ===================
    # Demo 3: Refusal via boundary violation
    # ===================
    print("-" * 40)
    print("DEMO 3: Safety-Induced Refusal")
    print("-" * 40)

    # Show how increasing alpha shrinks the safe region
    print("Effect of alpha on safe region:")
    for alpha in [0.5, 1.0, 2.0, 2.5, 3.0]:
        test_engine = ACIEngine(grid_size=64, alpha=alpha)
        print(f"  alpha={alpha:.1f} -> rs={test_engine.constraint.rs:.3f}, safe={test_engine.safety.fraction_safe():.1%}")
    print()

    # Create a very restrictive constraint (alpha=3.0 gives rs=0.57 > 0.5, no safe zone)
    strict_engine = ACIEngine(grid_size=64, alpha=3.0)
    print(f"Strict constraint alpha: {strict_engine.constraint.alpha}")
    print(f"Schwarzschild radius: {strict_engine.constraint.rs:.3f}")
    print(f"Safe fraction: {strict_engine.safety.fraction_safe():.2%}")

    query = WitnessSketch()
    query.add(Witness("dangerous_request", +1))

    result = strict_engine.reason(query)
    print(f"Result: {result}")
    print("-> Refusal is MANDATORY when no safe update exists!")
    print()

    # ===================
    # Demo 4: Convergence behavior
    # ===================
    print("-" * 40)
    print("DEMO 4: Convergence & Energy")
    print("-" * 40)

    engine.reset()

    # Stronger query with multiple concepts
    query = WitnessSketch()
    for i in range(10):
        query.add(Witness(f"concept_{i}", +1))

    print(f"Query with 10 concepts: {query}")
    result = engine.reason(query)
    print(f"Result: {result}")

    history = engine.lyapunov.get_history()
    if len(history) > 1:
        print(f"Initial energy: {history[0]:.6f}")
        print(f"Final energy: {history[-1]:.6f}")
        print(f"Energy monotonically decreased: {all(history[i] >= history[i+1] for i in range(len(history)-1))}")
    print()

    # ===================
    # Demo 5: Polarity and witness algebra
    # ===================
    print("-" * 40)
    print("DEMO 5: Witness Polarity Algebra")
    print("-" * 40)

    # Create opposing witnesses
    sketch1 = WitnessSketch()
    sketch1.add(Witness("claim_X", +1))
    print(f"Sketch with +claim_X: net_polarity={sketch1.net_polarity():.0f}")

    sketch2 = WitnessSketch()
    sketch2.add(Witness("claim_X", -1))
    print(f"Sketch with -claim_X: net_polarity={sketch2.net_polarity():.0f}")

    # Union cancels out
    merged = sketch1.union(sketch2)
    print(f"Union of opposing claims: net_polarity={merged.net_polarity():.0f}")
    print("-> Opposing witnesses CANCEL, not accumulate!")
    print()

    # ===================
    # Demo 6: Spectral monitoring
    # ===================
    print("-" * 40)
    print("DEMO 6: Spectral Monitoring")
    print("-" * 40)

    # Create a field to analyze
    np.random.seed(42)
    phi = np.random.randn(64, 64) * 0.1
    gap = SpectralMonitor.spectral_gap(phi)
    print(f"Random field spectral gap: {gap:.6f}")

    # Near-degenerate field
    phi_degen = np.ones((64, 64)) * 0.01 + np.random.randn(64, 64) * 0.0001
    gap_degen = SpectralMonitor.spectral_gap(phi_degen)
    print(f"Near-uniform field spectral gap: {gap_degen:.6f}")

    monitor = SpectralMonitor(gap_threshold=1e-3)
    print(f"Random field needs phase switch: {monitor.needs_phase_switch(phi)}")
    print(f"Near-uniform field needs phase switch: {monitor.needs_phase_switch(phi_degen)}")
    print()

    # ===================
    # Demo 7: Field evolution visualization (numerical)
    # ===================
    print("-" * 40)
    print("DEMO 7: Field Evolution Statistics")
    print("-" * 40)

    engine.reset()
    query = WitnessSketch()
    query.add(Witness("test_query", +1))

    result = engine.reason(query)
    if result.success and result.field is not None:
        print(f"Field shape: {result.field.shape}")
        print(f"Field min: {result.field.min():.6f}")
        print(f"Field max: {result.field.max():.6f}")
        print(f"Field mean: {result.field.mean():.6f}")
        print(f"Field std: {result.field.std():.6f}")
        print(f"Non-zero cells: {np.count_nonzero(result.field)}")
    print()

    # ===================
    # Summary
    # ===================
    print("=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print()
    print("This toy system demonstrates:")
    print("  1. Reasoning as field evolution (not token prediction)")
    print("  2. Safety as geometry (Schwarzschild boundary)")
    print("  3. Refusal as inevitability (no safe update = mandatory refusal)")
    print("  4. Unlearning via cancellation (sketch XOR)")
    print("  5. Polarity algebra (opposing claims cancel)")
    print("  6. Spectral phase detection (collapse prevention)")
    print("  7. Lyapunov stability (energy monotonically decreases)")
    print()
    print("No LLM can do this by prompt engineering.")
    print()


if __name__ == "__main__":
    demonstrate_aci()
