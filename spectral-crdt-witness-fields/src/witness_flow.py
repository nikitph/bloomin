"""
Witness Flow Equations - Thermodynamic Evolution of Semantic Fields

Implements the witness flow equation:
    dŴ(k)/dt = λₖŴ(k) - μₖH(Ŵ(k))

Where:
- λₖ: Growth/decay rate for mode k (eigenvalue-dependent)
- μₖ: Entropy damping coefficient
- H(Ŵ(k)): Local entropy of mode k

Key properties:
1. Each spectral mode evolves INDEPENDENTLY
2. High-entropy modes decay (noise dissipates)
3. Low-entropy modes persist (stable semantics)
4. Fixed points correspond to Bellman optimality

This is the thermodynamic core of SCWF:
"Dynamic programming is not algorithmic, but thermodynamic:
 it is the zero-flux equilibrium of witness flow."
"""

import numpy as np
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy.integrate import solve_ivp

from spectral_transform import SpectralWitnessField, SpectralWitnessTransform
from witness_algebra import WitnessField


class FlowType(Enum):
    """Types of witness flow dynamics"""
    GRADIENT = "gradient"  # Gradient descent on semantic energy
    RICCI = "ricci"  # Ricci flow (curvature-driven)
    HEAT = "heat"  # Heat equation (entropy maximizing)
    BELLMAN = "bellman"  # Bellman update (optimal control)
    DAMPED = "damped"  # Damped oscillation (stable convergence)


@dataclass
class FlowParameters:
    """Parameters for witness flow evolution"""
    # Growth/decay rates per mode
    lambda_base: float = 0.1  # Base growth rate
    lambda_eigenvalue_scale: float = 1.0  # How much eigenvalue affects growth

    # Entropy damping
    mu_base: float = 0.01  # Base entropy damping
    mu_frequency_scale: float = 2.0  # Higher modes damped more

    # Stability
    min_coefficient: float = 1e-10  # Minimum coefficient magnitude
    max_coefficient: float = 100.0  # Maximum coefficient magnitude

    # Convergence
    convergence_threshold: float = 1e-6  # Fixed point detection
    max_iterations: int = 1000


class WitnessFlowEquation:
    """
    The witness flow equation system.

    Evolves spectral witness fields according to:
        dŴ(k)/dt = λₖŴ(k) - μₖH(Ŵ(k))

    The flow has several important fixed points:
    1. Zero field (trivial)
    2. Delta distributions (certain knowledge)
    3. Bellman optimal (maximum expected utility)
    """

    def __init__(
        self,
        params: Optional[FlowParameters] = None,
        flow_type: FlowType = FlowType.DAMPED
    ):
        self.params = params or FlowParameters()
        self.flow_type = flow_type

    def compute_growth_rates(
        self,
        spectral: SpectralWitnessField
    ) -> np.ndarray:
        """
        Compute λₖ for each mode.

        Growth rate depends on:
        - Eigenvalue (stronger modes grow faster)
        - Mode index (higher modes decay)
        """
        n = spectral.n_modes
        base = self.params.lambda_base

        # Eigenvalue contribution
        eigenvalue_contrib = self.params.lambda_eigenvalue_scale * np.log(
            spectral.eigenvalues + 1e-10
        )

        # Frequency decay (higher modes decay)
        frequency_decay = -np.linspace(0, 0.1, n)

        return base + eigenvalue_contrib + frequency_decay

    def compute_damping_rates(
        self,
        spectral: SpectralWitnessField
    ) -> np.ndarray:
        """
        Compute μₖ for each mode.

        Damping rate depends on:
        - Base damping
        - Mode frequency (higher modes damped more)
        """
        n = spectral.n_modes
        base = self.params.mu_base

        # Higher modes get more damping
        frequency_factor = np.linspace(1, self.params.mu_frequency_scale, n)

        return base * frequency_factor

    def compute_mode_entropy(
        self,
        coefficients: np.ndarray
    ) -> np.ndarray:
        """
        Compute local entropy H(Ŵ(k)) for each mode.

        Uses magnitude distribution as proxy for entropy.
        """
        magnitudes = np.abs(coefficients)
        total = np.sum(magnitudes) + 1e-10

        # Normalized "probability" per mode
        probs = magnitudes / total

        # Local entropy contribution
        # H_k = -p_k * log(p_k)
        with np.errstate(divide='ignore', invalid='ignore'):
            entropy = -probs * np.log(probs + 1e-10)
            entropy = np.nan_to_num(entropy, nan=0.0)

        return entropy

    def flow_derivative(
        self,
        t: float,
        coefficients_flat: np.ndarray,
        spectral_template: SpectralWitnessField
    ) -> np.ndarray:
        """
        Compute dŴ/dt for the flow equation.

        Args:
            t: Time (for time-dependent flows)
            coefficients_flat: Real-valued flattened coefficients
            spectral_template: Template for spectral field structure

        Returns:
            Time derivative of coefficients
        """
        n = spectral_template.n_modes

        # Reconstruct complex coefficients
        coefficients = coefficients_flat[:n] + 1j * coefficients_flat[n:]

        # Get flow parameters
        lambdas = self.compute_growth_rates(spectral_template)
        mus = self.compute_damping_rates(spectral_template)
        entropies = self.compute_mode_entropy(coefficients)

        # Different flow types
        if self.flow_type == FlowType.GRADIENT:
            # Gradient flow: dŴ/dt = -∇E(Ŵ)
            # Simple approximation: flow toward lower entropy
            d_coeff = -mus * entropies * coefficients

        elif self.flow_type == FlowType.RICCI:
            # Ricci flow: dŴ/dt = -R(Ŵ)
            # Curvature-driven: large coefficients shrink, small grow
            curvature = np.abs(coefficients) - np.mean(np.abs(coefficients))
            d_coeff = -lambdas * curvature * np.sign(coefficients)

        elif self.flow_type == FlowType.HEAT:
            # Heat equation: dŴ/dt = ΔŴ
            # Laplacian approximation via second differences
            laplacian = np.zeros_like(coefficients)
            laplacian[1:-1] = (
                coefficients[:-2] - 2 * coefficients[1:-1] + coefficients[2:]
            )
            d_coeff = lambdas[0] * laplacian

        elif self.flow_type == FlowType.BELLMAN:
            # Bellman update: dŴ/dt = T(Ŵ) - Ŵ where T is Bellman operator
            # Approximation: bias toward dominant mode
            dominant = np.argmax(np.abs(coefficients))
            target = np.zeros_like(coefficients)
            target[dominant] = np.sum(np.abs(coefficients))
            d_coeff = lambdas * (target - coefficients)

        elif self.flow_type == FlowType.DAMPED:
            # Damped flow: dŴ/dt = λŴ - μH(Ŵ)
            # The core SCWF flow equation
            growth = lambdas * coefficients
            damping = mus * entropies * coefficients
            d_coeff = growth - damping

        else:
            d_coeff = np.zeros_like(coefficients)

        # Flatten to real
        return np.concatenate([d_coeff.real, d_coeff.imag])

    def evolve(
        self,
        spectral: SpectralWitnessField,
        time_span: Tuple[float, float] = (0, 10),
        n_steps: int = 100
    ) -> List[SpectralWitnessField]:
        """
        Evolve spectral witness field through time.

        Returns list of snapshots at each time step.
        """
        # Flatten coefficients to real for ODE solver
        coefficients_flat = np.concatenate([
            spectral.coefficients.real,
            spectral.coefficients.imag
        ])

        # Time points
        t_eval = np.linspace(time_span[0], time_span[1], n_steps)

        # Solve ODE
        solution = solve_ivp(
            lambda t, y: self.flow_derivative(t, y, spectral),
            time_span,
            coefficients_flat,
            t_eval=t_eval,
            method='RK45'
        )

        # Extract snapshots
        snapshots = []
        for i in range(len(solution.t)):
            y = solution.y[:, i]
            n = spectral.n_modes
            new_coeffs = y[:n] + 1j * y[n:]

            # Clamp coefficients
            magnitudes = np.abs(new_coeffs)
            magnitudes = np.clip(
                magnitudes,
                self.params.min_coefficient,
                self.params.max_coefficient
            )
            phases = np.angle(new_coeffs)
            new_coeffs = magnitudes * np.exp(1j * phases)

            snapshot = SpectralWitnessField(
                coefficients=new_coeffs,
                eigenvalues=spectral.eigenvalues,
                eigenvectors=spectral.eigenvectors,
                positive_mask=spectral.positive_mask,
                negative_mask=spectral.negative_mask,
                dimension=spectral.dimension,
                n_modes=spectral.n_modes,
                entity_id=f"{spectral.entity_id}@t={solution.t[i]:.2f}",
                version=spectral.version
            )
            snapshots.append(snapshot)

        return snapshots

    def find_fixed_point(
        self,
        spectral: SpectralWitnessField,
        max_iterations: Optional[int] = None
    ) -> Tuple[SpectralWitnessField, int, bool]:
        """
        Find fixed point of witness flow.

        Returns:
            (fixed_point, iterations, converged)
        """
        if max_iterations is None:
            max_iterations = self.params.max_iterations

        current = spectral
        dt = 0.1

        for iteration in range(max_iterations):
            # Take one step
            snapshots = self.evolve(current, time_span=(0, dt), n_steps=2)
            next_state = snapshots[-1]

            # Check convergence
            diff = np.abs(next_state.coefficients - current.coefficients)
            max_diff = np.max(diff)

            if max_diff < self.params.convergence_threshold:
                return next_state, iteration, True

            current = next_state

        return current, max_iterations, False


class WitnessFlowSimulator:
    """
    Simulator for witness flow dynamics with visualization support.

    Tracks:
    - Spectral energy over time
    - Entropy evolution
    - Mode dynamics
    - Fixed point convergence
    """

    def __init__(
        self,
        flow_equation: Optional[WitnessFlowEquation] = None
    ):
        self.flow = flow_equation or WitnessFlowEquation()
        self.history: List[SpectralWitnessField] = []
        self.metrics_history: List[dict] = []

    def simulate(
        self,
        initial: SpectralWitnessField,
        time_span: Tuple[float, float] = (0, 100),
        n_steps: int = 1000,
        track_metrics: bool = True
    ) -> SpectralWitnessField:
        """
        Run full simulation from initial state.

        Returns final state.
        """
        self.history = self.flow.evolve(initial, time_span, n_steps)

        if track_metrics:
            self.metrics_history = []
            for state in self.history:
                metrics = {
                    'energy': state.energy,
                    'entropy': state.entropy,
                    'high_freq_energy': state.high_frequency_energy(),
                    'low_freq_energy': state.low_frequency_energy(),
                    'dominant_mode': int(state.dominant_modes(1)[0]),
                    'n_active_modes': int(np.sum(np.abs(state.coefficients) > 0.01))
                }
                self.metrics_history.append(metrics)

        return self.history[-1]

    def detect_phase_transition(
        self,
        threshold: float = 0.5
    ) -> Optional[int]:
        """
        Detect phase transition in flow.

        A phase transition occurs when:
        - Entropy drops suddenly
        - Energy concentrates in few modes
        """
        if len(self.metrics_history) < 2:
            return None

        entropies = [m['entropy'] for m in self.metrics_history]

        for i in range(1, len(entropies)):
            # Large entropy drop = phase transition
            if entropies[i-1] > 0 and (entropies[i-1] - entropies[i]) / entropies[i-1] > threshold:
                return i

        return None

    def get_attractor_type(self) -> str:
        """
        Classify the attractor of the flow.

        Returns one of:
        - "fixed_point": Converged to stable state
        - "limit_cycle": Oscillating
        - "chaotic": No clear pattern
        - "collapse": All modes decayed
        """
        if len(self.history) < 10:
            return "unknown"

        final_states = self.history[-10:]
        energies = [s.energy for s in final_states]

        # Check for collapse
        if np.mean(energies) < 1e-6:
            return "collapse"

        # Check for fixed point (constant energy)
        if np.std(energies) / (np.mean(energies) + 1e-10) < 0.01:
            return "fixed_point"

        # Check for limit cycle (periodic energy)
        fft = np.fft.fft(energies)
        if np.max(np.abs(fft[1:])) > 0.5 * np.abs(fft[0]):
            return "limit_cycle"

        return "chaotic"


class BellmanWitnessFlow:
    """
    Bellman-optimal witness flow.

    This implements Theorem 1 from the SCWF theory:
    "Dynamic programming is thermodynamic equilibrium of witness flow."

    The fixed point W* satisfies:
        W* = F(W*) where F is the Bellman operator

    This corresponds to optimal substructure in DP.
    """

    def __init__(
        self,
        discount_factor: float = 0.99,
        value_threshold: float = 0.01
    ):
        self.gamma = discount_factor
        self.threshold = value_threshold

    def bellman_operator(
        self,
        spectral: SpectralWitnessField,
        reward_field: Optional[SpectralWitnessField] = None
    ) -> SpectralWitnessField:
        """
        Apply Bellman operator to spectral field.

        T(V)(s) = max_a [R(s,a) + γ Σ P(s'|s,a)V(s')]

        In spectral domain, this becomes pointwise operations:
        T(Ŵ)(k) = max(R̂(k), γ * Ŵ(k))
        """
        if reward_field is None:
            # Default: uniform reward
            reward_coeffs = np.ones(spectral.n_modes) * 0.1
        else:
            reward_coeffs = np.abs(reward_field.coefficients)

        # Bellman update in spectral domain
        current_value = np.abs(spectral.coefficients)
        new_value = np.maximum(reward_coeffs, self.gamma * current_value)

        # Preserve phases
        phases = np.angle(spectral.coefficients)
        new_coefficients = new_value * np.exp(1j * phases)

        return SpectralWitnessField(
            coefficients=new_coefficients,
            eigenvalues=spectral.eigenvalues,
            eigenvectors=spectral.eigenvectors,
            positive_mask=spectral.positive_mask,
            negative_mask=spectral.negative_mask,
            dimension=spectral.dimension,
            n_modes=spectral.n_modes,
            entity_id=f"T({spectral.entity_id})",
            version=spectral.version + 1
        )

    def solve_value_iteration(
        self,
        initial: SpectralWitnessField,
        reward_field: Optional[SpectralWitnessField] = None,
        max_iterations: int = 1000
    ) -> Tuple[SpectralWitnessField, int]:
        """
        Find Bellman fixed point via value iteration.

        Returns:
            (optimal_field, iterations)
        """
        current = initial

        for iteration in range(max_iterations):
            next_state = self.bellman_operator(current, reward_field)

            # Check convergence
            diff = np.max(np.abs(next_state.coefficients - current.coefficients))
            if diff < self.threshold:
                return next_state, iteration

            current = next_state

        return current, max_iterations


# Convenience functions
def evolve_field(
    field: WitnessField,
    time: float = 10.0,
    flow_type: FlowType = FlowType.DAMPED
) -> WitnessField:
    """Evolve a witness field through time and return final state"""
    from spectral_transform import to_spectral, from_spectral

    spectral = to_spectral(field)
    flow = WitnessFlowEquation(flow_type=flow_type)
    snapshots = flow.evolve(spectral, time_span=(0, time))

    return from_spectral(snapshots[-1])


def find_semantic_equilibrium(
    field: WitnessField
) -> Tuple[WitnessField, bool]:
    """Find thermodynamic equilibrium of witness field"""
    from spectral_transform import to_spectral, from_spectral

    spectral = to_spectral(field)
    flow = WitnessFlowEquation(flow_type=FlowType.DAMPED)
    fixed_point, _, converged = flow.find_fixed_point(spectral)

    return from_spectral(fixed_point), converged
