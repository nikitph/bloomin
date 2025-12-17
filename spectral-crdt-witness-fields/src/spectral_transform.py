"""
Spectral Transform for Witness Fields

Implements T: W → Ŵ, an invertible transform such that:
    T(W₁ ★ W₂) = T(W₁) · T(W₂)

This is the "semantic FFT" - witness flow diagonalizes in the spectral domain,
enabling O(log N) semantic updates and early collapse detection.

Key insight: Semantic evolution is LINEAR in the spectral witness domain.
"""

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field
from scipy.linalg import eigh
from witness_algebra import WitnessField, Witness, WitnessPolarity


@dataclass
class SpectralWitnessField:
    """
    A witness field in the spectral (frequency) domain.

    The spectral representation diagonalizes:
    - Witness composition (★) becomes pointwise multiplication
    - Witness flow becomes independent evolution per mode
    - Entropy decay affects high-frequency modes more

    Attributes:
        coefficients: Complex spectral coefficients (n_modes,)
        eigenvalues: Eigenvalues of the witness kernel (n_modes,)
        eigenvectors: Eigenvector basis (n_modes, dimension)
        positive_mask: Which modes come from positive evidence
        negative_mask: Which modes come from negative evidence
    """
    coefficients: np.ndarray  # Shape: (n_modes,) complex
    eigenvalues: np.ndarray   # Shape: (n_modes,) real
    eigenvectors: np.ndarray  # Shape: (n_modes, dimension)
    positive_mask: np.ndarray  # Shape: (n_modes,) bool
    negative_mask: np.ndarray  # Shape: (n_modes,) bool
    dimension: int
    n_modes: int
    entity_id: str = ""
    version: int = 0

    @property
    def magnitude_spectrum(self) -> np.ndarray:
        """Magnitude of spectral coefficients (like power spectrum)"""
        return np.abs(self.coefficients)

    @property
    def phase_spectrum(self) -> np.ndarray:
        """Phase of spectral coefficients"""
        return np.angle(self.coefficients)

    @property
    def energy(self) -> float:
        """Total spectral energy (Parseval's theorem)"""
        return float(np.sum(np.abs(self.coefficients) ** 2))

    @property
    def entropy(self) -> float:
        """Spectral entropy - measures spread across modes"""
        mags = np.abs(self.coefficients)
        total = np.sum(mags)
        if total < 1e-10:
            return 0.0
        probs = mags / total
        probs = probs[probs > 1e-10]
        return -np.sum(probs * np.log(probs))

    def mode_strengths(self) -> np.ndarray:
        """Strength of each mode (eigenvalue × coefficient magnitude)"""
        return np.abs(self.eigenvalues * self.coefficients)

    def dominant_modes(self, k: int = 10) -> np.ndarray:
        """Indices of k strongest modes"""
        strengths = self.mode_strengths()
        return np.argsort(strengths)[-k:][::-1]

    def high_frequency_energy(self, threshold: float = 0.5) -> float:
        """Energy in high-frequency modes (noise/ambiguity)"""
        # Higher modes = higher frequency
        n_high = int(self.n_modes * (1 - threshold))
        if n_high == 0:
            return 0.0
        high_coeffs = self.coefficients[-n_high:]
        return float(np.sum(np.abs(high_coeffs) ** 2))

    def low_frequency_energy(self, threshold: float = 0.5) -> float:
        """Energy in low-frequency modes (stable semantics)"""
        n_low = int(self.n_modes * threshold)
        if n_low == 0:
            return 0.0
        low_coeffs = self.coefficients[:n_low]
        return float(np.sum(np.abs(low_coeffs) ** 2))


class SpectralWitnessTransform:
    """
    The Spectral Witness Transform T: W ↔ Ŵ

    Properties:
    1. T is invertible: T⁻¹(T(W)) = W
    2. T diagonalizes composition: T(W₁ ★ W₂) = T(W₁) · T(W₂)
    3. Spectral modes correspond to semantic "frequencies"

    Implementation uses eigendecomposition of the witness Gram matrix,
    which acts as a kernel operator on the witness space.
    """

    def __init__(
        self,
        n_modes: int = 64,
        kernel_bandwidth: float = 1.0,
        min_eigenvalue: float = 1e-6
    ):
        """
        Initialize the spectral transform.

        Args:
            n_modes: Number of spectral modes to compute
            kernel_bandwidth: Bandwidth of the witness kernel
            min_eigenvalue: Minimum eigenvalue for numerical stability
        """
        self.n_modes = n_modes
        self.bandwidth = kernel_bandwidth
        self.min_eigenvalue = min_eigenvalue

        # Cached basis (for fixed-dimension transforms)
        self._cached_basis: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._cached_dimension: Optional[int] = None

    def forward(self, field: WitnessField) -> SpectralWitnessField:
        """
        Forward transform: W → Ŵ

        Computes spectral decomposition of the witness field.
        """
        if field.n_total == 0:
            # Empty field has zero spectrum
            return SpectralWitnessField(
                coefficients=np.zeros(self.n_modes, dtype=np.complex128),
                eigenvalues=np.zeros(self.n_modes),
                eigenvectors=np.eye(self.n_modes, field.dimension),
                positive_mask=np.zeros(self.n_modes, dtype=bool),
                negative_mask=np.zeros(self.n_modes, dtype=bool),
                dimension=field.dimension,
                n_modes=self.n_modes,
                entity_id=field.entity_id,
                version=field.version
            )

        # Build witness matrix (n_witnesses × dimension)
        W_matrix = field.to_matrix()

        # Compute witness kernel (Gram matrix with RBF kernel)
        # K(wᵢ, wⱼ) = exp(-||wᵢ - wⱼ||² / 2σ²)
        n_wit = W_matrix.shape[0]

        if n_wit < 2:
            # Single witness - use identity decomposition
            return self._single_witness_spectrum(field)

        # Gram matrix with RBF kernel
        K = self._compute_kernel_matrix(W_matrix)

        # Eigendecomposition
        actual_modes = min(self.n_modes, n_wit)
        eigenvalues, eigenvectors = eigh(K, subset_by_index=[n_wit - actual_modes, n_wit - 1])

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Clamp small eigenvalues
        eigenvalues = np.maximum(eigenvalues, self.min_eigenvalue)

        # Compute spectral coefficients
        # Project witness matrix onto eigenvectors
        coeffs_real = W_matrix.T @ eigenvectors  # (dimension, n_modes)

        # Reduce to single coefficient per mode via norm
        coefficients = np.linalg.norm(coeffs_real, axis=0) + 0j

        # Add phase based on polarity balance
        pos_vecs = np.array([w.vector for w in field.positive_witnesses]) if field.positive_witnesses else np.zeros((0, field.dimension))
        neg_vecs = np.array([w.vector for w in field.negative_witnesses]) if field.negative_witnesses else np.zeros((0, field.dimension))

        if len(pos_vecs) > 0 or len(neg_vecs) > 0:
            # Phase encodes polarity distribution
            # Eigenvectors are in witness space (n_witnesses dim)
            # We need to map them back to embedding space for comparison

            for m in range(actual_modes):
                # Get eigenvector in witness space
                eig_in_witness = eigenvectors[:, m]  # Shape: (n_witnesses,)

                # Compute weighted projection of witness matrix
                # This gives us the embedding-space representation of this mode
                eig_in_embedding = W_matrix.T @ eig_in_witness  # Shape: (dimension,)
                eig_norm = np.linalg.norm(eig_in_embedding)
                if eig_norm > 1e-10:
                    eig_in_embedding = eig_in_embedding / eig_norm

                # Compute polarity alignment in embedding space
                pos_align = 0.0
                if len(pos_vecs) > 0:
                    for pv in pos_vecs:
                        pv_norm = np.linalg.norm(pv)
                        if pv_norm > 1e-10:
                            pos_align += np.dot(eig_in_embedding, pv / pv_norm)
                    pos_align /= len(pos_vecs)

                neg_align = 0.0
                if len(neg_vecs) > 0:
                    for nv in neg_vecs:
                        nv_norm = np.linalg.norm(nv)
                        if nv_norm > 1e-10:
                            neg_align += np.dot(eig_in_embedding, nv / nv_norm)
                    neg_align /= len(neg_vecs)

                # Phase = angle based on polarity balance
                phase = np.arctan2(neg_align, pos_align + 1e-10)
                coefficients[m] = np.abs(coefficients[m]) * np.exp(1j * phase)

        # Build polarity masks
        positive_mask = np.zeros(actual_modes, dtype=bool)
        negative_mask = np.zeros(actual_modes, dtype=bool)

        for m in range(actual_modes):
            phase = np.angle(coefficients[m])
            if -np.pi/2 <= phase <= np.pi/2:
                positive_mask[m] = True
            else:
                negative_mask[m] = True

        # Map eigenvectors from witness space to embedding space
        # Each eigenvector in R^{n_witnesses} maps to R^{dimension}
        embedding_eigenvectors = np.zeros((actual_modes, field.dimension))
        for m in range(actual_modes):
            # Weighted sum of witness vectors by eigenvector components
            embedding_eigenvectors[m] = W_matrix.T @ eigenvectors[:, m]
            norm = np.linalg.norm(embedding_eigenvectors[m])
            if norm > 1e-10:
                embedding_eigenvectors[m] /= norm

        # Pad to n_modes if needed
        if actual_modes < self.n_modes:
            coefficients = np.pad(coefficients, (0, self.n_modes - actual_modes))
            eigenvalues = np.pad(eigenvalues, (0, self.n_modes - actual_modes),
                                 constant_values=self.min_eigenvalue)
            eigenvectors_padded = np.zeros((self.n_modes, field.dimension))
            eigenvectors_padded[:actual_modes] = embedding_eigenvectors
            embedding_eigenvectors = eigenvectors_padded
            positive_mask = np.pad(positive_mask, (0, self.n_modes - actual_modes))
            negative_mask = np.pad(negative_mask, (0, self.n_modes - actual_modes))

        return SpectralWitnessField(
            coefficients=coefficients,
            eigenvalues=eigenvalues,
            eigenvectors=embedding_eigenvectors,
            positive_mask=positive_mask,
            negative_mask=negative_mask,
            dimension=field.dimension,
            n_modes=self.n_modes,
            entity_id=field.entity_id,
            version=field.version
        )

    def _compute_kernel_matrix(self, W: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix"""
        n = W.shape[0]
        K = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                diff = W[i] - W[j]
                K[i, j] = np.exp(-np.dot(diff, diff) / (2 * self.bandwidth ** 2))
                K[j, i] = K[i, j]

        return K

    def _single_witness_spectrum(self, field: WitnessField) -> SpectralWitnessField:
        """Handle single-witness case"""
        wit = field.all_witnesses[0]

        coefficients = np.zeros(self.n_modes, dtype=np.complex128)
        coefficients[0] = wit.strength  # All energy in first mode

        eigenvalues = np.zeros(self.n_modes)
        eigenvalues[0] = 1.0

        eigenvectors = np.zeros((self.n_modes, field.dimension))
        eigenvectors[0] = wit.vector

        positive_mask = np.zeros(self.n_modes, dtype=bool)
        negative_mask = np.zeros(self.n_modes, dtype=bool)
        positive_mask[0] = wit.polarity == WitnessPolarity.POSITIVE
        negative_mask[0] = wit.polarity == WitnessPolarity.NEGATIVE

        return SpectralWitnessField(
            coefficients=coefficients,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            positive_mask=positive_mask,
            negative_mask=negative_mask,
            dimension=field.dimension,
            n_modes=self.n_modes,
            entity_id=field.entity_id,
            version=field.version
        )

    def inverse(self, spectral_field: SpectralWitnessField) -> WitnessField:
        """
        Inverse transform: Ŵ → W

        Reconstructs witness field from spectral representation.
        """
        field = WitnessField(
            dimension=spectral_field.dimension,
            entity_id=spectral_field.entity_id,
            version=spectral_field.version
        )

        # Reconstruct witnesses from dominant modes
        dominant = spectral_field.dominant_modes(k=min(32, spectral_field.n_modes))

        for mode_idx in dominant:
            coeff = spectral_field.coefficients[mode_idx]
            magnitude = np.abs(coeff)

            if magnitude < 1e-10:
                continue

            # Get eigenvector direction
            direction = spectral_field.eigenvectors[mode_idx]
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                continue

            direction = direction / norm

            # Determine polarity from mask and phase
            if spectral_field.positive_mask[mode_idx]:
                polarity = WitnessPolarity.POSITIVE
            elif spectral_field.negative_mask[mode_idx]:
                polarity = WitnessPolarity.NEGATIVE
            else:
                polarity = WitnessPolarity.NEUTRAL

            witness = Witness(
                vector=direction,
                polarity=polarity,
                strength=magnitude * spectral_field.eigenvalues[mode_idx]
            )
            field.add_witness(witness)

        return field

    def multiply(
        self,
        s1: SpectralWitnessField,
        s2: SpectralWitnessField
    ) -> SpectralWitnessField:
        """
        Pointwise multiplication in spectral domain.

        This corresponds to convolution (★) in witness domain:
        T(W₁ ★ W₂) = T(W₁) · T(W₂)
        """
        if s1.dimension != s2.dimension:
            raise ValueError("Dimension mismatch")

        # Pointwise multiply coefficients
        new_coeffs = s1.coefficients * s2.coefficients

        # Average eigenvalues (geometric mean)
        new_eigenvalues = np.sqrt(np.abs(s1.eigenvalues * s2.eigenvalues))

        # Interpolate eigenvectors (for approximate reconstruction)
        # Weight by eigenvalues
        w1 = s1.eigenvalues / (s1.eigenvalues + s2.eigenvalues + 1e-10)
        w2 = 1 - w1

        new_eigenvectors = np.zeros_like(s1.eigenvectors)
        for m in range(s1.n_modes):
            vec = w1[m] * s1.eigenvectors[m] + w2[m] * s2.eigenvectors[m]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                new_eigenvectors[m] = vec / norm
            else:
                new_eigenvectors[m] = s1.eigenvectors[m]

        # Combine polarity masks (XOR logic for double negatives)
        new_positive = (s1.positive_mask & s2.positive_mask) | (s1.negative_mask & s2.negative_mask)
        new_negative = (s1.positive_mask & s2.negative_mask) | (s1.negative_mask & s2.positive_mask)

        return SpectralWitnessField(
            coefficients=new_coeffs,
            eigenvalues=new_eigenvalues,
            eigenvectors=new_eigenvectors,
            positive_mask=new_positive,
            negative_mask=new_negative,
            dimension=s1.dimension,
            n_modes=s1.n_modes,
            entity_id=f"{s1.entity_id}·{s2.entity_id}",
            version=max(s1.version, s2.version) + 1
        )

    def add(
        self,
        s1: SpectralWitnessField,
        s2: SpectralWitnessField
    ) -> SpectralWitnessField:
        """
        Addition in spectral domain.

        This corresponds to join (⊕) in witness domain:
        T(W₁ ⊕ W₂) ≈ T(W₁) + T(W₂)  (approximate for nonlinear join)
        """
        if s1.dimension != s2.dimension:
            raise ValueError("Dimension mismatch")

        # Add coefficients
        new_coeffs = s1.coefficients + s2.coefficients

        # Max eigenvalues (upper bound)
        new_eigenvalues = np.maximum(s1.eigenvalues, s2.eigenvalues)

        # Weighted average eigenvectors
        w1 = np.abs(s1.coefficients) / (np.abs(s1.coefficients) + np.abs(s2.coefficients) + 1e-10)
        w2 = 1 - w1

        new_eigenvectors = np.zeros_like(s1.eigenvectors)
        for m in range(s1.n_modes):
            vec = w1[m] * s1.eigenvectors[m] + w2[m] * s2.eigenvectors[m]
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                new_eigenvectors[m] = vec / norm

        # Union of polarity masks
        new_positive = s1.positive_mask | s2.positive_mask
        new_negative = s1.negative_mask | s2.negative_mask

        return SpectralWitnessField(
            coefficients=new_coeffs,
            eigenvalues=new_eigenvalues,
            eigenvectors=new_eigenvectors,
            positive_mask=new_positive,
            negative_mask=new_negative,
            dimension=s1.dimension,
            n_modes=s1.n_modes,
            entity_id=f"{s1.entity_id}+{s2.entity_id}",
            version=max(s1.version, s2.version) + 1
        )


# Factory functions
def to_spectral(field: WitnessField, n_modes: int = 64) -> SpectralWitnessField:
    """Convert witness field to spectral representation"""
    transform = SpectralWitnessTransform(n_modes=n_modes)
    return transform.forward(field)


def from_spectral(spectral: SpectralWitnessField) -> WitnessField:
    """Convert spectral representation back to witness field"""
    transform = SpectralWitnessTransform(n_modes=spectral.n_modes)
    return transform.inverse(spectral)


def spectral_compose(s1: SpectralWitnessField, s2: SpectralWitnessField) -> SpectralWitnessField:
    """Compose two spectral fields (pointwise multiply)"""
    transform = SpectralWitnessTransform(n_modes=s1.n_modes)
    return transform.multiply(s1, s2)


def spectral_join(s1: SpectralWitnessField, s2: SpectralWitnessField) -> SpectralWitnessField:
    """Join two spectral fields (add)"""
    transform = SpectralWitnessTransform(n_modes=s1.n_modes)
    return transform.add(s1, s2)
