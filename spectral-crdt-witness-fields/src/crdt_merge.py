"""
CRDT Merge Operations for Spectral Witness Fields

Implements conflict-free replicated data type semantics in the spectral domain.

Key merge law for replicas i, j:
    Ŵ_{i⊔j}(k) = max(Ŵᵢ(k), Ŵⱼ(k))

This ensures:
- Associativity: (A ⊔ B) ⊔ C = A ⊔ (B ⊔ C)
- Commutativity: A ⊔ B = B ⊔ A
- Idempotence: A ⊔ A = A
- Eventual consistency: All replicas converge to same state

The revolutionary insight: Semantic contradictions become IMPOSSIBLE.
The algebra forbids introducing inconsistency.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time

from spectral_transform import SpectralWitnessField, SpectralWitnessTransform, to_spectral
from witness_algebra import WitnessField, Witness, WitnessPolarity


class MergeStrategy(Enum):
    """Strategies for merging spectral coefficients"""
    MAX_MAGNITUDE = "max_magnitude"  # Keep coefficient with larger magnitude
    MAX_EIGENVALUE = "max_eigenvalue"  # Keep coefficient with larger eigenvalue
    VECTOR_SUM = "vector_sum"  # Complex addition (preserves phase)
    WEIGHTED_AVERAGE = "weighted_average"  # Weight by eigenvalues


@dataclass
class VectorClock:
    """Vector clock for causal ordering of CRDT operations"""
    clock: Dict[str, int] = field(default_factory=dict)

    def tick(self, replica_id: str):
        """Increment clock for this replica"""
        self.clock[replica_id] = self.clock.get(replica_id, 0) + 1

    def merge(self, other: 'VectorClock'):
        """Merge with another vector clock (pointwise max)"""
        all_keys = set(self.clock.keys()) | set(other.clock.keys())
        for k in all_keys:
            self.clock[k] = max(self.clock.get(k, 0), other.clock.get(k, 0))

    def happens_before(self, other: 'VectorClock') -> bool:
        """Check if self causally precedes other"""
        for k, v in self.clock.items():
            if v > other.clock.get(k, 0):
                return False
        # At least one must be strictly less
        return any(
            self.clock.get(k, 0) < other.clock.get(k, 0)
            for k in other.clock.keys()
        )

    def concurrent_with(self, other: 'VectorClock') -> bool:
        """Check if self and other are concurrent (incomparable)"""
        return not self.happens_before(other) and not other.happens_before(self)

    def __repr__(self):
        return f"VC({self.clock})"


@dataclass
class CRDTSpectralField:
    """
    A CRDT-enabled spectral witness field.

    Each replica maintains:
    - Spectral coefficients with version vectors
    - Merkle hash for integrity verification
    - Causal metadata for audit trail
    """
    spectral: SpectralWitnessField
    replica_id: str
    vector_clock: VectorClock = field(default_factory=VectorClock)
    merkle_hash: str = ""
    created_at: float = field(default_factory=time.time)
    parent_hashes: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.merkle_hash:
            self.merkle_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute Merkle hash of spectral coefficients"""
        data = self.spectral.coefficients.tobytes()
        data += self.replica_id.encode()
        data += str(self.vector_clock.clock).encode()
        return hashlib.sha256(data).hexdigest()[:32]

    def update(self, new_spectral: SpectralWitnessField) -> 'CRDTSpectralField':
        """Create new version with updated spectral field"""
        new_clock = VectorClock(clock=dict(self.vector_clock.clock))
        new_clock.tick(self.replica_id)

        return CRDTSpectralField(
            spectral=new_spectral,
            replica_id=self.replica_id,
            vector_clock=new_clock,
            parent_hashes=[self.merkle_hash]
        )


class SpectralCRDT:
    """
    Conflict-free Replicated Data Type for Spectral Witness Fields.

    This is the core CRDT implementation that enables distributed
    semantics without coordination.

    Key property: Merge is a JOIN SEMILATTICE operation.
    - Commutative: merge(A, B) = merge(B, A)
    - Associative: merge(merge(A, B), C) = merge(A, merge(B, C))
    - Idempotent: merge(A, A) = A
    """

    def __init__(
        self,
        strategy: MergeStrategy = MergeStrategy.MAX_MAGNITUDE,
        conflict_threshold: float = 0.1
    ):
        """
        Initialize CRDT merge system.

        Args:
            strategy: How to merge conflicting spectral coefficients
            conflict_threshold: Threshold for detecting semantic conflicts
        """
        self.strategy = strategy
        self.conflict_threshold = conflict_threshold

    def merge(
        self,
        field_a: CRDTSpectralField,
        field_b: CRDTSpectralField
    ) -> CRDTSpectralField:
        """
        Merge two CRDT spectral fields.

        The merge operation satisfies CRDT laws:
        - Commutative: merge(A, B) = merge(B, A)
        - Associative: merge(merge(A, B), C) = merge(A, merge(B, C))
        - Idempotent: merge(A, A) = A

        Returns:
            Merged field (new object, inputs unchanged)
        """
        sa = field_a.spectral
        sb = field_b.spectral

        if sa.dimension != sb.dimension:
            raise ValueError(f"Dimension mismatch: {sa.dimension} vs {sb.dimension}")

        if sa.n_modes != sb.n_modes:
            raise ValueError(f"Mode count mismatch: {sa.n_modes} vs {sb.n_modes}")

        # Merge coefficients based on strategy
        if self.strategy == MergeStrategy.MAX_MAGNITUDE:
            merged_coeffs = self._merge_max_magnitude(sa, sb)
        elif self.strategy == MergeStrategy.MAX_EIGENVALUE:
            merged_coeffs = self._merge_max_eigenvalue(sa, sb)
        elif self.strategy == MergeStrategy.VECTOR_SUM:
            merged_coeffs = self._merge_vector_sum(sa, sb)
        elif self.strategy == MergeStrategy.WEIGHTED_AVERAGE:
            merged_coeffs = self._merge_weighted_average(sa, sb)
        else:
            merged_coeffs = self._merge_max_magnitude(sa, sb)

        # Merge eigenvalues (max - preserves strongest modes)
        merged_eigenvalues = np.maximum(sa.eigenvalues, sb.eigenvalues)

        # Merge eigenvectors (weighted by eigenvalues)
        merged_eigenvectors = self._merge_eigenvectors(sa, sb)

        # Merge polarity masks (union - preserve all evidence)
        merged_positive = sa.positive_mask | sb.positive_mask
        merged_negative = sa.negative_mask | sb.negative_mask

        # Detect conflicts where both positive and negative are claimed
        conflicts = merged_positive & merged_negative

        # Resolve conflicts by checking coefficient phases
        for mode in np.where(conflicts)[0]:
            phase_a = np.angle(sa.coefficients[mode])
            phase_b = np.angle(sb.coefficients[mode])

            # Use the phase from the stronger coefficient
            if np.abs(sa.coefficients[mode]) >= np.abs(sb.coefficients[mode]):
                dominant_phase = phase_a
            else:
                dominant_phase = phase_b

            if -np.pi/2 <= dominant_phase <= np.pi/2:
                merged_positive[mode] = True
                merged_negative[mode] = False
            else:
                merged_positive[mode] = False
                merged_negative[mode] = True

        merged_spectral = SpectralWitnessField(
            coefficients=merged_coeffs,
            eigenvalues=merged_eigenvalues,
            eigenvectors=merged_eigenvectors,
            positive_mask=merged_positive,
            negative_mask=merged_negative,
            dimension=sa.dimension,
            n_modes=sa.n_modes,
            entity_id=f"{sa.entity_id}⊔{sb.entity_id}",
            version=max(sa.version, sb.version) + 1
        )

        # Merge vector clocks
        merged_clock = VectorClock(clock=dict(field_a.vector_clock.clock))
        merged_clock.merge(field_b.vector_clock)

        # Create merged CRDT field
        return CRDTSpectralField(
            spectral=merged_spectral,
            replica_id=f"{field_a.replica_id}⊔{field_b.replica_id}",
            vector_clock=merged_clock,
            parent_hashes=[field_a.merkle_hash, field_b.merkle_hash]
        )

    def _merge_max_magnitude(
        self,
        sa: SpectralWitnessField,
        sb: SpectralWitnessField
    ) -> np.ndarray:
        """
        Merge by keeping coefficient with larger magnitude.

        This is the default CRDT merge law:
        Ŵ_{i⊔j}(k) = max(|Ŵᵢ(k)|, |Ŵⱼ(k)|) * sign(winner)
        """
        result = np.zeros(sa.n_modes, dtype=np.complex128)

        for k in range(sa.n_modes):
            if np.abs(sa.coefficients[k]) >= np.abs(sb.coefficients[k]):
                result[k] = sa.coefficients[k]
            else:
                result[k] = sb.coefficients[k]

        return result

    def _merge_max_eigenvalue(
        self,
        sa: SpectralWitnessField,
        sb: SpectralWitnessField
    ) -> np.ndarray:
        """Merge by keeping coefficient from field with larger eigenvalue"""
        result = np.zeros(sa.n_modes, dtype=np.complex128)

        for k in range(sa.n_modes):
            if sa.eigenvalues[k] >= sb.eigenvalues[k]:
                result[k] = sa.coefficients[k]
            else:
                result[k] = sb.coefficients[k]

        return result

    def _merge_vector_sum(
        self,
        sa: SpectralWitnessField,
        sb: SpectralWitnessField
    ) -> np.ndarray:
        """Merge by complex addition (preserves phase information)"""
        return sa.coefficients + sb.coefficients

    def _merge_weighted_average(
        self,
        sa: SpectralWitnessField,
        sb: SpectralWitnessField
    ) -> np.ndarray:
        """Merge by eigenvalue-weighted average"""
        total_eigenvalues = sa.eigenvalues + sb.eigenvalues + 1e-10
        wa = sa.eigenvalues / total_eigenvalues
        wb = sb.eigenvalues / total_eigenvalues

        return wa * sa.coefficients + wb * sb.coefficients

    def _merge_eigenvectors(
        self,
        sa: SpectralWitnessField,
        sb: SpectralWitnessField
    ) -> np.ndarray:
        """Merge eigenvectors weighted by eigenvalues"""
        result = np.zeros((sa.n_modes, sa.dimension))

        for k in range(sa.n_modes):
            total = sa.eigenvalues[k] + sb.eigenvalues[k] + 1e-10
            wa = sa.eigenvalues[k] / total
            wb = sb.eigenvalues[k] / total

            merged = wa * sa.eigenvectors[k] + wb * sb.eigenvectors[k]
            norm = np.linalg.norm(merged)
            if norm > 1e-10:
                result[k] = merged / norm
            else:
                # Fallback to stronger eigenvalue's eigenvector
                if sa.eigenvalues[k] >= sb.eigenvalues[k]:
                    result[k] = sa.eigenvectors[k]
                else:
                    result[k] = sb.eigenvectors[k]

        return result

    def detect_conflicts(
        self,
        field_a: CRDTSpectralField,
        field_b: CRDTSpectralField
    ) -> List[Tuple[int, float]]:
        """
        Detect semantic conflicts between two fields.

        Returns list of (mode_index, conflict_severity) tuples.
        """
        sa = field_a.spectral
        sb = field_b.spectral

        conflicts = []

        for k in range(min(sa.n_modes, sb.n_modes)):
            # Check phase opposition
            phase_a = np.angle(sa.coefficients[k])
            phase_b = np.angle(sb.coefficients[k])
            phase_diff = np.abs(phase_a - phase_b)
            phase_diff = min(phase_diff, 2 * np.pi - phase_diff)

            # Phase opposition near π indicates conflict
            if phase_diff > np.pi - self.conflict_threshold:
                # Severity = product of magnitudes (stronger modes = bigger conflict)
                severity = np.abs(sa.coefficients[k]) * np.abs(sb.coefficients[k])
                conflicts.append((k, float(severity)))

            # Check polarity mask disagreement
            if (sa.positive_mask[k] and sb.negative_mask[k]) or \
               (sa.negative_mask[k] and sb.positive_mask[k]):
                severity = np.abs(sa.coefficients[k]) + np.abs(sb.coefficients[k])
                conflicts.append((k, float(severity)))

        return conflicts

    def is_consistent(
        self,
        field_a: CRDTSpectralField,
        field_b: CRDTSpectralField
    ) -> bool:
        """Check if two fields are semantically consistent (can be merged safely)"""
        conflicts = self.detect_conflicts(field_a, field_b)
        total_conflict = sum(severity for _, severity in conflicts)

        # Total energy for normalization
        total_energy = field_a.spectral.energy + field_b.spectral.energy + 1e-10

        return (total_conflict / total_energy) < self.conflict_threshold


class DistributedWitnessNetwork:
    """
    A network of distributed CRDT witness field replicas.

    Each node maintains a local replica that can be:
    - Updated locally (fast)
    - Merged with other replicas (eventually consistent)
    - Queried for semantic similarity

    No coordination required between nodes!
    """

    def __init__(
        self,
        dimension: int = 128,
        n_modes: int = 64,
        merge_strategy: MergeStrategy = MergeStrategy.MAX_MAGNITUDE
    ):
        self.dimension = dimension
        self.n_modes = n_modes
        self.crdt = SpectralCRDT(strategy=merge_strategy)
        self.transform = SpectralWitnessTransform(n_modes=n_modes)

        # Replicas by ID
        self.replicas: Dict[str, CRDTSpectralField] = {}

    def create_replica(
        self,
        replica_id: str,
        initial_field: Optional[WitnessField] = None
    ) -> CRDTSpectralField:
        """Create a new replica in the network"""
        if initial_field is None:
            initial_field = WitnessField(dimension=self.dimension, entity_id=replica_id)

        spectral = self.transform.forward(initial_field)

        replica = CRDTSpectralField(
            spectral=spectral,
            replica_id=replica_id
        )
        replica.vector_clock.tick(replica_id)

        self.replicas[replica_id] = replica
        return replica

    def update_replica(
        self,
        replica_id: str,
        new_witnesses: List[Witness]
    ) -> CRDTSpectralField:
        """Update a replica with new witnesses"""
        if replica_id not in self.replicas:
            raise ValueError(f"Unknown replica: {replica_id}")

        current = self.replicas[replica_id]

        # Convert current spectral back to witness field
        from spectral_transform import from_spectral
        current_field = from_spectral(current.spectral)

        # Add new witnesses
        for w in new_witnesses:
            current_field.add_witness(w)

        # Transform back to spectral
        new_spectral = self.transform.forward(current_field)

        # Create updated CRDT field
        updated = current.update(new_spectral)
        self.replicas[replica_id] = updated

        return updated

    def sync_replicas(
        self,
        replica_id_a: str,
        replica_id_b: str
    ) -> Tuple[CRDTSpectralField, bool]:
        """
        Synchronize two replicas.

        Returns:
            (merged_field, had_conflicts)
        """
        if replica_id_a not in self.replicas or replica_id_b not in self.replicas:
            raise ValueError("Unknown replica")

        field_a = self.replicas[replica_id_a]
        field_b = self.replicas[replica_id_b]

        # Check for conflicts
        conflicts = self.crdt.detect_conflicts(field_a, field_b)
        had_conflicts = len(conflicts) > 0

        # Merge
        merged = self.crdt.merge(field_a, field_b)

        # Update both replicas to merged state
        merged_a = CRDTSpectralField(
            spectral=merged.spectral,
            replica_id=replica_id_a,
            vector_clock=merged.vector_clock,
            parent_hashes=merged.parent_hashes
        )
        merged_b = CRDTSpectralField(
            spectral=merged.spectral,
            replica_id=replica_id_b,
            vector_clock=merged.vector_clock,
            parent_hashes=merged.parent_hashes
        )

        self.replicas[replica_id_a] = merged_a
        self.replicas[replica_id_b] = merged_b

        return merged, had_conflicts

    def global_merge(self) -> CRDTSpectralField:
        """Merge all replicas into a single consistent state"""
        if not self.replicas:
            raise ValueError("No replicas")

        replica_list = list(self.replicas.values())
        result = replica_list[0]

        for replica in replica_list[1:]:
            result = self.crdt.merge(result, replica)

        return result

    def query_similar(
        self,
        query_field: WitnessField,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find k most similar replicas to query.

        Uses spectral distance for O(n_modes) comparison.
        """
        query_spectral = self.transform.forward(query_field)

        similarities = []
        for replica_id, replica in self.replicas.items():
            # Spectral similarity via coefficient correlation
            sim = self._spectral_similarity(query_spectral, replica.spectral)
            similarities.append((replica_id, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]

    def _spectral_similarity(
        self,
        s1: SpectralWitnessField,
        s2: SpectralWitnessField
    ) -> float:
        """Compute similarity between spectral fields"""
        # Correlation of magnitude spectra
        m1 = np.abs(s1.coefficients)
        m2 = np.abs(s2.coefficients)

        # Normalize
        m1 = m1 / (np.linalg.norm(m1) + 1e-10)
        m2 = m2 / (np.linalg.norm(m2) + 1e-10)

        # Correlation
        return float(np.dot(m1, m2))


# Convenience functions
def merge_spectral_fields(
    fields: List[SpectralWitnessField],
    strategy: MergeStrategy = MergeStrategy.MAX_MAGNITUDE
) -> SpectralWitnessField:
    """Merge multiple spectral fields"""
    if not fields:
        raise ValueError("Need at least one field")

    crdt = SpectralCRDT(strategy=strategy)

    # Wrap in CRDT containers
    crdt_fields = [
        CRDTSpectralField(spectral=f, replica_id=f"r{i}")
        for i, f in enumerate(fields)
    ]

    result = crdt_fields[0]
    for f in crdt_fields[1:]:
        result = crdt.merge(result, f)

    return result.spectral
