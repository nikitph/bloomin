"""
Witness Algebra - The foundational algebraic structure for SCWF

Implements (W, ⊕) - a commutative, idempotent monoid of witness sets
with ★ as convolution-like witness composition.

Key properties:
- ⊕ is associative, commutative, and idempotent (W ⊕ W = W)
- ★ is convolution-like: (W₁ ★ W₂)(x) = Σᵧ W₁(y)W₂(x-y)
- Partial order ⪯ induced by information refinement
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib


class WitnessPolarity(Enum):
    """Polarity of evidence: positive (supports), negative (refutes), or neutral"""
    POSITIVE = 1
    NEGATIVE = -1
    NEUTRAL = 0


@dataclass
class Witness:
    """
    A single witness - a piece of evidence on S^{d-1}

    Attributes:
        vector: Unit vector representation
        polarity: Whether this supports (+), refutes (-), or is neutral
        strength: Confidence/weight of this evidence
        source_id: Provenance tracking
        timestamp: Logical clock for CRDT ordering
    """
    vector: np.ndarray
    polarity: WitnessPolarity = WitnessPolarity.POSITIVE
    strength: float = 1.0
    source_id: str = ""
    timestamp: int = 0

    def __post_init__(self):
        # Normalize to unit sphere
        norm = np.linalg.norm(self.vector)
        if norm > 1e-10:
            self.vector = self.vector / norm

    @property
    def dimension(self) -> int:
        return len(self.vector)

    def hash(self) -> str:
        """Content-addressable hash for deduplication"""
        data = self.vector.tobytes() + str(self.polarity.value).encode()
        return hashlib.sha256(data).hexdigest()[:16]

    def similarity(self, other: 'Witness') -> float:
        """Cosine similarity between witnesses"""
        return float(np.dot(self.vector, other.vector))

    def __eq__(self, other):
        if not isinstance(other, Witness):
            return False
        return np.allclose(self.vector, other.vector, atol=1e-6)

    def __hash__(self):
        return hash(self.hash())


@dataclass
class WitnessField:
    """
    A field of witnesses - the fundamental SCWF data structure.

    This is an element of the witness algebra (W, ⊕).
    Each field contains:
    - Positive witnesses W⁺ (supporting evidence)
    - Negative witnesses W⁻ (refuting evidence)
    - Spectral coefficients (computed lazily)

    The field satisfies:
    - Information monotonicity: more witnesses = more information
    - Hemisphere constraint: witnesses in same hemisphere are consistent
    """
    dimension: int
    positive_witnesses: List[Witness] = field(default_factory=list)
    negative_witnesses: List[Witness] = field(default_factory=list)
    entity_id: str = ""
    version: int = 0  # Vector clock component

    # Cached computations
    _spectral_coeffs: Optional[np.ndarray] = field(default=None, repr=False)
    _entropy: Optional[float] = field(default=None, repr=False)

    def __post_init__(self):
        self._invalidate_cache()

    def _invalidate_cache(self):
        self._spectral_coeffs = None
        self._entropy = None

    @property
    def all_witnesses(self) -> List[Witness]:
        return self.positive_witnesses + self.negative_witnesses

    @property
    def n_positive(self) -> int:
        return len(self.positive_witnesses)

    @property
    def n_negative(self) -> int:
        return len(self.negative_witnesses)

    @property
    def n_total(self) -> int:
        return self.n_positive + self.n_negative

    def add_witness(self, witness: Witness):
        """Add a witness to the field"""
        self._invalidate_cache()
        if witness.polarity == WitnessPolarity.POSITIVE:
            self.positive_witnesses.append(witness)
        elif witness.polarity == WitnessPolarity.NEGATIVE:
            self.negative_witnesses.append(witness)
        else:
            # Neutral witnesses go to positive by default
            self.positive_witnesses.append(witness)
        self.version += 1

    def positive_centroid(self) -> Optional[np.ndarray]:
        """Normalized centroid of positive witnesses"""
        if not self.positive_witnesses:
            return None
        vecs = np.array([w.vector * w.strength for w in self.positive_witnesses])
        centroid = np.mean(vecs, axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 1e-10 else centroid

    def negative_centroid(self) -> Optional[np.ndarray]:
        """Normalized centroid of negative witnesses"""
        if not self.negative_witnesses:
            return None
        vecs = np.array([w.vector * w.strength for w in self.negative_witnesses])
        centroid = np.mean(vecs, axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 1e-10 else centroid

    def consistency_score(self) -> float:
        """
        Measure consistency of the witness field.

        Returns value in [0, 1]:
        - 1.0 = perfectly consistent (all witnesses in same hemisphere)
        - 0.0 = maximally inconsistent (positive/negative overlap)
        """
        if self.n_positive == 0 or self.n_negative == 0:
            return 1.0

        pos_centroid = self.positive_centroid()
        neg_centroid = self.negative_centroid()

        if pos_centroid is None or neg_centroid is None:
            return 1.0

        # Consistency is high when centroids point in opposite directions
        separation = -np.dot(pos_centroid, neg_centroid)
        return max(0.0, min(1.0, (separation + 1.0) / 2.0))

    def witness_entropy(self) -> float:
        """
        Compute witness entropy H_W - measures information content.

        High entropy = witnesses are spread out (uncertain)
        Low entropy = witnesses are concentrated (certain)
        """
        if self._entropy is not None:
            return self._entropy

        if self.n_total == 0:
            self._entropy = 0.0
            return self._entropy

        # Compute pairwise similarities
        all_vecs = np.array([w.vector for w in self.all_witnesses])
        similarities = all_vecs @ all_vecs.T

        # Convert to probability distribution via softmax
        # Higher temperature = more uniform distribution
        temperature = 1.0
        exp_sims = np.exp(similarities / temperature)
        probs = exp_sims / np.sum(exp_sims)

        # Shannon entropy
        probs_flat = probs.flatten()
        probs_flat = probs_flat[probs_flat > 1e-10]
        self._entropy = -np.sum(probs_flat * np.log(probs_flat))

        return self._entropy

    def to_matrix(self) -> np.ndarray:
        """Convert to matrix representation (n_witnesses x dimension)"""
        if self.n_total == 0:
            return np.zeros((0, self.dimension))

        all_vecs = []
        for w in self.positive_witnesses:
            all_vecs.append(w.vector * w.strength)
        for w in self.negative_witnesses:
            all_vecs.append(-w.vector * w.strength)  # Negate for negative

        return np.array(all_vecs)

    @classmethod
    def from_vectors(
        cls,
        vectors: np.ndarray,
        polarities: Optional[List[WitnessPolarity]] = None,
        entity_id: str = ""
    ) -> 'WitnessField':
        """Create field from array of vectors"""
        n, d = vectors.shape

        if polarities is None:
            polarities = [WitnessPolarity.POSITIVE] * n

        field = cls(dimension=d, entity_id=entity_id)

        for i, (vec, pol) in enumerate(zip(vectors, polarities)):
            witness = Witness(vector=vec, polarity=pol)
            field.add_witness(witness)

        return field


class WitnessAlgebra:
    """
    The witness algebra (W, ⊕, ★) - a commutative idempotent monoid.

    Operations:
    - ⊕ (join): Combines witness fields (idempotent, commutative, associative)
    - ★ (convolution): Composes witness fields (like semantic composition)

    This algebra satisfies:
    1. W ⊕ W = W (idempotent)
    2. W₁ ⊕ W₂ = W₂ ⊕ W₁ (commutative)
    3. (W₁ ⊕ W₂) ⊕ W₃ = W₁ ⊕ (W₂ ⊕ W₃) (associative)
    """

    def __init__(
        self,
        dimension: int,
        dedup_threshold: float = 0.99,
        max_witnesses_per_field: int = 1024
    ):
        self.dimension = dimension
        self.dedup_threshold = dedup_threshold
        self.max_witnesses = max_witnesses_per_field

    def join(self, w1: WitnessField, w2: WitnessField) -> WitnessField:
        """
        ⊕ operation: Join two witness fields.

        Properties:
        - Idempotent: W ⊕ W = W
        - Commutative: W₁ ⊕ W₂ = W₂ ⊕ W₁
        - Associative: (W₁ ⊕ W₂) ⊕ W₃ = W₁ ⊕ (W₂ ⊕ W₃)

        Implementation:
        - Union of witnesses with deduplication
        - Conflicting witnesses are both kept (preserved for resolution)
        """
        if w1.dimension != w2.dimension:
            raise ValueError(f"Dimension mismatch: {w1.dimension} vs {w2.dimension}")

        result = WitnessField(
            dimension=w1.dimension,
            entity_id=f"{w1.entity_id}⊕{w2.entity_id}",
            version=max(w1.version, w2.version) + 1
        )

        # Collect all witnesses
        all_positive = w1.positive_witnesses + w2.positive_witnesses
        all_negative = w1.negative_witnesses + w2.negative_witnesses

        # Deduplicate positive witnesses
        result.positive_witnesses = self._deduplicate_witnesses(all_positive)

        # Deduplicate negative witnesses
        result.negative_witnesses = self._deduplicate_witnesses(all_negative)

        # Apply max witness limit (keep strongest)
        if result.n_total > self.max_witnesses:
            all_w = result.all_witnesses
            all_w.sort(key=lambda w: w.strength, reverse=True)
            all_w = all_w[:self.max_witnesses]

            result.positive_witnesses = [
                w for w in all_w if w.polarity == WitnessPolarity.POSITIVE
            ]
            result.negative_witnesses = [
                w for w in all_w if w.polarity == WitnessPolarity.NEGATIVE
            ]

        return result

    def _deduplicate_witnesses(self, witnesses: List[Witness]) -> List[Witness]:
        """Remove near-duplicate witnesses, keeping the stronger one"""
        if not witnesses:
            return []

        deduped = []
        for w in witnesses:
            is_dup = False
            for i, existing in enumerate(deduped):
                if w.similarity(existing) > self.dedup_threshold:
                    # Keep the stronger one
                    if w.strength > existing.strength:
                        deduped[i] = w
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(w)

        return deduped

    def convolve(self, w1: WitnessField, w2: WitnessField) -> WitnessField:
        """
        ★ operation: Convolve two witness fields.

        This is analogous to semantic composition:
        (W₁ ★ W₂)(x) = Σᵧ W₁(y)W₂(x-y)

        In practice, we compute cross-products of witnesses
        and take principal components.
        """
        if w1.dimension != w2.dimension:
            raise ValueError(f"Dimension mismatch: {w1.dimension} vs {w2.dimension}")

        result = WitnessField(
            dimension=w1.dimension,
            entity_id=f"{w1.entity_id}★{w2.entity_id}",
            version=max(w1.version, w2.version) + 1
        )

        # Compute convolution via cross-products
        # For efficiency, we use random sampling if too many pairs
        max_pairs = 256

        # Positive ★ Positive → Positive
        pos_pairs = self._sample_pairs(
            w1.positive_witnesses,
            w2.positive_witnesses,
            max_pairs
        )
        for wa, wb in pos_pairs:
            # Convolution: element-wise product normalized to sphere
            conv_vec = wa.vector * wb.vector
            norm = np.linalg.norm(conv_vec)
            if norm > 1e-10:
                conv_vec = conv_vec / norm
                result.add_witness(Witness(
                    vector=conv_vec,
                    polarity=WitnessPolarity.POSITIVE,
                    strength=wa.strength * wb.strength
                ))

        # Negative ★ Negative → Positive (double negative)
        neg_pairs = self._sample_pairs(
            w1.negative_witnesses,
            w2.negative_witnesses,
            max_pairs
        )
        for wa, wb in neg_pairs:
            conv_vec = wa.vector * wb.vector
            norm = np.linalg.norm(conv_vec)
            if norm > 1e-10:
                conv_vec = conv_vec / norm
                result.add_witness(Witness(
                    vector=conv_vec,
                    polarity=WitnessPolarity.POSITIVE,
                    strength=wa.strength * wb.strength
                ))

        # Positive ★ Negative → Negative
        cross_pairs = self._sample_pairs(
            w1.positive_witnesses,
            w2.negative_witnesses,
            max_pairs
        )
        for wa, wb in cross_pairs:
            conv_vec = wa.vector * wb.vector
            norm = np.linalg.norm(conv_vec)
            if norm > 1e-10:
                conv_vec = conv_vec / norm
                result.add_witness(Witness(
                    vector=conv_vec,
                    polarity=WitnessPolarity.NEGATIVE,
                    strength=wa.strength * wb.strength
                ))

        # Deduplicate
        result.positive_witnesses = self._deduplicate_witnesses(result.positive_witnesses)
        result.negative_witnesses = self._deduplicate_witnesses(result.negative_witnesses)

        return result

    def _sample_pairs(
        self,
        list1: List[Witness],
        list2: List[Witness],
        max_pairs: int
    ) -> List[Tuple[Witness, Witness]]:
        """Sample pairs from two lists"""
        if not list1 or not list2:
            return []

        total_pairs = len(list1) * len(list2)

        if total_pairs <= max_pairs:
            return [(a, b) for a in list1 for b in list2]

        # Random sampling
        pairs = []
        for _ in range(max_pairs):
            a = list1[np.random.randint(len(list1))]
            b = list2[np.random.randint(len(list2))]
            pairs.append((a, b))

        return pairs

    def identity(self) -> WitnessField:
        """Identity element for ⊕: empty field"""
        return WitnessField(dimension=self.dimension, entity_id="∅")

    def is_refinement(self, w1: WitnessField, w2: WitnessField) -> bool:
        """
        Check if w1 ⪯ w2 (w2 refines w1).

        w2 refines w1 if w2 contains more specific information.
        """
        # w2 refines w1 if w2 has more witnesses and lower entropy
        if w2.n_total < w1.n_total:
            return False

        return w2.witness_entropy() <= w1.witness_entropy()

    def semantic_distance(self, w1: WitnessField, w2: WitnessField) -> float:
        """
        Compute semantic distance between witness fields.

        This is the REWA metric: based on witness overlap and entropy.
        """
        if w1.n_total == 0 and w2.n_total == 0:
            return 0.0

        if w1.n_total == 0 or w2.n_total == 0:
            return 1.0

        # Compute witness overlap
        overlap = 0.0
        count = 0

        for w1_wit in w1.all_witnesses:
            for w2_wit in w2.all_witnesses:
                sim = w1_wit.similarity(w2_wit)
                # Only count if same polarity
                if w1_wit.polarity == w2_wit.polarity:
                    overlap += max(0, sim)
                else:
                    overlap -= max(0, sim)  # Penalize opposite polarity
                count += 1

        if count == 0:
            return 1.0

        normalized_overlap = overlap / count
        return 1.0 - max(0, min(1, (normalized_overlap + 1) / 2))


# Convenience functions
def join_fields(*fields: WitnessField) -> WitnessField:
    """Join multiple witness fields"""
    if not fields:
        raise ValueError("Need at least one field")

    algebra = WitnessAlgebra(dimension=fields[0].dimension)
    result = fields[0]

    for f in fields[1:]:
        result = algebra.join(result, f)

    return result


def convolve_fields(*fields: WitnessField) -> WitnessField:
    """Convolve multiple witness fields"""
    if not fields:
        raise ValueError("Need at least one field")

    algebra = WitnessAlgebra(dimension=fields[0].dimension)
    result = fields[0]

    for f in fields[1:]:
        result = algebra.convolve(result, f)

    return result
