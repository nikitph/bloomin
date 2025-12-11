"""
WitnessSet: A collection of unit vectors on S^{d-1} representing evidence for a variable.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass, field


@dataclass
class WitnessSet:
    """
    A set of witnesses (unit vectors on the sphere) for a causal variable.

    Each witness represents a piece of evidence - could be from text embeddings,
    attribute encodings, or learned representations.

    Attributes:
        variable: Name of the variable this witness set represents
        witnesses: Array of unit vectors, shape (n_witnesses, d)
        metadata: Optional metadata about each witness (source, confidence, etc.)
        dimension: Embedding dimension d
    """
    variable: str
    witnesses: np.ndarray
    metadata: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Validate and normalize witnesses to unit sphere."""
        if len(self.witnesses.shape) == 1:
            self.witnesses = self.witnesses.reshape(1, -1)

        # Normalize to unit sphere
        norms = np.linalg.norm(self.witnesses, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        self.witnesses = self.witnesses / norms

        # Ensure metadata list matches
        if len(self.metadata) < len(self.witnesses):
            self.metadata.extend([{}] * (len(self.witnesses) - len(self.metadata)))

    @property
    def dimension(self) -> int:
        """Embedding dimension d."""
        return self.witnesses.shape[1]

    @property
    def n_witnesses(self) -> int:
        """Number of witnesses."""
        return self.witnesses.shape[0]

    def add_witness(self, w: np.ndarray, meta: Optional[Dict[str, Any]] = None):
        """Add a new witness to the set."""
        w = np.atleast_2d(w)
        norm = np.linalg.norm(w, axis=1, keepdims=True)
        norm = np.where(norm > 1e-10, norm, 1.0)
        w = w / norm

        self.witnesses = np.vstack([self.witnesses, w])
        self.metadata.append(meta or {})

    def get_witness(self, idx: int) -> np.ndarray:
        """Get a single witness by index."""
        return self.witnesses[idx]

    def subset(self, indices: List[int]) -> 'WitnessSet':
        """Create a subset of witnesses."""
        return WitnessSet(
            variable=self.variable,
            witnesses=self.witnesses[indices],
            metadata=[self.metadata[i] for i in indices]
        )

    def filter_by_metadata(self, key: str, value: Any) -> 'WitnessSet':
        """Filter witnesses by metadata value."""
        indices = [i for i, m in enumerate(self.metadata) if m.get(key) == value]
        return self.subset(indices)

    def mean_witness(self) -> np.ndarray:
        """Compute the (normalized) mean of all witnesses."""
        mean = np.mean(self.witnesses, axis=0)
        norm = np.linalg.norm(mean)
        if norm > 1e-10:
            return mean / norm
        return mean

    def pairwise_similarities(self) -> np.ndarray:
        """Compute pairwise cosine similarities between witnesses."""
        return self.witnesses @ self.witnesses.T

    def min_similarity(self) -> float:
        """Minimum pairwise similarity (for hemisphere check)."""
        sims = self.pairwise_similarities()
        np.fill_diagonal(sims, 1.0)  # Ignore self-similarity
        return float(np.min(sims))

    def merge(self, other: 'WitnessSet') -> 'WitnessSet':
        """Merge with another witness set."""
        if self.dimension != other.dimension:
            raise ValueError(f"Dimension mismatch: {self.dimension} vs {other.dimension}")

        return WitnessSet(
            variable=f"{self.variable}+{other.variable}",
            witnesses=np.vstack([self.witnesses, other.witnesses]),
            metadata=self.metadata + other.metadata
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'variable': self.variable,
            'witnesses': self.witnesses.tolist(),
            'metadata': self.metadata,
            'dimension': self.dimension,
            'n_witnesses': self.n_witnesses
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WitnessSet':
        """Create from dictionary."""
        return cls(
            variable=data['variable'],
            witnesses=np.array(data['witnesses']),
            metadata=data.get('metadata', [])
        )

    def __len__(self) -> int:
        return self.n_witnesses

    def __repr__(self) -> str:
        return f"WitnessSet(variable='{self.variable}', n={self.n_witnesses}, d={self.dimension})"
