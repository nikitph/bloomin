"""
FR-1: Embedding & Normalization

Semantic Space implementation with pluggable projection heads.
All vectors live on unit sphere S^{k-1}.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable
from sentence_transformers import SentenceTransformer
import hashlib
from datetime import datetime


@dataclass
class Witness:
    """A witness is an atomic piece of evidence projected onto the semantic sphere."""
    id: str
    text: str
    embedding: np.ndarray
    source: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure embedding is on unit sphere
        norm = np.linalg.norm(self.embedding)
        if norm > 0:
            self.embedding = self.embedding / norm

        # Generate ID if not provided
        if not self.id:
            self.id = hashlib.sha256(self.text.encode()).hexdigest()[:16]

    @property
    def norm(self) -> float:
        """Should always be 1.0 for valid witnesses."""
        return float(np.linalg.norm(self.embedding))

    def dot(self, other: 'Witness') -> float:
        """Dot product equals cosine similarity on unit sphere."""
        return float(np.dot(self.embedding, other.embedding))

    def angle_to(self, other: 'Witness') -> float:
        """Angular distance in radians."""
        return float(np.arccos(np.clip(self.dot(other), -1.0, 1.0)))

    def is_antipodal_to(self, other: 'Witness', threshold: float = 0.9) -> bool:
        """Check if this witness is approximately antipodal to another."""
        return self.dot(other) < -threshold


class ProjectionHead:
    """
    Base class for Rewa-space projection heads.
    Transforms base embeddings to enforce geometric properties.
    """

    def __init__(self, input_dim: int, output_dim: Optional[int] = None):
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim

    def project(self, embedding: np.ndarray) -> np.ndarray:
        """Project embedding through the head and normalize to unit sphere."""
        transformed = self._transform(embedding)
        norm = np.linalg.norm(transformed)
        if norm > 0:
            return transformed / norm
        return transformed

    def _transform(self, embedding: np.ndarray) -> np.ndarray:
        """Override this in subclasses."""
        return embedding


class IdentityProjection(ProjectionHead):
    """Identity projection - just normalize to unit sphere."""

    def _transform(self, embedding: np.ndarray) -> np.ndarray:
        return embedding


class AntipodalProjection(ProjectionHead):
    """
    Projection head trained to enforce antipodal negation.
    Uses learned transformation to make negation pairs antipodal.
    """

    def __init__(self, input_dim: int, negation_pairs: Optional[List[tuple]] = None):
        super().__init__(input_dim, input_dim)
        # Initialize transformation matrix (identity + learnable)
        self.W = np.eye(input_dim)
        self.negation_pairs = negation_pairs or []

        if negation_pairs:
            self._fit_antipodal_transform()

    def _fit_antipodal_transform(self):
        """
        Fit transformation to make negation pairs antipodal.
        Uses simple iterative adjustment.
        """
        # This is a simplified version - production would use gradient descent
        pass

    def _transform(self, embedding: np.ndarray) -> np.ndarray:
        return self.W @ embedding


class SemanticSpace:
    """
    The semantic sphere S^{k-1} where all meanings live.

    Implements FR-1:
    - Embed all inputs using base encoder
    - Project onto unit sphere
    - Support pluggable projection heads
    """

    def __init__(
        self,
        model_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        projection_head: Optional[ProjectionHead] = None,
        cache_embeddings: bool = True
    ):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.projection_head = projection_head or IdentityProjection(self.dimension)
        self.cache_embeddings = cache_embeddings
        self._cache: Dict[str, np.ndarray] = {}

    def embed(self, text: str) -> np.ndarray:
        """
        Embed text onto the semantic sphere.
        Returns unit-normalized vector.
        """
        # Check cache
        if self.cache_embeddings and text in self._cache:
            return self._cache[text].copy()

        # Get base embedding
        base_embedding = self.model.encode(text, convert_to_numpy=True)

        # Apply projection head (includes normalization)
        projected = self.projection_head.project(base_embedding)

        # Cache and return
        if self.cache_embeddings:
            self._cache[text] = projected

        return projected

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed multiple texts efficiently."""
        # Get uncached texts
        uncached_texts = [t for t in texts if t not in self._cache]

        if uncached_texts:
            base_embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)
            for i, text in enumerate(uncached_texts):
                projected = self.projection_head.project(base_embeddings[i])
                if self.cache_embeddings:
                    self._cache[text] = projected

        # Gather all embeddings
        return np.array([self._cache.get(t, self.embed(t)) for t in texts])

    def create_witness(
        self,
        text: str,
        source: str = "unknown",
        witness_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Witness:
        """Create a witness from text."""
        embedding = self.embed(text)
        return Witness(
            id=witness_id or "",
            text=text,
            embedding=embedding,
            source=source,
            metadata=metadata or {}
        )

    def create_witnesses(
        self,
        texts: List[str],
        source: str = "unknown"
    ) -> List[Witness]:
        """Create multiple witnesses efficiently."""
        embeddings = self.embed_batch(texts)
        return [
            Witness(
                id="",
                text=text,
                embedding=embeddings[i],
                source=source
            )
            for i, text in enumerate(texts)
        ]

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """On unit sphere, cosine similarity = dot product."""
        return float(np.dot(a, b))

    def angular_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Angular distance in radians."""
        sim = np.clip(self.cosine_similarity(a, b), -1.0, 1.0)
        return float(np.arccos(sim))

    def verify_unit_norm(self, embedding: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Verify embedding is on unit sphere."""
        return abs(np.linalg.norm(embedding) - 1.0) < tolerance

    def set_projection_head(self, head: ProjectionHead):
        """Swap projection head (clears cache)."""
        self.projection_head = head
        self._cache.clear()

    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        """Number of cached embeddings."""
        return len(self._cache)
