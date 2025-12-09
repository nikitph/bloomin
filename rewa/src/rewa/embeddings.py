"""
Embedding Support for REWA

Provides semantic embeddings using sentence-transformers.
This replaces brittle regex patterns with semantic similarity.
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from rewa.models import Vector
from rewa.geometry.spherical import normalize_embedding, cosine_similarity


class Embedder:
    """
    Sentence embedding using sentence-transformers.

    This provides the semantic foundation for REWA instead of
    brittle regex pattern matching.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder.

        Args:
            model_name: HuggingFace model name for sentence-transformers
        """
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, text: str) -> Vector:
        """Embed a single text string."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return normalize_embedding(embedding)

    def embed_batch(self, texts: List[str]) -> List[Vector]:
        """Embed multiple texts efficiently."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return [normalize_embedding(e) for e in embeddings]

    def similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        e1 = self.embed(text1)
        e2 = self.embed(text2)
        return cosine_similarity(e1, e2)


class SemanticMatcher:
    """
    Semantic matching using embeddings instead of regex.

    Pre-computes embeddings for concept anchors and matches
    via cosine similarity.
    """

    def __init__(self, embedder: Optional[Embedder] = None):
        self.embedder = embedder or Embedder()
        self._concept_embeddings: Dict[str, Vector] = {}
        self._type_anchors: Dict[str, List[str]] = {}
        self._property_anchors: Dict[str, List[Tuple[str, Any]]] = {}
        self._impossibility_anchors: List[Tuple[str, str]] = []

        self._setup_anchors()

    def _setup_anchors(self):
        """Setup semantic anchor phrases for concepts."""
        # Entity type anchors - phrases that indicate each type
        self._type_anchors = {
            "Weapon": [
                "gun", "firearm", "pistol", "rifle", "weapon",
                "handgun", "shotgun", "revolver", "assault rifle",
                "deadly weapon", "lethal weapon"
            ],
            "Toy": [
                "toy", "plaything", "children's toy", "play item",
                "nerf gun", "toy gun", "water gun", "fake gun"
            ],
            "Drug": [
                "drug", "medication", "medicine", "pharmaceutical",
                "prescription drug", "over the counter medicine",
                "treatment medication"
            ],
            "Treatment": [
                "medical treatment", "therapy", "cure", "remedy",
                "therapeutic intervention", "medical procedure"
            ],
            "MedicalCondition": [
                "disease", "illness", "medical condition", "disorder",
                "syndrome", "cancer", "infection", "ailment"
            ],
            "Device": [
                "electronic device", "charger", "appliance", "gadget",
                "equipment", "machine", "electrical device"
            ],
            "Person": [
                "person", "individual", "patient", "human", "someone"
            ],
        }

        # Property anchors - (phrase, value) pairs
        self._property_anchors = {
            "dangerous": [
                ("dangerous", True),
                ("lethal", True),
                ("deadly", True),
                ("harmful", True),
                ("hazardous", True),
                ("safe", False),
                ("harmless", False),
                ("non-lethal", False),
            ],
            "is_toy": [
                ("is a toy", True),
                ("toy version", True),
                ("for children to play with", True),
                ("is real", False),
                ("genuine", False),
                ("authentic", False),
            ],
            "is_real": [
                ("is real", True),
                ("genuine", True),
                ("authentic", True),
                ("actual", True),
                ("is fake", False),
                ("is a toy", False),
                ("replica", False),
            ],
            "has_side_effects": [
                ("has side effects", True),
                ("causes side effects", True),
                ("adverse effects", True),
                ("no side effects", False),
                ("zero side effects", False),
                ("without side effects", False),
                ("side effect free", False),
            ],
            "cures_cancer": [
                ("cures cancer", True),
                ("treats cancer", True),
                ("cancer treatment", True),
                ("eliminates cancer", True),
                ("fights cancer", True),
            ],
            "usable_for_self_defense": [
                ("for self defense", True),
                ("self protection", True),
                ("personal protection", True),
                ("defend yourself", True),
            ],
        }

        # Impossibility anchors - (pattern, reason) pairs
        self._impossibility_anchors = [
            # Medical impossibilities
            ("cure cancer without side effects",
             "Medical treatments that cure cancer inherently have side effects"),
            ("cancer treatment with zero side effects",
             "Effective cancer treatments have unavoidable side effects"),
            ("medicine that cures cancer with no adverse effects",
             "Cancer cures cannot avoid side effects"),

            # Physical impossibilities
            ("perpetual motion machine",
             "Perpetual motion violates the laws of thermodynamics"),
            ("infinite energy source",
             "Infinite energy violates conservation of energy"),
            ("free energy device",
             "Free energy violates physics"),
            ("faster than light travel",
             "Faster than light travel violates relativity"),

            # Logical impossibilities
            ("square circle",
             "A shape cannot be both square and circular"),
            ("married bachelor",
             "A bachelor by definition is unmarried"),
            ("dry water",
             "Water by definition is wet"),
        ]

    def _get_embedding(self, text: str) -> Vector:
        """Get or compute embedding for text (with caching)."""
        if text not in self._concept_embeddings:
            self._concept_embeddings[text] = self.embedder.embed(text)
        return self._concept_embeddings[text]

    def precompute_anchors(self):
        """Pre-compute embeddings for all anchors."""
        all_texts = []

        # Collect all anchor texts
        for anchors in self._type_anchors.values():
            all_texts.extend(anchors)

        for anchors in self._property_anchors.values():
            all_texts.extend([phrase for phrase, _ in anchors])

        all_texts.extend([pattern for pattern, _ in self._impossibility_anchors])

        # Batch embed
        embeddings = self.embedder.embed_batch(all_texts)

        # Store in cache
        for text, emb in zip(all_texts, embeddings):
            self._concept_embeddings[text] = emb

    def detect_types(
        self,
        text: str,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Detect entity types in text using semantic similarity.

        Returns dict of type -> confidence score.
        """
        text_emb = self.embedder.embed(text)
        detected = {}

        for entity_type, anchors in self._type_anchors.items():
            max_sim = 0.0
            for anchor in anchors:
                anchor_emb = self._get_embedding(anchor)
                sim = cosine_similarity(text_emb, anchor_emb)
                max_sim = max(max_sim, sim)

            if max_sim >= threshold:
                detected[entity_type] = float(max_sim)

        return detected

    def detect_properties(
        self,
        text: str,
        threshold: float = 0.5
    ) -> Dict[str, Tuple[Any, float]]:
        """
        Detect properties in text using semantic similarity.

        Returns dict of property -> (value, confidence).
        """
        text_emb = self.embedder.embed(text)
        detected = {}

        for prop_name, anchors in self._property_anchors.items():
            best_match = None
            best_sim = 0.0

            for anchor_phrase, value in anchors:
                anchor_emb = self._get_embedding(anchor_phrase)
                sim = cosine_similarity(text_emb, anchor_emb)

                if sim > best_sim and sim >= threshold:
                    best_sim = sim
                    best_match = (value, float(sim))

            if best_match:
                detected[prop_name] = best_match

        return detected

    def check_impossibility(
        self,
        query: str,
        threshold: float = 0.7
    ) -> List[Tuple[str, float]]:
        """
        Check if query describes something impossible.

        Returns list of (reason, confidence) for detected impossibilities.
        """
        query_emb = self.embedder.embed(query)
        impossibilities = []

        for pattern, reason in self._impossibility_anchors:
            pattern_emb = self._get_embedding(pattern)
            sim = cosine_similarity(query_emb, pattern_emb)

            if sim >= threshold:
                impossibilities.append((reason, float(sim)))

        # Sort by confidence
        impossibilities.sort(key=lambda x: x[1], reverse=True)
        return impossibilities

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        e1 = self.embedder.embed(text1)
        e2 = self.embedder.embed(text2)
        return cosine_similarity(e1, e2)


class SemanticNegationDetector:
    """
    Detects negation semantically rather than with regex.
    """

    def __init__(self, embedder: Optional[Embedder] = None):
        self.embedder = embedder or Embedder()

        # Negation phrase pairs: (affirmative, negative)
        self._negation_pairs = [
            ("is dangerous", "is not dangerous"),
            ("is safe", "is not safe"),
            ("is real", "is not real"),
            ("is a toy", "is not a toy"),
            ("has side effects", "has no side effects"),
            ("can be used", "cannot be used"),
            ("is effective", "is not effective"),
        ]

        self._pair_embeddings: Dict[str, Tuple[Vector, Vector]] = {}

    def precompute(self):
        """Pre-compute embeddings for negation pairs."""
        for aff, neg in self._negation_pairs:
            aff_emb = self.embedder.embed(aff)
            neg_emb = self.embedder.embed(neg)
            self._pair_embeddings[aff] = (aff_emb, neg_emb)

    def detect_negation(
        self,
        text: str,
        concept: str
    ) -> Tuple[bool, float]:
        """
        Detect if text negates a concept.

        Returns (is_negated, confidence).
        """
        text_emb = self.embedder.embed(text)

        # Find best matching pair
        best_pair = None
        best_sim = 0.0

        for aff, neg in self._negation_pairs:
            if concept.lower() in aff.lower():
                aff_emb, neg_emb = self._pair_embeddings.get(aff, (
                    self.embedder.embed(aff),
                    self.embedder.embed(neg)
                ))

                aff_sim = cosine_similarity(text_emb, aff_emb)
                neg_sim = cosine_similarity(text_emb, neg_emb)

                max_sim = max(aff_sim, neg_sim)
                if max_sim > best_sim:
                    best_sim = max_sim
                    best_pair = (aff_sim, neg_sim)

        if best_pair is None:
            return False, 0.0

        aff_sim, neg_sim = best_pair
        is_negated = neg_sim > aff_sim
        confidence = abs(neg_sim - aff_sim)

        return is_negated, float(confidence)


# Convenience function
def create_embedder(model_name: str = "all-MiniLM-L6-v2") -> Embedder:
    """Create an embedder with the specified model."""
    return Embedder(model_name)
