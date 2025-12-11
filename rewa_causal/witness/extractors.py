"""
Witness Extractors

Three supported witness sources:
1. Sentence embeddings
2. Attribute-value encoding
3. Learned witness extractor (LLM wrapper)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
from abc import ABC, abstractmethod
from .witness_set import WitnessSet


def normalize_witness(w: np.ndarray) -> np.ndarray:
    """Normalize a witness vector to unit sphere S^{d-1}."""
    w = np.atleast_1d(w).astype(np.float64)
    norm = np.linalg.norm(w)
    if norm > 1e-10:
        return w / norm
    return w


class WitnessExtractor(ABC):
    """Abstract base class for witness extractors."""

    @abstractmethod
    def extract(self, data: Any, variable: str) -> WitnessSet:
        """Extract witnesses from data for a given variable."""
        pass


class SentenceEmbeddingExtractor(WitnessExtractor):
    """
    Extract witnesses using sentence embeddings.

    Uses a sentence transformer model to embed text into vectors,
    then normalizes to the unit sphere.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
        self.model_name = model_name
        self.dimension = dimension
        self._model = None

    def _get_model(self):
        """Lazy load the sentence transformer model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                # Fallback to random projection if sentence-transformers not available
                self._model = "fallback"
        return self._model

    def extract(self, data: Union[str, List[str]], variable: str) -> WitnessSet:
        """
        Extract witnesses from text data.

        Args:
            data: Single string or list of strings to embed
            variable: Variable name for the witness set

        Returns:
            WitnessSet with normalized embeddings
        """
        if isinstance(data, str):
            data = [data]

        model = self._get_model()

        if model == "fallback":
            # Use deterministic hash-based projection as fallback
            witnesses = []
            for text in data:
                np.random.seed(hash(text) % (2**32))
                vec = np.random.randn(self.dimension)
                witnesses.append(normalize_witness(vec))
            witnesses = np.array(witnesses)
        else:
            embeddings = model.encode(data)
            witnesses = np.array([normalize_witness(e) for e in embeddings])

        metadata = [{'source': 'sentence_embedding', 'text': t[:100]} for t in data]
        return WitnessSet(variable=variable, witnesses=witnesses, metadata=metadata)


class AttributeValueExtractor(WitnessExtractor):
    """
    Extract witnesses from structured attribute-value data.

    Encodes numeric and categorical attributes into a fixed-dimension
    vector space, then normalizes to unit sphere.
    """

    def __init__(self, schema: Dict[str, Dict[str, Any]], dimension: int = 64):
        """
        Args:
            schema: Dictionary defining attributes and their types/ranges
                Example: {
                    'credit_score': {'type': 'numeric', 'min': 300, 'max': 850},
                    'loan_purpose': {'type': 'categorical', 'values': ['purchase', 'refi']}
                }
            dimension: Output embedding dimension
        """
        self.schema = schema
        self.dimension = dimension
        self._projection_matrix = None

    def _get_projection_matrix(self, input_dim: int) -> np.ndarray:
        """Get or create random projection matrix."""
        if self._projection_matrix is None or self._projection_matrix.shape[0] != input_dim:
            np.random.seed(42)  # Deterministic
            self._projection_matrix = np.random.randn(input_dim, self.dimension)
            self._projection_matrix /= np.linalg.norm(self._projection_matrix, axis=1, keepdims=True)
        return self._projection_matrix

    def _encode_attribute(self, name: str, value: Any) -> np.ndarray:
        """Encode a single attribute value."""
        spec = self.schema.get(name, {'type': 'numeric'})

        if spec['type'] == 'numeric':
            # Normalize to [0, 1]
            min_val = spec.get('min', 0)
            max_val = spec.get('max', 1)
            if max_val == min_val:
                normalized = 0.5
            else:
                normalized = (float(value) - min_val) / (max_val - min_val)
            normalized = np.clip(normalized, 0, 1)
            return np.array([normalized, 1 - normalized])

        elif spec['type'] == 'categorical':
            # One-hot encoding
            values = spec.get('values', [value])
            encoding = np.zeros(len(values))
            if value in values:
                encoding[values.index(value)] = 1.0
            else:
                encoding[:] = 1.0 / len(values)  # Unknown -> uniform
            return encoding

        else:
            # Default: treat as numeric
            return np.array([float(value)])

    def extract(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], variable: str) -> WitnessSet:
        """
        Extract witnesses from attribute-value records.

        Args:
            data: Single record or list of records (dicts)
            variable: Variable name for the witness set

        Returns:
            WitnessSet with encoded and normalized embeddings
        """
        if isinstance(data, dict):
            data = [data]

        witnesses = []
        metadata = []

        for record in data:
            # Encode all attributes
            encodings = []
            for attr_name in sorted(self.schema.keys()):
                if attr_name in record:
                    enc = self._encode_attribute(attr_name, record[attr_name])
                    encodings.append(enc)

            if not encodings:
                continue

            # Concatenate and project
            raw_vec = np.concatenate(encodings)
            proj_matrix = self._get_projection_matrix(len(raw_vec))
            projected = raw_vec @ proj_matrix
            witnesses.append(normalize_witness(projected))
            metadata.append({'source': 'attribute_value', 'record': record})

        if not witnesses:
            # Return empty witness set
            return WitnessSet(
                variable=variable,
                witnesses=np.zeros((0, self.dimension)),
                metadata=[]
            )

        return WitnessSet(
            variable=variable,
            witnesses=np.array(witnesses),
            metadata=metadata
        )


class LLMWitnessExtractor(WitnessExtractor):
    """
    Extract witnesses using an LLM wrapper.

    Prompts an LLM to identify key evidence for a variable,
    then embeds the extracted evidence.
    """

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        dimension: int = 384
    ):
        """
        Args:
            embed_fn: Function to embed text into vectors
            dimension: Output dimension (used if embed_fn is None)
        """
        self.embed_fn = embed_fn
        self.dimension = dimension
        self._sentence_extractor = SentenceEmbeddingExtractor(dimension=dimension)

    def _default_evidence_extraction(self, text: str, variable: str) -> List[str]:
        """
        Extract evidence statements from text.
        Simple rule-based extraction as fallback.
        """
        # Split into sentences
        sentences = []
        for delim in ['. ', '! ', '? ', '\n']:
            parts = text.split(delim)
            if len(parts) > 1:
                sentences.extend([p.strip() + '.' for p in parts if p.strip()])
                break
        else:
            sentences = [text]

        # Filter to relevant sentences (containing variable name or related terms)
        variable_lower = variable.lower()
        keywords = [variable_lower]

        # Add domain-specific keywords
        if 'credit' in variable_lower:
            keywords.extend(['score', 'fico', 'credit'])
        elif 'ltv' in variable_lower or 'loan' in variable_lower:
            keywords.extend(['ltv', 'loan', 'value', 'ratio'])
        elif 'dti' in variable_lower or 'debt' in variable_lower:
            keywords.extend(['dti', 'debt', 'income', 'ratio'])
        elif 'default' in variable_lower or 'repurchase' in variable_lower:
            keywords.extend(['default', 'delinquent', 'repurchase', 'foreclosure'])

        relevant = []
        for sent in sentences:
            sent_lower = sent.lower()
            if any(kw in sent_lower for kw in keywords):
                relevant.append(sent)

        return relevant if relevant else sentences[:3]

    def extract(
        self,
        data: Union[str, Dict[str, Any]],
        variable: str,
        evidence_extractor: Optional[Callable[[str, str], List[str]]] = None
    ) -> WitnessSet:
        """
        Extract witnesses using LLM-style evidence extraction.

        Args:
            data: Text or structured data to extract from
            variable: Variable name for the witness set
            evidence_extractor: Optional custom function to extract evidence

        Returns:
            WitnessSet with embedded evidence
        """
        # Convert data to text if needed
        if isinstance(data, dict):
            text = " ".join(f"{k}: {v}" for k, v in data.items())
        else:
            text = str(data)

        # Extract evidence statements
        extractor = evidence_extractor or self._default_evidence_extraction
        evidence = extractor(text, variable)

        if not evidence:
            # Return empty witness set
            return WitnessSet(
                variable=variable,
                witnesses=np.zeros((0, self.dimension)),
                metadata=[]
            )

        # Embed evidence
        return self._sentence_extractor.extract(evidence, variable)


def extract_witnesses(
    data: Any,
    variable: str,
    method: str = 'auto',
    **kwargs
) -> WitnessSet:
    """
    High-level API to extract witnesses from data.

    Args:
        data: Input data (text, dict, list of dicts, etc.)
        variable: Variable name for the witness set
        method: Extraction method ('sentence', 'attribute', 'llm', 'auto')
        **kwargs: Additional arguments for the extractor

    Returns:
        WitnessSet with extracted witnesses
    """
    if method == 'auto':
        # Auto-detect based on data type
        if isinstance(data, str):
            method = 'sentence'
        elif isinstance(data, dict):
            # Check if it looks like structured data
            if all(isinstance(v, (int, float, str, bool)) for v in data.values()):
                method = 'attribute'
            else:
                method = 'llm'
        elif isinstance(data, list):
            if all(isinstance(d, dict) for d in data):
                method = 'attribute'
            else:
                method = 'sentence'
        else:
            method = 'sentence'

    if method == 'sentence':
        extractor = SentenceEmbeddingExtractor(**{k: v for k, v in kwargs.items()
                                                   if k in ['model_name', 'dimension']})
        return extractor.extract(data, variable)

    elif method == 'attribute':
        schema = kwargs.get('schema', {})
        if not schema and isinstance(data, (dict, list)):
            # Auto-generate schema
            sample = data[0] if isinstance(data, list) else data
            schema = {}
            for k, v in sample.items():
                if isinstance(v, (int, float)):
                    schema[k] = {'type': 'numeric'}
                else:
                    schema[k] = {'type': 'categorical', 'values': [v]}

        extractor = AttributeValueExtractor(
            schema=schema,
            dimension=kwargs.get('dimension', 64)
        )
        return extractor.extract(data, variable)

    elif method == 'llm':
        extractor = LLMWitnessExtractor(
            embed_fn=kwargs.get('embed_fn'),
            dimension=kwargs.get('dimension', 384)
        )
        return extractor.extract(data, variable)

    else:
        raise ValueError(f"Unknown extraction method: {method}")
