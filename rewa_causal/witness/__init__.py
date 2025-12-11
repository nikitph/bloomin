"""
Witness Extraction Layer

Extracts witness sets from raw text, structured records, embeddings, or tabular rows.
Each witness is a unit vector on S^{d-1}.
"""

from .witness_set import WitnessSet
from .extractors import (
    extract_witnesses,
    normalize_witness,
    SentenceEmbeddingExtractor,
    AttributeValueExtractor,
    LLMWitnessExtractor
)

__all__ = [
    'WitnessSet',
    'extract_witnesses',
    'normalize_witness',
    'SentenceEmbeddingExtractor',
    'AttributeValueExtractor',
    'LLMWitnessExtractor'
]
