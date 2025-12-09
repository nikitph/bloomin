"""
Witness Extraction Module

Implements witness extraction for different REWA modes:
- Boolean: tokens, keywords, binary features
- Natural: counts, frequencies
- Real: embeddings, continuous scores
- Tropical: graph distances, min-plus algebra
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np

class WitnessType(Enum):
    BOOLEAN = "boolean"
    NATURAL = "natural"
    REAL = "real"
    TROPICAL = "tropical"

@dataclass
class Witness:
    """A witness observation from a document"""
    id: str
    feature: Any  # token, embedding, distance, etc.
    value: float  # weight or score
    witness_type: WitnessType
    metadata: Optional[Dict] = None

class WitnessExtractor:
    """Extract witnesses from documents"""
    
    def __init__(self, witness_types: List[WitnessType]):
        self.witness_types = witness_types
        
    def extract(self, document: Dict[str, Any]) -> List[Witness]:
        """Extract all configured witness types from a document"""
        witnesses = []
        
        if WitnessType.BOOLEAN in self.witness_types:
            witnesses.extend(self._extract_boolean(document))
        if WitnessType.NATURAL in self.witness_types:
            witnesses.extend(self._extract_natural(document))
        if WitnessType.REAL in self.witness_types:
            witnesses.extend(self._extract_real(document))
        if WitnessType.TROPICAL in self.witness_types:
            witnesses.extend(self._extract_tropical(document))
            
        return witnesses
    
    def _extract_boolean(self, document: Dict) -> List[Witness]:
        """Extract Boolean witnesses (tokens, keywords)"""
        witnesses = []
        text = document.get('text', '')
        doc_id = document['id']
        
        # Simple tokenization
        tokens = text.lower().split()
        unique_tokens = set(tokens)
        
        for token in unique_tokens:
            witnesses.append(Witness(
                id=f"{doc_id}_{token}",
                feature=token,
                value=1.0,  # Boolean: present or not
                witness_type=WitnessType.BOOLEAN
            ))
        
        return witnesses
    
    def _extract_natural(self, document: Dict) -> List[Witness]:
        """Extract Natural witnesses (counts, frequencies)"""
        witnesses = []
        text = document.get('text', '')
        doc_id = document['id']
        
        # Token counts
        tokens = text.lower().split()
        token_counts = {}
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        for token, count in token_counts.items():
            witnesses.append(Witness(
                id=f"{doc_id}_{token}_count",
                feature=token,
                value=float(count),
                witness_type=WitnessType.NATURAL
            ))
        
        return witnesses
    
    def _extract_real(self, document: Dict) -> List[Witness]:
        """Extract Real witnesses (embeddings, scores)"""
        witnesses = []
        
        # Placeholder: would use actual embedding model
        # For now, create simple TF-IDF-like scores
        text = document.get('text', '')
        doc_id = document['id']
        tokens = text.lower().split()
        
        if len(tokens) > 0:
            # Simple normalized term frequency
            token_counts = {}
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
            
            max_count = max(token_counts.values())
            for token, count in token_counts.items():
                normalized_score = count / max_count
                witnesses.append(Witness(
                    id=f"{doc_id}_{token}_real",
                    feature=token,
                    value=normalized_score,
                    witness_type=WitnessType.REAL
                ))
        
        return witnesses
    
    def _extract_tropical(self, document: Dict) -> List[Witness]:
        """Extract Tropical witnesses (graph distances, min-plus)"""
        witnesses = []
        
        # Placeholder: would compute actual graph distances
        # For now, use position-based distances
        text = document.get('text', '')
        doc_id = document['id']
        tokens = text.lower().split()
        
        # Distance from start of document (tropical min-plus)
        for i, token in enumerate(tokens):
            witnesses.append(Witness(
                id=f"{doc_id}_{token}_dist_{i}",
                feature=token,
                value=float(i),  # Distance in tropical semiring
                witness_type=WitnessType.TROPICAL
            ))
        
        return witnesses

def estimate_witness_distribution(witnesses: List[Witness]) -> Dict[str, float]:
    """Compute normalized witness distribution p_x"""
    distribution = {}
    total = 0.0
    
    for w in witnesses:
        key = f"{w.feature}_{w.witness_type.value}"
        distribution[key] = distribution.get(key, 0.0) + w.value
        total += w.value
    
    # Normalize
    if total > 0:
        for key in distribution:
            distribution[key] /= total
    
    return distribution
