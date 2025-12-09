"""
Topos Logic Module

Implements local propositions with gluing consistency:
1. Extract predicates from witness distributions
2. Build restriction maps for overlapping regions
3. Check gluing consistency
4. KL-projection for truth maintenance

Key concept: Local truth that glues consistently becomes global truth.
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Proposition:
    """A logical proposition about witnesses"""
    predicate: str              # e.g., "color=red"
    confidence: float           # Probability [0, 1]
    support: Set[str]           # Witness IDs supporting this

@dataclass
class LocalSection:
    """Local propositions over a region"""
    region_id: str
    witness_ids: Set[str]
    propositions: List[Proposition]

class ToposLogic:
    """Topos-theoretic local logic system"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.sections: Dict[str, LocalSection] = {}
    
    def extract_propositions(
        self,
        witness_dist: Dict[str, float],
        region_id: str
    ) -> List[Proposition]:
        """
        Extract propositions from witness distribution.
        
        For each witness, create a proposition if confidence > threshold.
        """
        propositions = []
        
        for witness, prob in witness_dist.items():
            if prob >= self.confidence_threshold:
                # Simple predicate: witness presence
                prop = Proposition(
                    predicate=f"has_{witness}",
                    confidence=prob,
                    support={witness}
                )
                propositions.append(prop)
        
        return propositions
    
    def build_section(
        self,
        region_id: str,
        witness_dist: Dict[str, float]
    ) -> LocalSection:
        """Build local section for a region"""
        witness_ids = set(witness_dist.keys())
        propositions = self.extract_propositions(witness_dist, region_id)
        
        section = LocalSection(
            region_id=region_id,
            witness_ids=witness_ids,
            propositions=propositions
        )
        
        self.sections[region_id] = section
        return section
    
    def compute_overlap(
        self,
        section1: LocalSection,
        section2: LocalSection
    ) -> Set[str]:
        """Compute witness overlap between two sections"""
        return section1.witness_ids & section2.witness_ids
    
    def restrict_proposition(
        self,
        prop: Proposition,
        overlap_witnesses: Set[str]
    ) -> Optional[Proposition]:
        """
        Restrict proposition to overlap region.
        
        Returns None if proposition has no support in overlap.
        """
        restricted_support = prop.support & overlap_witnesses
        
        if len(restricted_support) == 0:
            return None
        
        # Adjust confidence based on support overlap
        confidence_factor = len(restricted_support) / len(prop.support)
        
        return Proposition(
            predicate=prop.predicate,
            confidence=prop.confidence * confidence_factor,
            support=restricted_support
        )
    
    def check_gluing_consistency(
        self,
        section1: LocalSection,
        section2: LocalSection
    ) -> Tuple[bool, List[str]]:
        """
        Check if two sections agree on overlap.
        
        Returns:
            (is_consistent, list of inconsistent predicates)
        """
        overlap = self.compute_overlap(section1, section2)
        
        if len(overlap) == 0:
            return True, []  # No overlap, trivially consistent
        
        # Restrict propositions to overlap
        props1_restricted = {}
        for prop in section1.propositions:
            restricted = self.restrict_proposition(prop, overlap)
            if restricted:
                props1_restricted[restricted.predicate] = restricted
        
        props2_restricted = {}
        for prop in section2.propositions:
            restricted = self.restrict_proposition(prop, overlap)
            if restricted:
                props2_restricted[restricted.predicate] = restricted
        
        # Check consistency
        inconsistencies = []
        for pred in set(props1_restricted.keys()) & set(props2_restricted.keys()):
            conf1 = props1_restricted[pred].confidence
            conf2 = props2_restricted[pred].confidence
            
            # Inconsistent if confidences differ significantly
            if abs(conf1 - conf2) > 0.3:
                inconsistencies.append(pred)
        
        is_consistent = len(inconsistencies) == 0
        return is_consistent, inconsistencies
    
    def kl_projection(
        self,
        current_dist: Dict[str, float],
        constraint_witness: str,
        constraint_value: float
    ) -> Dict[str, float]:
        """
        KL-projection to incorporate new constraint.
        
        argmin_{q} KL(p || q) subject to q(witness) = value
        
        Simplified: Adjust distribution to match constraint.
        """
        new_dist = current_dist.copy()
        
        # Set constraint
        new_dist[constraint_witness] = constraint_value
        
        # Renormalize
        total = sum(new_dist.values())
        if total > 0:
            new_dist = {k: v/total for k, v in new_dist.items()}
        
        return new_dist
    
    def glue_sections(
        self,
        sections: List[LocalSection]
    ) -> Optional[List[Proposition]]:
        """
        Attempt to glue local sections into global propositions.
        
        Returns:
            Global propositions if consistent, None otherwise
        """
        if len(sections) == 0:
            return []
        
        # Check pairwise consistency
        for i in range(len(sections)):
            for j in range(i + 1, len(sections)):
                consistent, _ = self.check_gluing_consistency(sections[i], sections[j])
                if not consistent:
                    return None  # Cannot glue
        
        # Merge propositions
        global_props = {}
        for section in sections:
            for prop in section.propositions:
                if prop.predicate in global_props:
                    # Average confidence
                    existing = global_props[prop.predicate]
                    new_conf = (existing.confidence + prop.confidence) / 2
                    new_support = existing.support | prop.support
                    
                    global_props[prop.predicate] = Proposition(
                        predicate=prop.predicate,
                        confidence=new_conf,
                        support=new_support
                    )
                else:
                    global_props[prop.predicate] = prop
        
        return list(global_props.values())

class CompositionalQA:
    """Compositional question answering using Topos logic"""
    
    def __init__(self, topos: ToposLogic):
        self.topos = topos
    
    def answer_query(
        self,
        query: str,
        documents: List[Dict]
    ) -> List[str]:
        """
        Answer compositional query using local propositions.
        
        Args:
            query: Natural language query (e.g., "red cube")
            documents: List of documents with metadata
            
        Returns:
            List of matching document IDs
        """
        # Parse query into predicates (simplified)
        # Filter out common stopwords to focus on attributes
        stopwords = {'find', 'all', 'objects', 'object', 'retrieve', 'search', 'get'}
        all_tokens = query.lower().split()
        predicates = [t for t in all_tokens if t not in stopwords and not t.endswith('s')]
        
        # Handle plurals in query (e.g. "cubes" -> "cube")
        # Simple heuristic: if word ends in 's' and stems exists in metadata, use stem? 
        # For now, just strip 's' from query terms if they are likely plural object names
        predicates = []
        for t in all_tokens:
            if t in stopwords:
                continue
            if t.endswith('s') and len(t) > 3:
                predicates.append(t[:-1])
            else:
                predicates.append(t)
        
        # Find documents matching all predicates
        matching_docs = []
        
        for doc in documents:
            # Check if document has all required attributes
            matches = True
            for pred in predicates:
                # Check metadata (handle both dict and dataclass)
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                if metadata:
                    found = False
                    for key, value in metadata.items():
                        if pred in str(value).lower():
                            found = True
                            break
                    if not found:
                        matches = False
                        break
            
            if matches:
                doc_id = doc.id if hasattr(doc, 'id') else doc.get('id', '')
                matching_docs.append(doc_id)
        
        return matching_docs
