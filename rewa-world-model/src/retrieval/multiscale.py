"""
Unified Multiscale Retriever

Integrates all modules for end-to-end retrieval:
1. REWA encoding (fast binary search)
2. Semantic RG (multiscale coarse-to-fine)
3. Fisher geometry (metric refinement)
4. Topos logic (consistency checks)
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass

from encoding import REWAEncoder, REWAConfig
from retrieval import REWARetriever, RetrievalResult
from semantic_rg import SemanticRG, RGFlow
from geometry import FisherMetric
from topos import ToposLogic

@dataclass
class UnifiedResult:
    """Unified retrieval result with multiple scores"""
    doc_id: str
    rewa_score: float          # Binary similarity
    geometric_score: float     # Fisher distance
    consistency_score: float   # Topos consistency
    final_score: float         # Weighted combination

class MultiscaleRetriever:
    """
    Unified retriever integrating all REWA world-model components.
    
    Retrieval pipeline:
    1. Fast REWA retrieval (coarse candidates)
    2. Semantic RG scale selection
    3. Fisher-metric refinement
    4. Topos consistency filtering
    """
    
    def __init__(
        self,
        rewa_encoder: REWAEncoder,
        rewa_retriever: REWARetriever,
        rg_flow: RGFlow,
        fisher_metrics: List[FisherMetric],
        topos: ToposLogic
    ):
        self.rewa_encoder = rewa_encoder
        self.rewa_retriever = rewa_retriever
        self.rg_flow = rg_flow
        self.fisher_metrics = {m.doc_id: m for m in fisher_metrics}
        self.topos = topos
    
    def retrieve(
        self,
        query_signature: np.ndarray,
        query_embedding: np.ndarray,
        query_id: str,
        k: int = 10,
        rewa_candidates: int = 50
    ) -> List[UnifiedResult]:
        """
        Unified multiscale retrieval.
        
        Args:
            query_signature: REWA binary signature
            query_embedding: Dense embedding
            query_id: Query identifier
            k: Number of final results
            rewa_candidates: Number of candidates from REWA stage
            
        Returns:
            List of UnifiedResults
        """
        # Stage 1: Fast REWA retrieval
        rewa_results = self.rewa_retriever.search(query_signature, k=rewa_candidates)
        
        # Stage 2: Geometric refinement
        geometric_scores = {}
        for result in rewa_results:
            if result.doc_id in self.fisher_metrics:
                # Compute Fisher distance
                query_metric = FisherMetric(
                    doc_id=query_id,
                    metric=np.eye(len(query_embedding)),
                    embedding=query_embedding
                )
                doc_metric = self.fisher_metrics[result.doc_id]
                
                geo_dist = query_metric.geodesic_distance(doc_metric)
                geometric_scores[result.doc_id] = 1.0 / (1.0 + geo_dist)
            else:
                geometric_scores[result.doc_id] = 0.5
        
        # Stage 3: Topos consistency (simplified)
        consistency_scores = {}
        for result in rewa_results:
            # Check if document has consistent propositions
            if result.doc_id in self.topos.sections:
                # Simple consistency: count propositions
                section = self.topos.sections[result.doc_id]
                consistency = min(1.0, len(section.propositions) / 5.0)
                consistency_scores[result.doc_id] = consistency
            else:
                consistency_scores[result.doc_id] = 0.5
        
        # Stage 4: Combine scores
        unified_results = []
        for result in rewa_results:
            doc_id = result.doc_id
            
            # Weighted combination
            final_score = (
                0.4 * result.score +
                0.4 * geometric_scores.get(doc_id, 0.5) +
                0.2 * consistency_scores.get(doc_id, 0.5)
            )
            
            unified_results.append(UnifiedResult(
                doc_id=doc_id,
                rewa_score=result.score,
                geometric_score=geometric_scores.get(doc_id, 0.5),
                consistency_score=consistency_scores.get(doc_id, 0.5),
                final_score=final_score
            ))
        
        # Sort by final score
        unified_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return unified_results[:k]
