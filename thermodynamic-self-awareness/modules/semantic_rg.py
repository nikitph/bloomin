"""
Semantic Renormalization Group Module

Implements coarse-graining and consolidation of semantic concepts
across scales.
"""

import numpy as np
from typing import List, Dict, Set, Optional
from dataclasses import dataclass


@dataclass
class RGPacket:
    """Renormalized concept packet"""
    scale: int
    prototype: np.ndarray
    support_ids: Set[str]
    statistics: Dict
    parent_scale: Optional[int] = None


class SemanticRG:
    """
    Semantic Renormalization Group for hierarchical abstraction.
    
    Implements:
    - Coarse-graining of witness representations across scales
    - Consolidation decision logic
    - Scale-dependent prototype extraction
    """
    
    def __init__(
        self,
        consolidation_threshold: int = 10,
        similarity_threshold: float = 0.7
    ):
        self.consolidation_threshold = consolidation_threshold
        self.similarity_threshold = similarity_threshold
        self.scale_hierarchy: Dict[int, List[RGPacket]] = {}
        
    def coarse_grain(
        self,
        witness_sets: List,
        current_scale: int,
        target_scale: int
    ) -> List[RGPacket]:
        """
        Coarse-grain witness sets to target scale.
        
        Args:
            witness_sets: List of WitnessSet objects
            current_scale: Current resolution scale
            target_scale: Target coarser scale
            
        Returns:
            List of RGPacket objects at target scale
        """
        if len(witness_sets) == 0:
            return []
        
        # Cluster similar witnesses
        clusters = self._cluster_witnesses(witness_sets)
        
        packets = []
        for cluster_ids in clusters:
            # Extract prototype (mean of cluster)
            cluster_witnesses = [ws for ws in witness_sets if ws.item_id in cluster_ids]
            
            if len(cluster_witnesses) == 0:
                continue
            
            # Compute prototype
            embeddings = [ws.embedding for ws in cluster_witnesses if ws.embedding is not None]
            if len(embeddings) == 0:
                continue
            
            prototype = np.mean(embeddings, axis=0)
            
            # Create packet
            packet = RGPacket(
                scale=target_scale,
                prototype=prototype,
                support_ids=cluster_ids,
                statistics={
                    'cluster_size': len(cluster_ids),
                    'variance': np.var(embeddings, axis=0).mean()
                },
                parent_scale=current_scale
            )
            
            packets.append(packet)
        
        # Store in hierarchy
        if target_scale not in self.scale_hierarchy:
            self.scale_hierarchy[target_scale] = []
        self.scale_hierarchy[target_scale].extend(packets)
        
        return packets
    
    def _cluster_witnesses(self, witness_sets: List) -> List[Set[str]]:
        """
        Cluster witness sets by similarity.
        
        Args:
            witness_sets: List of WitnessSet objects
            
        Returns:
            List of clusters (sets of item IDs)
        """
        if len(witness_sets) == 0:
            return []
        
        # Simple agglomerative clustering
        clusters = [{ws.item_id} for ws in witness_sets]
        embeddings = {ws.item_id: ws.embedding for ws in witness_sets if ws.embedding is not None}
        
        # Merge similar clusters
        merged = True
        while merged and len(clusters) > 1:
            merged = False
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # Compute cluster similarity (mean pairwise)
                    sim = self._cluster_similarity(clusters[i], clusters[j], embeddings)
                    
                    if sim > self.similarity_threshold:
                        # Merge clusters
                        clusters[i] = clusters[i].union(clusters[j])
                        clusters.pop(j)
                        merged = True
                        break
                
                if merged:
                    break
        
        return clusters
    
    def _cluster_similarity(
        self,
        cluster1: Set[str],
        cluster2: Set[str],
        embeddings: Dict[str, np.ndarray]
    ) -> float:
        """Compute similarity between two clusters"""
        sims = []
        
        for id1 in cluster1:
            for id2 in cluster2:
                if id1 in embeddings and id2 in embeddings:
                    # Cosine similarity
                    emb1 = embeddings[id1]
                    emb2 = embeddings[id2]
                    
                    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-10)
                    sims.append(sim)
        
        return np.mean(sims) if sims else 0.0
    
    def should_consolidate(self, recent_witnesses: List) -> bool:
        """
        Decide whether to consolidate based on statistics.
        
        Args:
            recent_witnesses: Recent witness sets
            
        Returns:
            True if consolidation should occur
        """
        # Consolidate if we have enough recent witnesses
        return len(recent_witnesses) >= self.consolidation_threshold
    
    def get_statistics(self) -> Dict:
        """Get RG statistics"""
        return {
            'num_scales': len(self.scale_hierarchy),
            'total_packets': sum(len(packets) for packets in self.scale_hierarchy.values()),
            'max_scale': max(self.scale_hierarchy.keys()) if self.scale_hierarchy else 0
        }
