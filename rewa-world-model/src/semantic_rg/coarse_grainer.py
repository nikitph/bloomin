"""
Semantic RG (Renormalization Group) Module

Implements multiscale coarse-graining of witness distributions:
1. Block witnesses at different scales
2. Compute renormalized metrics
3. Preserve mutual information across scales
4. Enable scale transfer and one-shot learning

Key concept: RG flow reveals intrinsic structure by iteratively coarse-graining.
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sklearn.cluster import AgglomerativeClustering

@dataclass
class RGScale:
    """Representation at a single RG scale"""
    scale: int                          # Scale index (0 = finest)
    witness_blocks: List[List[str]]     # Grouped witnesses
    coarse_distributions: List[Dict[str, float]]  # Coarse-grained distributions
    metrics: List[np.ndarray]           # Renormalized Fisher metrics
    compression_ratio: float            # Size reduction vs. previous scale

@dataclass
class RGFlow:
    """Complete RG flow across scales"""
    scales: List[RGScale]
    mutual_information: List[float]     # MI preservation per scale
    
class SemanticRG:
    """Semantic Renormalization Group coarse-grainer"""
    
    def __init__(self, num_scales: int = 3, block_size_base: int = 2):
        self.num_scales = num_scales
        self.block_size_base = block_size_base
    
    def cluster_witnesses(
        self,
        witness_ids: List[str],
        embeddings: np.ndarray,
        n_clusters: int
    ) -> List[List[int]]:
        """
        Cluster witnesses using hierarchical clustering on embeddings.
        
        Args:
            witness_ids: List of witness identifiers
            embeddings: Witness embeddings (n_witnesses, dim)
            n_clusters: Number of clusters
            
        Returns:
            List of cluster assignments (cluster_id -> [witness_indices])
        """
        if len(witness_ids) <= n_clusters:
            # Each witness is its own cluster
            return [[i] for i in range(len(witness_ids))]
        
        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        labels = clustering.fit_predict(embeddings)
        
        # Group by cluster
        clusters = [[] for _ in range(n_clusters)]
        for i, label in enumerate(labels):
            clusters[label].append(i)
        
        return clusters
    
    def coarse_grain_distribution(
        self,
        distribution: Dict[str, float],
        witness_blocks: List[List[str]]
    ) -> Dict[str, float]:
        """
        Coarse-grain a witness distribution by summing over blocks.
        
        Args:
            distribution: Original witness distribution
            witness_blocks: Grouping of witnesses into blocks
            
        Returns:
            Coarse-grained distribution over blocks
        """
        coarse_dist = {}
        
        for block_id, block in enumerate(witness_blocks):
            block_key = f"block_{block_id}"
            block_prob = 0.0
            
            for witness in block:
                block_prob += distribution.get(witness, 0.0)
            
            if block_prob > 0:
                coarse_dist[block_key] = block_prob
        
        # Normalize
        total = sum(coarse_dist.values())
        if total > 0:
            coarse_dist = {k: v/total for k, v in coarse_dist.items()}
        
        return coarse_dist
    
    def renormalize_metric(
        self,
        metric: np.ndarray,
        block_indices: List[List[int]]
    ) -> np.ndarray:
        """
        Renormalize Fisher metric by averaging over blocks.
        
        Args:
            metric: Original metric (d, d)
            block_indices: Grouping of dimensions into blocks
            
        Returns:
            Renormalized metric (n_blocks, n_blocks)
        """
        n_blocks = len(block_indices)
        renorm_metric = np.zeros((n_blocks, n_blocks))
        
        for i, block_i in enumerate(block_indices):
            for j, block_j in enumerate(block_indices):
                # Average metric over block
                if len(block_i) > 0 and len(block_j) > 0:
                    sub_metric = metric[np.ix_(block_i, block_j)]
                    renorm_metric[i, j] = np.mean(sub_metric)
        
        # Ensure positive-definite
        renorm_metric = (renorm_metric + renorm_metric.T) / 2
        renorm_metric += np.eye(n_blocks) * 1e-4
        
        return renorm_metric
    
    def compute_mutual_information(
        self,
        dist1: Dict[str, float],
        dist2: Dict[str, float]
    ) -> float:
        """
        Compute mutual information between two distributions.
        
        Simplified: I(X;Y) ≈ -Σ p(x) log(p(x)/q(x))
        """
        mi = 0.0
        
        for key in dist1:
            p = dist1[key]
            q = dist2.get(key, 1e-10)
            
            if p > 1e-10:
                mi += p * np.log(p / q)
        
        return max(0, mi)
    
    def build_rg_flow(
        self,
        witness_distributions: List[Dict[str, float]],
        metrics: List[np.ndarray],
        doc_ids: List[str]
    ) -> RGFlow:
        """
        Build complete RG flow across scales.
        
        Args:
            witness_distributions: Original witness distributions
            metrics: Original Fisher metrics
            doc_ids: Document IDs
            
        Returns:
            RGFlow object
        """
        scales = []
        current_dists = witness_distributions
        current_metrics = metrics
        
        # Extract unique witnesses
        all_witnesses = set()
        for dist in witness_distributions:
            all_witnesses.update(dist.keys())
        witness_list = list(all_witnesses)
        
        # Create simple embeddings for witnesses (hash-based)
        witness_embeddings = np.array([
            [hash(w) % 1000 / 1000.0 for _ in range(8)]
            for w in witness_list
        ])
        
        for scale_idx in range(self.num_scales):
            # Determine block size
            block_size = self.block_size_base ** (scale_idx + 1)
            n_blocks = max(1, len(witness_list) // block_size)
            
            print(f"  Scale {scale_idx}: {len(witness_list)} witnesses → {n_blocks} blocks")
            
            # Cluster witnesses
            if scale_idx == 0:
                # Finest scale: no clustering
                witness_blocks = [[w] for w in witness_list]
            else:
                # Cluster
                if len(witness_list) <= n_blocks:
                    # Not enough witnesses to cluster
                    witness_blocks = [[w] for w in witness_list]
                else:
                    clusters = self.cluster_witnesses(
                        witness_list,
                        witness_embeddings,
                        n_blocks
                    )
                    # Build blocks from cluster indices
                    witness_blocks = []
                    for cluster in clusters:
                        block = []
                        for idx in cluster:
                            if idx < len(witness_list):
                                block.append(witness_list[idx])
                        if block:  # Only add non-empty blocks
                            witness_blocks.append(block)
            
            # Coarse-grain distributions
            coarse_dists = [
                self.coarse_grain_distribution(dist, witness_blocks)
                for dist in current_dists
            ]
            
            # Renormalize metrics
            # (simplified: just subsample dimensions)
            if scale_idx > 0 and len(current_metrics) > 0:
                d = len(current_metrics[0])
                n_blocks_metric = max(1, d // block_size)
                block_indices = [
                    list(range(i * block_size, min((i+1) * block_size, d)))
                    for i in range(n_blocks_metric)
                ]
                
                renorm_metrics = [
                    self.renormalize_metric(m, block_indices)
                    for m in current_metrics
                ]
            else:
                renorm_metrics = current_metrics
            
            # Compute compression ratio
            if scale_idx > 0 and len(scales) > 0:
                prev_size = len(scales[-1].witness_blocks)
                curr_size = len(witness_blocks)
                compression = prev_size / curr_size if curr_size > 0 else 1.0
            else:
                compression = 1.0
            
            # Create scale
            scale = RGScale(
                scale=scale_idx,
                witness_blocks=witness_blocks,
                coarse_distributions=coarse_dists,
                metrics=renorm_metrics,
                compression_ratio=compression
            )
            scales.append(scale)
            
            # Update for next iteration
            current_dists = coarse_dists
            current_metrics = renorm_metrics
            witness_list = [f"block_{i}" for i in range(len(witness_blocks))]
        
        # Compute MI preservation
        mi_values = []
        for i in range(len(scales) - 1):
            # Average MI between consecutive scales
            avg_mi = 0.0
            for j in range(min(len(scales[i].coarse_distributions), 
                              len(scales[i+1].coarse_distributions))):
                mi = self.compute_mutual_information(
                    scales[i].coarse_distributions[j],
                    scales[i+1].coarse_distributions[j]
                )
                avg_mi += mi
            avg_mi /= len(scales[i].coarse_distributions)
            mi_values.append(avg_mi)
        
        return RGFlow(scales=scales, mutual_information=mi_values)
