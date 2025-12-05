"""
REWAMemory Module

Implements witness extraction, retrieval, Fisher metric computation,
and manifold prediction for the thermodynamic self-awareness framework.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from scipy.spatial.distance import cosine
from scipy.stats import entropy


@dataclass
class WitnessSet:
    """Container for witness representations"""
    item_id: str
    witnesses: np.ndarray  # Shape: (K, L) binary witness matrix
    embedding: Optional[np.ndarray] = None  # Optional dense embedding
    metadata: Optional[Dict] = None
    

@dataclass
class AbstractionPacket:
    """Container for abstracted concepts from RG"""
    scale: int
    prototype: np.ndarray
    support_set: Set[str]  # Item IDs in this abstraction
    statistics: Dict


class REWAMemory:
    """
    REWA Memory System with witness-based retrieval and geometric operations.
    
    Implements:
    - Witness extraction from inputs
    - K-nearest neighbor retrieval via witness overlap
    - Fisher information metric computation
    - Manifold state prediction
    - Abstraction storage and hierarchical organization
    """
    
    def __init__(
        self,
        K: int = 32,  # Number of hash functions
        L: int = 64,  # Witness dimension
        metric_epsilon: float = 1e-6,
        seed: int = 42
    ):
        self.K = K
        self.L = L
        self.metric_epsilon = metric_epsilon
        self.rng = np.random.RandomState(seed)
        
        # Storage
        self.memory: Dict[str, WitnessSet] = {}
        self.abstractions: Dict[int, List[AbstractionPacket]] = {}  # scale -> packets
        
        # Geometry
        self.fisher_metric_cache: Optional[np.ndarray] = None
        self.curvature_cache: Dict[str, float] = {}

        
        # Hash functions (random projections for witness extraction)
        self.hash_functions = self._initialize_hash_functions()
        
    def _initialize_hash_functions(self) -> List[np.ndarray]:
        """Initialize K random projection matrices for witness extraction"""
        return [self.rng.randn(self.L, self.L) for _ in range(self.K)]
    
    def extract_witnesses(self, input_data: np.ndarray, item_id: str = None) -> WitnessSet:
        """
        Extract binary witness representation from input.
        
        Args:
            input_data: Input vector or feature representation
            item_id: Optional identifier for this item
            
        Returns:
            WitnessSet containing binary witnesses
        """
        if item_id is None:
            item_id = f"item_{len(self.memory)}"
            
        # Project input through hash functions and threshold
        witnesses = np.zeros((self.K, self.L), dtype=np.int8)
        
        for k in range(self.K):
            projection = self.hash_functions[k] @ input_data[:self.L]
            witnesses[k] = (projection > 0).astype(np.int8)
        
        witness_set = WitnessSet(
            item_id=item_id,
            witnesses=witnesses,
            embedding=input_data,
            metadata={}
        )
        
        return witness_set
    
    def store(self, witness_set: WitnessSet):
        """Store a witness set in memory"""
        self.memory[witness_set.item_id] = witness_set
    
    def retrieve(
        self,
        query_witnesses: WitnessSet,
        k: int = 10,
        return_stats: bool = True
    ) -> List[Tuple[str, Dict]]:
        """
        Retrieve k nearest neighbors based on witness overlap.
        
        Args:
            query_witnesses: Query witness set
            k: Number of neighbors to retrieve
            return_stats: Whether to return overlap statistics
            
        Returns:
            List of (item_id, stats_dict) tuples
        """
        if len(self.memory) == 0:
            return []
        
        overlaps = []
        
        for item_id, stored_ws in self.memory.items():
            # Compute witness overlap (Hamming similarity)
            overlap = np.mean(query_witnesses.witnesses == stored_ws.witnesses)
            
            stats = {
                'overlap': overlap,
                'hamming_distance': 1.0 - overlap
            }
            
            # Add embedding distance if available
            if query_witnesses.embedding is not None and stored_ws.embedding is not None:
                emb_dist = cosine(query_witnesses.embedding, stored_ws.embedding)
                stats['embedding_distance'] = emb_dist
            
            overlaps.append((item_id, stats))
        
        # Sort by overlap (descending)
        overlaps.sort(key=lambda x: x[1]['overlap'], reverse=True)
        
        return overlaps[:k]
    
    def fisher_metric(self, region_ids: Optional[List[str]] = None) -> np.ndarray:
        """
        Compute Fisher information metric for a region of the manifold.
        
        Args:
            region_ids: Item IDs defining the region. If None, use all items.
            
        Returns:
            Fisher metric tensor (approximation via covariance)
        """
        if region_ids is None:
            region_ids = list(self.memory.keys())
        
        if len(region_ids) == 0:
            return np.eye(self.L) * self.metric_epsilon
        
        # Gather embeddings
        embeddings = []
        for item_id in region_ids:
            if item_id in self.memory and self.memory[item_id].embedding is not None:
                embeddings.append(self.memory[item_id].embedding)
        
        if len(embeddings) == 0:
            return np.eye(self.L) * self.metric_epsilon
        
        embeddings = np.array(embeddings)
        
        # Fisher metric approximation: covariance of embeddings
        # (In practice, this would be the Hessian of log-likelihood)
        mean = np.mean(embeddings, axis=0)
        centered = embeddings - mean
        
        # Ensure we don't exceed dimension
        dim = min(centered.shape[1], self.L)
        cov = (centered[:, :dim].T @ centered[:, :dim]) / len(embeddings)
        
        # Regularize
        fisher = cov + np.eye(dim) * self.metric_epsilon
        
        # Pad if necessary
        if dim < self.L:
            full_fisher = np.eye(self.L) * self.metric_epsilon
            full_fisher[:dim, :dim] = fisher
            fisher = full_fisher
        
        self.fisher_metric_cache = fisher
        return fisher
    
    def predict_next_state(self, current_witnesses: WitnessSet, horizon: int = 1) -> WitnessSet:
        """
        Predict next manifold state based on current witnesses.
        
        This is a simple momentum-based predictor; in practice would use
        learned dynamics or gradient flow.
        
        Args:
            current_witnesses: Current state
            horizon: Prediction horizon
            
        Returns:
            Predicted witness set
        """
        # Simple prediction: retrieve nearest neighbors and average
        neighbors = self.retrieve(current_witnesses, k=5, return_stats=False)
        
        if len(neighbors) == 0:
            return current_witnesses
        
        # Average neighbor embeddings
        neighbor_embeddings = []
        for item_id, _ in neighbors:
            if self.memory[item_id].embedding is not None:
                neighbor_embeddings.append(self.memory[item_id].embedding)
        
        if len(neighbor_embeddings) == 0:
            return current_witnesses
        
        predicted_embedding = np.mean(neighbor_embeddings, axis=0)
        
        # Extract witnesses from prediction
        predicted_ws = self.extract_witnesses(
            predicted_embedding,
            item_id=f"{current_witnesses.item_id}_pred_t{horizon}"
        )
        
        return predicted_ws
    
    def store_abstraction(self, packet: AbstractionPacket):
        """Store an abstraction packet from Semantic RG"""
        if packet.scale not in self.abstractions:
            self.abstractions[packet.scale] = []
        self.abstractions[packet.scale].append(packet)
    
    def sample_from_manifold(self, n_samples: int = 1) -> List[WitnessSet]:
        """
        Sample witness sets from the manifold for dreaming.
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            List of sampled witness sets
        """
        if len(self.memory) == 0:
            return []
        
        # Sample random items from memory
        item_ids = list(self.memory.keys())
        sampled_ids = self.rng.choice(item_ids, size=min(n_samples, len(item_ids)), replace=False)
        
        return [self.memory[item_id] for item_id in sampled_ids]
    
    def compute_semantic_energy(self, region_ids: Optional[List[str]] = None) -> float:
        """
        Compute semantic energy E_t = mean embedding distortion.
        
        Args:
            region_ids: Items to compute over. If None, use all.
            
        Returns:
            Energy value
        """
        if region_ids is None:
            region_ids = list(self.memory.keys())
        
        if len(region_ids) == 0:
            return 0.0
        
        # Compute mean distance to prototype (centroid)
        embeddings = []
        for item_id in region_ids:
            if item_id in self.memory and self.memory[item_id].embedding is not None:
                embeddings.append(self.memory[item_id].embedding)
        
        if len(embeddings) == 0:
            return 0.0
        
        embeddings = np.array(embeddings)
        centroid = np.mean(embeddings, axis=0)
        
        # Mean squared distance
        distances = np.linalg.norm(embeddings - centroid, axis=1)
        energy = np.mean(distances ** 2)
        
        return energy
    
    def get_statistics(self) -> Dict:
        """Get memory statistics"""
        return {
            'num_items': len(self.memory),
            'num_abstractions': sum(len(packets) for packets in self.abstractions.values()),
            'max_scale': max(self.abstractions.keys()) if self.abstractions else 0,
            'has_fisher_metric': self.fisher_metric_cache is not None
        }
