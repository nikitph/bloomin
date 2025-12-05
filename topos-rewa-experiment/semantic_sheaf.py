"""
Semantic Sheaf over Witness Manifold
"""

import numpy as np
from config import CONFIG
from utils import kl_divergence


class SemanticSheaf:
    """
    Sheaf of propositions over witness manifold
    Maps geometric neighborhoods to logical propositions
    """
    
    def __init__(self, manifold, dataset):
        """
        Initialize semantic sheaf
        
        Args:
            manifold: WitnessManifold instance
            dataset: CLEVRLiteDataset instance
        """
        self.manifold = manifold
        self.dataset = dataset
        self.local_propositions = {}  # Map: Region_ID -> {Predicates}
    
    def define_open_set(self, center_concept, radius=None):
        """
        Define open set U = {x | d_F(x, center) < r}
        Returns indices of items in this semantic neighborhood
        
        Args:
            center_concept: Center embedding vector
            radius: Neighborhood radius (uses CONFIG if None)
        
        Returns:
            List of indices in the open set
        """
        if radius is None:
            radius = CONFIG["NEIGHBORHOOD_RADIUS"]
        
        # Get witness distribution for center
        p_center = self.manifold.get_distribution(center_concept)
        
        # Find all points within Fisher distance radius
        neighbors = []
        for i, x in enumerate(self.dataset.data):
            p_x = self.manifold.get_distribution(x)
            d_fisher = self.manifold.fisher_distance(p_center, p_x)
            
            if d_fisher < radius:
                neighbors.append(i)
        
        return neighbors
    
    def assign_local_truth(self, U, predicate_function):
        """
        Assign local truth value to a predicate over open set U
        v_U(P) = E[P(x) | x in U]
        
        Args:
            U: List of indices in open set
            predicate_function: Function that evaluates predicate on data point
        
        Returns:
            Mean truth value over the neighborhood
        """
        if len(U) == 0:
            return 0.0
        
        truth_values = [predicate_function(self.dataset.data[i]) for i in U]
        return np.mean(truth_values)
    
    def glue_open_sets(self, U1, U2):
        """
        Gluing operation: intersection of open sets
        The global section exists only where local sections agree
        
        Args:
            U1: First open set (list of indices)
            U2: Second open set (list of indices)
        
        Returns:
            Intersection of open sets
        """
        return list(set(U1) & set(U2))
    
    def verify_consistency(self, x_index, prototype_distributions, threshold=None):
        """
        Verify consistency via restriction maps
        Check if item satisfies local logic constraints
        
        Args:
            x_index: Index of data point
            prototype_distributions: List of prototype distributions to check against
            threshold: Consistency threshold (uses CONFIG if None)
        
        Returns:
            True if consistent, False otherwise
        """
        if threshold is None:
            threshold = CONFIG["CONSISTENCY_THRESHOLD"]
        
        x = self.dataset.data[x_index]
        p_x = self.manifold.get_distribution(x)
        
        # Check KL divergence with each prototype
        for p_proto in prototype_distributions:
            kl = kl_divergence(p_x, p_proto)
            if kl > threshold:
                return False
        
        return True
    
    def compose_concepts(self, concept1_embedding, concept2_embedding):
        """
        Compose two concepts via sheaf gluing
        
        Args:
            concept1_embedding: Embedding for first concept
            concept2_embedding: Embedding for second concept
        
        Returns:
            List of indices satisfying both concepts
        """
        # Define open sets for each concept
        U1 = self.define_open_set(concept1_embedding)
        U2 = self.define_open_set(concept2_embedding)
        
        # Glue (intersect)
        U_glued = self.glue_open_sets(U1, U2)
        
        # Verify consistency
        p1 = self.manifold.get_distribution(concept1_embedding)
        p2 = self.manifold.get_distribution(concept2_embedding)
        
        consistent_items = []
        for idx in U_glued:
            if self.verify_consistency(idx, [p1, p2]):
                consistent_items.append(idx)
        
        return consistent_items
