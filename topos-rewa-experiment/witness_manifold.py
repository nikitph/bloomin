"""
Witness Manifold with Fisher Geometry
"""

import numpy as np
from sklearn.cluster import KMeans
from config import CONFIG
from utils import softmax, kl_divergence, euclidean_distance


class WitnessManifold:
    """
    Manifold structure over witness prototypes with Fisher-Rao geometry
    """
    
    def __init__(self, data):
        """
        Initialize witness manifold from data
        
        Args:
            data: Data points (N x D numpy array)
        """
        self.data = data
        self.n_witnesses = CONFIG["N_WITNESSES"]
        self.beta = CONFIG["FISHER_BETA"]
        
        # Learn witness prototypes via K-means
        print(f"Learning {self.n_witnesses} witness prototypes...")
        kmeans = KMeans(
            n_clusters=self.n_witnesses,
            random_state=CONFIG["RANDOM_SEED"],
            n_init=10
        )
        kmeans.fit(data)
        self.prototypes = kmeans.cluster_centers_
        print(f"Witness prototypes learned. Shape: {self.prototypes.shape}")
    
    def get_distribution(self, x):
        """
        Get probability distribution over witnesses for point x
        p_x(w) ~ exp(-dist(x, w) / beta)
        
        Args:
            x: Data point (D,) or (N x D)
        
        Returns:
            Probability distribution over witnesses (N_WITNESSES,) or (N x N_WITNESSES)
        """
        if x.ndim == 1:
            # Single point
            dists = euclidean_distance(x, self.prototypes)
            return softmax(dists, beta=self.beta)
        else:
            # Multiple points
            dists = euclidean_distance(x, self.prototypes)
            return np.array([softmax(d, beta=self.beta) for d in dists])
    
    def fisher_distance(self, p_x, p_y):
        """
        Approximation of Fisher-Rao distance
        d_F ≈ sqrt(2 * KL(p_x || p_y))
        
        Args:
            p_x: Probability distribution over witnesses
            p_y: Probability distribution over witnesses
        
        Returns:
            Fisher distance approximation
        """
        kl = kl_divergence(p_x, p_y)
        return np.sqrt(2 * kl)
    
    def get_prototype_distribution(self, concept_embedding):
        """
        Get witness distribution for a concept prototype
        
        Args:
            concept_embedding: Embedding vector for concept
        
        Returns:
            Probability distribution over witnesses
        """
        return self.get_distribution(concept_embedding)
