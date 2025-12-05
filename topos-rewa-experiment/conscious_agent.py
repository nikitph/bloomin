"""
Conscious Agent: Combines Memory, Logic, and Dynamics
"""

import torch
import numpy as np
from config import CONFIG
from neural_encoder import NeuralEncoder
from ricci_flow import entropy


class ConsciousAgent:
    """
    Agent with Memory (Manifold), Logic (Sheaf), and Dynamics (Flow)
    Can autonomously expand its ontology through concept invention
    """
    
    def __init__(self, initial_concepts=None):
        """
        Initialize conscious agent
        
        Args:
            initial_concepts: List of initial concept names (e.g., ["Red", "Blue"])
        """
        self.dim_input = CONFIG["DIM_INPUT"]
        self.n_witnesses = CONFIG["N_WITNESSES"]
        
        # Neural encoder (learnable)
        self.encoder = NeuralEncoder(
            self.dim_input,
            self.n_witnesses,
            hidden_dim=128
        )
        
        # Ontology: known concepts
        if initial_concepts is None:
            self.ontology = ["Red", "Blue"]
        else:
            self.ontology = initial_concepts
        
        # Memory: concept embeddings and their witness distributions
        self.concept_embeddings = {}
        self.concept_prototypes = {}  # Witness distributions
        
        # Initialize with random embeddings
        for concept in self.ontology:
            self._initialize_concept(concept)
    
    def _initialize_concept(self, concept_name):
        """Initialize a concept with random embedding"""
        # Create a random embedding
        embedding = torch.randn(self.dim_input)
        embedding = embedding / torch.norm(embedding)  # Normalize
        
        self.concept_embeddings[concept_name] = embedding
        
        # Get witness distribution
        with torch.no_grad():
            prototype = self.encoder(embedding)
        
        self.concept_prototypes[concept_name] = prototype
    
    def perceive(self, concept_name_or_embedding):
        """
        Get witness distribution for a concept or embedding
        
        Args:
            concept_name_or_embedding: Either a concept name (str) or embedding (tensor)
        
        Returns:
            Witness distribution
        """
        if isinstance(concept_name_or_embedding, str):
            # Look up concept
            if concept_name_or_embedding not in self.concept_embeddings:
                raise ValueError(f"Unknown concept: {concept_name_or_embedding}")
            embedding = self.concept_embeddings[concept_name_or_embedding]
        else:
            embedding = concept_name_or_embedding
        
        with torch.no_grad():
            return self.encoder(embedding)
    
    def calculate_free_energy(self, witness_dist):
        """
        Calculate Free Energy: F = E - TS
        E = Distance to nearest valid concept (Semantic Distortion)
        S = Entropy of the witness distribution
        
        Args:
            witness_dist: Witness distribution (n_witnesses,)
        
        Returns:
            Free energy value
        """
        # Find nearest concept
        nearest_concept, dist = self.find_nearest_prototype(witness_dist)
        
        # Calculate entropy
        S = entropy(witness_dist)
        
        # Free Energy
        T = CONFIG.get("MANIFOLD_TEMP", 0.1)
        F = dist - T * S
        
        return F.item()
    
    def find_nearest_prototype(self, witness_dist):
        """
        Find nearest concept prototype to given witness distribution
        
        Args:
            witness_dist: Witness distribution (n_witnesses,)
        
        Returns:
            (concept_name, distance)
        """
        min_dist = float('inf')
        nearest_concept = None
        
        epsilon = 1e-10
        p = torch.clamp(witness_dist, epsilon, 1.0)
        
        for concept_name, prototype in self.concept_prototypes.items():
            q = torch.clamp(prototype, epsilon, 1.0)
            # KL divergence
            kl = torch.sum(p * torch.log(p / q))
            
            if kl < min_dist:
                min_dist = kl
                nearest_concept = concept_name
        
        return nearest_concept, min_dist
    
    def register_prototype(self, concept_name, witness_dist, embedding=None):
        """
        Add new concept to ontology
        
        Args:
            concept_name: Name of new concept
            witness_dist: Witness distribution for this concept
            embedding: Optional embedding (if None, create from witness dist)
        """
        self.ontology.append(concept_name)
        
        if embedding is None:
            # Create embedding (inverse of encoder - just use random for now)
            embedding = torch.randn(self.dim_input)
            embedding = embedding / torch.norm(embedding)
        
        self.concept_embeddings[concept_name] = embedding
        self.concept_prototypes[concept_name] = witness_dist
    
    def get_concept_embedding(self, concept_name):
        """Get embedding for a concept"""
        if concept_name not in self.concept_embeddings:
            raise ValueError(f"Unknown concept: {concept_name}")
        return self.concept_embeddings[concept_name]
    
    def get_all_prototypes(self):
        """Get all concept prototypes as a list"""
        return [self.concept_prototypes[c] for c in self.ontology]
