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
        
        # Initialize with STRUCTURED prototypes (not uniform!)
        for concept in self.ontology:
            self._initialize_concept_structured(concept)
    
    def _peaked_distribution(self, peak_idx, sharpness=5.0, noise_width=3):
        """
        Create a peaked probability distribution
        
        Args:
            peak_idx: Index of the peak
            sharpness: How sharp the peak is (higher = more peaked)
            noise_width: Width of noise around peak
        
        Returns:
            Probability distribution with peak at peak_idx
        """
        logits = torch.zeros(self.n_witnesses)
        
        # High value at peak
        logits[peak_idx] = sharpness
        
        # Add noise to nearby dimensions
        start = max(0, peak_idx - noise_width)
        end = min(self.n_witnesses, peak_idx + noise_width + 1)
        logits[start:end] += torch.randn(end - start) * 0.5
        
        # Convert to probability distribution
        return torch.softmax(logits, dim=0)
    
    def _initialize_concept_structured(self, concept_name):
        """
        Initialize concept with MEANINGFUL structure
        
        Semantic subspaces:
        - Color concepts: dims 0-29 (Red=5, Blue=15, Green=25)
        - Shape concepts: dims 30-59 (Circle=35, Square=45, Triangle=55)
        - Texture concepts: dims 60-89 (Smooth=65, Rough=75)
        - Size concepts: dims 90-119 (Small=95, Large=105)
        """
        # Map concepts to semantic subspaces
        concept_peaks = {
            # Colors (dims 0-29)
            "Red": 5,
            "Blue": 15,
            "Green": 25,
            "Purple": 10,  # Between Red and Blue
            
            # Shapes (dims 30-59)
            "Circle": 35,
            "Square": 45,
            "Triangle": 55,
            
            # Abstract categories (broader distributions)
            "Color": None,  # Will be computed as centroid
            "Shape": None,
            "VisualAttribute": None,
        }
        
        if concept_name in concept_peaks and concept_peaks[concept_name] is not None:
            # Concrete concept - peaked distribution
            peak_idx = concept_peaks[concept_name]
            prototype = self._peaked_distribution(peak_idx, sharpness=8.0)
        else:
            # Unknown concept or abstract - use random peaked distribution
            peak_idx = torch.randint(0, self.n_witnesses, (1,)).item()
            prototype = self._peaked_distribution(peak_idx, sharpness=5.0)
        
        # Create embedding (inverse problem - just use random for now)
        embedding = torch.randn(self.dim_input)
        embedding = embedding / torch.norm(embedding)
        
        self.concept_embeddings[concept_name] = embedding
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
    
    def resolve_contradiction(self, concept_a, concept_b, new_concept_name=None):
        """
        Resolve contradiction between two concepts by inventing a new one
        This enables cascading creativity - using invented concepts as inputs
        
        Args:
            concept_a: First concept name
            concept_b: Second concept name
            new_concept_name: Optional name for invented concept
        
        Returns:
            (new_concept_name, free_energy_reduction, success)
        """
        import torch.optim as optim
        from ricci_flow import contrastive_loss, curvature_penalty
        
        # Get prototypes
        if concept_a not in self.concept_prototypes:
            raise ValueError(f"Unknown concept: {concept_a}")
        if concept_b not in self.concept_prototypes:
            raise ValueError(f"Unknown concept: {concept_b}")
        
        p_a = self.concept_prototypes[concept_a]
        p_b = self.concept_prototypes[concept_b]
        
        # Create dream state (superposition)
        p_dream = 0.5 * p_a + 0.5 * p_b
        
        # Add noise for creativity
        noise = torch.randn_like(p_dream) * CONFIG.get("DREAM_TEMP", 1.0) * 0.1
        p_dream = p_dream + noise
        
        # Normalize
        p_dream = torch.clamp(p_dream, min=0.0)
        p_dream = p_dream / torch.sum(p_dream)
        
        # Calculate initial Free Energy
        F_initial = self.calculate_free_energy(p_dream)
        
        # Check if invention is needed
        threshold = CONFIG.get("DISSONANCE_THRESHOLD", 0.5)
        
        if F_initial <= threshold:
            # No invention needed - concepts are compatible
            return None, 0.0, False
        
        # Invent new concept via Ricci Flow
        p_new = p_dream.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([p_new], lr=0.01)
        
        # Get all other prototypes for repulsion
        other_prototypes = [self.concept_prototypes[c] for c in self.ontology 
                           if c not in [concept_a, concept_b]]
        
        # Optimize
        for step in range(CONFIG.get("HEALING_STEPS", 200)):
            optimizer.zero_grad()
            
            p_normalized = torch.softmax(p_new, dim=0)
            
            # Contrastive loss: pull toward parents, push away from others
            loss = contrastive_loss(
                anchor=p_normalized,
                positive=p_normalized,
                negatives=other_prototypes[:min(5, len(other_prototypes))],  # Limit for efficiency
                temperature=0.1
            )
            
            # Also stay close to parents
            epsilon = 1e-10
            p_norm_clipped = torch.clamp(p_normalized, epsilon, 1.0)
            for p_parent in [p_a, p_b]:
                p_parent_clipped = torch.clamp(p_parent, epsilon, 1.0)
                kl = torch.sum(p_norm_clipped * torch.log(p_norm_clipped / p_parent_clipped))
                loss += kl * 0.1
            
            # Curvature regularization
            if len(other_prototypes) > 0:
                all_protos = [p_normalized] + other_prototypes[:3]
                loss += curvature_penalty(all_protos, alpha=0.01)
            
            loss.backward()
            optimizer.step()
        
        # Final prototype
        with torch.no_grad():
            p_final = torch.softmax(p_new, dim=0)
        
        # Generate name if not provided
        if new_concept_name is None:
            new_concept_name = self._generate_concept_name(concept_a, concept_b)
        
        # Register new concept
        self.register_prototype(new_concept_name, p_final)
        
        # Calculate final Free Energy
        F_final = self.calculate_free_energy(p_final)
        reduction = F_initial - F_final
        
        return new_concept_name, reduction, True
    
    def _generate_concept_name(self, parent_a, parent_b):
        """Generate name for invented concept based on parents"""
        # Known color mixtures
        known_mixtures = {
            ('Red', 'Blue'): 'Purple',
            ('Blue', 'Red'): 'Purple',
            ('Red', 'Yellow'): 'Orange',
            ('Yellow', 'Red'): 'Orange',
            ('Blue', 'Yellow'): 'Green',
            ('Yellow', 'Blue'): 'Green',
            ('Purple', 'Orange'): 'Mauve',
            ('Orange', 'Purple'): 'Mauve',
            ('Orange', 'Green'): 'Chartreuse',
            ('Green', 'Orange'): 'Chartreuse',
            ('Green', 'Purple'): 'Teal',
            ('Purple', 'Green'): 'Teal',
        }
        
        key = (parent_a, parent_b)
        if key in known_mixtures:
            return known_mixtures[key]
        else:
            # Generic name
            return f"Concept_{len(self.ontology)}"
    
    def find_similar_concepts(self, threshold=2.0, concrete_only=None):
        """
        Find pairs of concepts that are similar (low KL divergence)
        These are candidates for abstraction
        
        Args:
            threshold: Maximum KL divergence to consider similar
            concrete_only: List of concrete concept names to consider (if None, use all)
        
        Returns:
            List of (concept_a, concept_b, distance) tuples, sorted by distance
        """
        pairs = []
        
        # Use provided concrete concepts or all concepts
        if concrete_only is not None:
            concepts = [c for c in concrete_only if c in self.ontology]
        else:
            concepts = list(self.ontology)
        
        epsilon = 1e-10
        
        for i in range(len(concepts)):
            for j in range(i+1, len(concepts)):
                p_a = self.concept_prototypes[concepts[i]]
                p_b = self.concept_prototypes[concepts[j]]
                
                # Symmetric KL divergence
                p_a_clipped = torch.clamp(p_a, epsilon, 1.0)
                p_b_clipped = torch.clamp(p_b, epsilon, 1.0)
                
                kl_ab = torch.sum(p_a_clipped * torch.log(p_a_clipped / p_b_clipped))
                kl_ba = torch.sum(p_b_clipped * torch.log(p_b_clipped / p_a_clipped))
                
                dist = 0.5 * (kl_ab + kl_ba)
                
                if dist < threshold:
                    pairs.append((concepts[i], concepts[j], dist.item()))
        
        return sorted(pairs, key=lambda x: x[2])
    
    def ask_question(self, concept_a, concept_b):
        """
        Generate a question about two concepts
        
        Returns:
            Question string
        """
        return f"What do {concept_a} and {concept_b} have in common?"
    
    def discover_abstraction(self, concept_a, concept_b, abstraction_name=None):
        """
        Discover abstraction by finding commonality between compatible concepts
        Similar to resolve_contradiction but for creating superordinate categories
        
        Args:
            concept_a: First concept name
            concept_b: Second concept name
            abstraction_name: Optional name for abstraction
        
        Returns:
            (abstraction_name, free_energy_reduction, success)
        """
        import torch.optim as optim
        from ricci_flow import contrastive_loss
        
        # Get prototypes
        if concept_a not in self.concept_prototypes:
            raise ValueError(f"Unknown concept: {concept_a}")
        if concept_b not in self.concept_prototypes:
            raise ValueError(f"Unknown concept: {concept_b}")
        
        p_a = self.concept_prototypes[concept_a]
        p_b = self.concept_prototypes[concept_b]
        
        # Create abstraction as centroid (broader distribution)
        p_abstraction_init = (p_a + p_b) / 2.0
        p_abstraction_init = p_abstraction_init / torch.sum(p_abstraction_init)
        
        # Calculate initial Free Energy
        F_before = (self.calculate_free_energy(p_a) + self.calculate_free_energy(p_b)) / 2.0
        
        # Make it learnable
        p_abstraction = p_abstraction_init.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([p_abstraction], lr=0.02)
        
        # Get other concepts for repulsion
        other_prototypes = [self.concept_prototypes[c] for c in self.ontology 
                           if c not in [concept_a, concept_b]]
        
        # Optimize: stay close to both concepts, push away from unrelated ones
        for step in range(50):
            optimizer.zero_grad()
            
            p_normalized = torch.softmax(p_abstraction, dim=0)
            
            # Loss 1: Stay close to both concepts (attraction)
            epsilon = 1e-10
            p_norm_clipped = torch.clamp(p_normalized, epsilon, 1.0)
            
            attraction_loss = 0.0
            for p_concrete in [p_a, p_b]:
                p_concrete_clipped = torch.clamp(p_concrete, epsilon, 1.0)
                kl = torch.sum(p_norm_clipped * torch.log(p_norm_clipped / p_concrete_clipped))
                attraction_loss += kl
            
            attraction_loss = attraction_loss / 2.0
            
            # Loss 2: Push away from unrelated concepts (repulsion)
            if len(other_prototypes) > 0:
                repulsion_loss = -contrastive_loss(
                    anchor=p_normalized,
                    positive=p_normalized,
                    negatives=other_prototypes[:min(3, len(other_prototypes))],
                    temperature=0.1
                )
            else:
                repulsion_loss = 0.0
            
            # Total loss
            loss = attraction_loss + 0.3 * repulsion_loss
            
            loss.backward()
            optimizer.step()
        
        # Final abstraction
        with torch.no_grad():
            p_final = torch.softmax(p_abstraction, dim=0)
        
        # Generate name if not provided
        if abstraction_name is None:
            abstraction_name = self._generate_abstraction_name(concept_a, concept_b)
        
        # Register abstraction
        self.register_prototype(abstraction_name, p_final)
        
        # Calculate Free Energy reduction
        F_after = self.calculate_free_energy(p_final)
        reduction = F_before - F_after
        
        return abstraction_name, reduction, True
    
    def _generate_abstraction_name(self, concept_a, concept_b):
        """Generate name for abstraction based on constituent concepts"""
        # Known animal groupings
        known_abstractions = {
            ('Dog', 'Cat'): 'Pet',
            ('Cat', 'Dog'): 'Pet',
            ('Horse', 'Cow'): 'FarmAnimal',
            ('Cow', 'Horse'): 'FarmAnimal',
            ('Pet', 'FarmAnimal'): 'Mammal',
            ('FarmAnimal', 'Pet'): 'Mammal',
        }
        
        key = (concept_a, concept_b)
        if key in known_abstractions:
            return known_abstractions[key]
        else:
            # Generic name
            return f"Abstraction_{len(self.ontology)}"
    
    def self_directed_learning_step(self, similarity_threshold=2.0, concrete_concepts=None):
        """
        One step of self-directed learning
        
        1. Find similar concepts (only among concrete ones)
        2. Ask question about most similar pair
        3. Discover abstraction
        4. Update ontology
        
        Args:
            similarity_threshold: Maximum KL divergence to consider similar
            concrete_concepts: List of concrete concept names (excludes abstractions)
        
        Returns:
            (question, answer, abstraction_name, F_reduction, success)
        """
        # Find most similar pair (only among concrete concepts)
        pairs = self.find_similar_concepts(threshold=similarity_threshold, concrete_only=concrete_concepts)
        
        if not pairs:
            return None, "No more similar concepts found", None, 0.0, False
        
        concept_a, concept_b, dist = pairs[0]
        
        # Ask question
        question = self.ask_question(concept_a, concept_b)
        
        # Discover abstraction
        abstraction, F_reduction, success = self.discover_abstraction(concept_a, concept_b)
        
        if success:
            answer = f"Invented: {abstraction} (ΔF = {F_reduction:.4f})"
        else:
            answer = "No abstraction needed"
        
        return question, answer, abstraction, F_reduction, success
    
    def find_intersection(self, concept_a, concept_b):
        """
        Find the intersection (commonality) between two concepts.
        Implements Subtractive Logic via product of distributions.
        
        Args:
            concept_a, concept_b: Concept names
            
        Returns:
            (result_concept, distance_to_result, p_intersect)
        """
        if concept_a not in self.concept_prototypes:
            raise ValueError(f"Unknown concept: {concept_a}")
        if concept_b not in self.concept_prototypes:
            raise ValueError(f"Unknown concept: {concept_b}")
            
        p_a = self.concept_prototypes[concept_a]
        p_b = self.concept_prototypes[concept_b]
        
        # Intersection = Pointwise product (AND logic in probability space)
        # This finds the region where both distributions are high
        p_intersect = p_a * p_b
        
        # Check if intersection is non-trivial
        if torch.sum(p_intersect) < 1e-6:
            return None, 0.0, None
            
        # Renormalize to make it a valid probability distribution
        p_intersect = p_intersect / torch.sum(p_intersect)
        
        # Find nearest existing concept to this intersection
        nearest, dist = self.find_nearest_prototype(p_intersect)
        
        return nearest, dist, p_intersect

