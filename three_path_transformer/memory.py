import torch
import numpy as np
import time
from collections import defaultdict

class ConceptMemory:
    """
    Hierarchical storage of learned concept embeddings.
    Simulates the "Long-Term Memory" or "Cerebral Cortex".
    """
    def __init__(self):
        self.concepts = {} # name -> embedding
        self.metadata = {} # name -> {generation, parents, created_at}
        self.hierarchy = defaultdict(list) # generation -> [names]
        
    def store(self, name, embedding, generation=0, parents=None):
        """
        Store a new concept.
        Args:
            name: Unique ID
            embedding: [dim] Tensor
            generation: Hierarchy depth (0=Primary)
            parents: List of parent names
        """
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu()
            
        self.concepts[name] = embedding
        self.metadata[name] = {
            'generation': generation,
            'parents': parents or [],
            'created_at': time.time()
        }
        # Avoid duplicates in hierarchy list
        if name not in self.hierarchy[generation]:
            self.hierarchy[generation].append(name)
            
    def retrieve(self, name):
        """Retrieve concept embedding"""
        if name not in self.concepts:
            raise KeyError(f"Concept '{name}' not found")
        return self.concepts[name]
        
    def get_all_embeddings(self):
        """Return dict of all embeddings (e.g. for sleep consolidation)"""
        return self.concepts
        
    def bulk_update(self, new_embeddings):
        """Update multiple concepts at once (after sleep)"""
        for name, emb in new_embeddings.items():
            if name in self.concepts:
                # Ensure tensor is detached/cpu
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu()
                self.concepts[name] = emb
                
    def get_hierarchy(self, depth):
        """Get all concepts at specific depth"""
        return self.hierarchy[depth]
        
    def get_parent(self, name):
        """Get the primary parent (first one)"""
        parents = self.metadata[name]['parents']
        if not parents:
            return None
        return self.concepts[parents[0]] # Return actual embedding? Or name? PRD says embedding.
        
    def measure_sharpness(self):
        """
        Measure average sharpness (Inverse Entropy).
        Returns dict with mean_entropy, std_entropy, sharpness.
        """
        entropies = []
        for emb in self.concepts.values():
            # emb should be probability dist
            # H = -sum(p * log(p))
            p = torch.clamp(emb, 1e-10, 1.0)
            H = -torch.sum(p * torch.log(p))
            entropies.append(H.item())
            
        if not entropies:
            return {'mean_entropy': 0, 'sharpness': 0}
            
        mean_H = np.mean(entropies)
        return {
            'mean_entropy': mean_H,
            'std_entropy': np.std(entropies),
            'sharpness': 1.0 / (mean_H + 1e-6)
        }
