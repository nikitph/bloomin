"""
Synthetic Dataset Generator

Creates test datasets for evaluating different aspects of the REWA world-model:
1. Hierarchical Gaussians (test hyperbolic vs Euclidean geometry)
2. CLEVR-style compositional data (test Topos reasoning)
3. Graph structures (test Tropical witnesses)
"""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class SyntheticDocument:
    """A synthetic document for testing"""
    id: str
    text: str
    metadata: Dict
    embedding: np.ndarray = None

class HierarchicalGaussianGenerator:
    """
    Generate hierarchical Gaussian clusters to test geometric properties.
    
    Tests whether the system can:
    - Detect hyperbolic vs Euclidean structure
    - Preserve hierarchical relationships
    - Estimate correct intrinsic dimension
    """
    
    def __init__(self, n_levels: int = 3, branching_factor: int = 3, 
                 dim: int = 128, seed: int = 42):
        self.n_levels = n_levels
        self.branching_factor = branching_factor
        self.dim = dim
        self.rng = np.random.RandomState(seed)
    
    def generate(self, docs_per_leaf: int = 10) -> List[SyntheticDocument]:
        """Generate hierarchical Gaussian dataset"""
        documents = []
        doc_id = 0
        
        # Build tree structure
        def build_tree(level: int, parent_center: np.ndarray, path: List[int]):
            nonlocal doc_id
            
            if level == self.n_levels:
                # Leaf node: generate documents
                for i in range(docs_per_leaf):
                    # Sample from Gaussian around parent center
                    embedding = parent_center + self.rng.randn(self.dim) * 0.1
                    embedding = embedding / np.linalg.norm(embedding)  # Normalize
                    
                    text = f"Document at path {path}, sample {i}"
                    metadata = {
                        'level': level,
                        'path': path.copy(),
                        'cluster_id': '_'.join(map(str, path))
                    }
                    
                    documents.append(SyntheticDocument(
                        id=f"doc_{doc_id}",
                        text=text,
                        metadata=metadata,
                        embedding=embedding
                    ))
                    doc_id += 1
            else:
                # Internal node: create children
                for i in range(self.branching_factor):
                    # Create child center with increasing separation at higher levels
                    separation = 2.0 ** (self.n_levels - level)
                    direction = self.rng.randn(self.dim)
                    direction = direction / np.linalg.norm(direction)
                    child_center = parent_center + direction * separation
                    
                    build_tree(level + 1, child_center, path + [i])
        
        # Start from root
        root_center = self.rng.randn(self.dim)
        root_center = root_center / np.linalg.norm(root_center)
        build_tree(0, root_center, [])
        
        return documents

class CompositionalQAGenerator:
    """
    Generate CLEVR-style compositional QA data to test Topos reasoning.
    
    Tests whether the system can:
    - Extract local propositions (color, shape, size)
    - Perform gluing across overlapping regions
    - Answer multi-hop compositional queries
    """
    
    def __init__(self, n_objects: int = 100, seed: int = 42):
        self.n_objects = n_objects
        self.rng = np.random.RandomState(seed)
        
        self.colors = ['red', 'blue', 'green', 'yellow', 'purple']
        self.shapes = ['cube', 'sphere', 'cylinder', 'cone']
        self.sizes = ['small', 'medium', 'large']
    
    def generate(self) -> Tuple[List[SyntheticDocument], List[Dict]]:
        """
        Generate compositional QA dataset
        
        Returns:
            documents: List of object descriptions
            qa_pairs: List of {question, answer, reasoning_type}
        """
        documents = []
        
        # Generate objects
        for i in range(self.n_objects):
            color = self.rng.choice(self.colors)
            shape = self.rng.choice(self.shapes)
            size = self.rng.choice(self.sizes)
            
            text = f"A {size} {color} {shape}"
            metadata = {
                'color': color,
                'shape': shape,
                'size': size,
                'object_id': i
            }
            
            documents.append(SyntheticDocument(
                id=f"obj_{i}",
                text=text,
                metadata=metadata
            ))
        
        # Generate QA pairs
        qa_pairs = []
        
        # Single-hop queries
        for _ in range(20):
            color = self.rng.choice(self.colors)
            matching = [d for d in documents if d.metadata['color'] == color]
            qa_pairs.append({
                'question': f"Find all {color} objects",
                'answer': [d.id for d in matching],
                'reasoning_type': 'single_attribute'
            })
        
        # Multi-hop queries (conjunction)
        for _ in range(20):
            color = self.rng.choice(self.colors)
            shape = self.rng.choice(self.shapes)
            matching = [d for d in documents 
                       if d.metadata['color'] == color and d.metadata['shape'] == shape]
            qa_pairs.append({
                'question': f"Find all {color} {shape}s",
                'answer': [d.id for d in matching],
                'reasoning_type': 'conjunction'
            })
        
        return documents, qa_pairs

class GraphDistanceGenerator:
    """
    Generate graph structures to test Tropical witnesses.
    
    Tests whether the system can:
    - Encode graph distances using min-plus algebra
    - Preserve shortest path relationships
    - Handle mixed-modality (graph + text)
    """
    
    def __init__(self, n_nodes: int = 50, edge_prob: float = 0.1, seed: int = 42):
        self.n_nodes = n_nodes
        self.edge_prob = edge_prob
        self.rng = np.random.RandomState(seed)
    
    def generate(self) -> Tuple[List[SyntheticDocument], np.ndarray]:
        """
        Generate graph dataset
        
        Returns:
            documents: List of node descriptions
            distances: Distance matrix (n_nodes x n_nodes)
        """
        import networkx as nx
        
        # Generate random graph
        G = nx.erdos_renyi_graph(self.n_nodes, self.edge_prob, seed=self.rng)
        
        # Compute all-pairs shortest paths
        distances = np.full((self.n_nodes, self.n_nodes), np.inf)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i == j:
                    distances[i, j] = 0
                elif G.has_edge(i, j):
                    distances[i, j] = 1
        
        # Floyd-Warshall
        for k in range(self.n_nodes):
            for i in range(self.n_nodes):
                for j in range(self.n_nodes):
                    distances[i, j] = min(distances[i, j], 
                                        distances[i, k] + distances[k, j])
        
        # Create documents
        documents = []
        for i in range(self.n_nodes):
            neighbors = list(G.neighbors(i))
            text = f"Node {i} connected to {len(neighbors)} nodes"
            metadata = {
                'node_id': i,
                'degree': len(neighbors),
                'neighbors': neighbors
            }
            
            documents.append(SyntheticDocument(
                id=f"node_{i}",
                text=text,
                metadata=metadata
            ))
        
        return documents, distances
