import numpy as np
from math import sqrt

def normalize(vec):
    """L2 normalization"""
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-12)

def euclidean_distance(v1, v2):
    """Euclidean distance"""
    return np.linalg.norm(v1 - v2)

def random_normal(size):
    """Generate random normal vector"""
    return np.random.randn(size)

class VectorDatabase:
    """
    Storage for raw vectors and metadata
    """
    
    def __init__(self):
        self.vectors = []      # List of numpy arrays
        self.metadata = []     # List of metadata dicts
        self.index_map = {}    # vector_id -> index
    
    def insert(self, vector, metadata=None):
        """Insert a vector"""
        idx = len(self.vectors)
        self.vectors.append(normalize(vector))  # Normalize
        self.metadata.append(metadata or {})
        
        # Optional: Store in index map
        if metadata and 'id' in metadata:
            self.index_map[metadata['id']] = idx
        
        return idx
    
    def get_vector(self, idx):
        """Retrieve vector by index"""
        return self.vectors[idx]
    
    def get_metadata(self, idx):
        """Retrieve metadata by index"""
        return self.metadata[idx]
    
    def get_nearest_to_point(self, point, k=1):
        """
        Brute force nearest neighbors (for refinement phase)
        """
        distances = [
            (i, euclidean_distance(point, vec))
            for i, vec in enumerate(self.vectors)
        ]
        distances.sort(key=lambda x: x[1])
        return distances[:k]
