"""
CLEVR-lite data generation for compositional reasoning
"""

import numpy as np
from config import CONFIG


class CLEVRLiteDataset:
    """
    Synthetic dataset with compositional attributes (Color × Shape)
    Geometric embeddings where specific dimensions encode attributes
    """
    
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.colors = ["red", "blue", "green"]
        self.shapes = ["square", "circle", "triangle"]
        
        # Create attribute vectors
        self.color_vectors = self._create_attribute_vectors(
            self.colors,
            CONFIG["COLOR_DIM_START"],
            CONFIG["COLOR_DIM_END"]
        )
        
        self.shape_vectors = self._create_attribute_vectors(
            self.shapes,
            CONFIG["SHAPE_DIM_START"],
            CONFIG["SHAPE_DIM_END"]
        )
        
        self.data = []
        self.labels = []
        self._generate_data()
    
    def _create_attribute_vectors(self, attributes, dim_start, dim_end):
        """
        Create orthogonal vectors for attributes in specified dimension range
        """
        vectors = {}
        dim_size = dim_end - dim_start
        
        for i, attr in enumerate(attributes):
            vec = np.zeros(dim_size)
            # Create distinct patterns for each attribute
            vec[i * (dim_size // len(attributes))] = 1.0
            vectors[attr] = vec
        
        return vectors
    
    def _create_attribute_vector(self, color, shape):
        """
        Create base vector for a color-shape combination
        """
        vec = np.zeros(CONFIG["DIM_INPUT"])
        
        # Color dimensions
        vec[CONFIG["COLOR_DIM_START"]:CONFIG["COLOR_DIM_END"]] = self.color_vectors[color]
        
        # Shape dimensions
        vec[CONFIG["SHAPE_DIM_START"]:CONFIG["SHAPE_DIM_END"]] = self.shape_vectors[shape]
        
        # Add small random noise to remaining dimensions
        noise_dims = CONFIG["DIM_INPUT"] - CONFIG["NOISE_DIM_START"]
        vec[CONFIG["NOISE_DIM_START"]:] = np.random.randn(noise_dims) * 0.01
        
        return vec
    
    def _generate_data(self):
        """
        Generate clusters for each color-shape combination
        """
        for color in self.colors:
            for shape in self.shapes:
                base_vec = self._create_attribute_vector(color, shape)
                
                # Generate cluster around base vector
                cluster = np.random.randn(
                    CONFIG["SAMPLES_PER_COMBO"],
                    CONFIG["DIM_INPUT"]
                ) * CONFIG["CLUSTER_SIGMA"] + base_vec
                
                self.data.extend(cluster)
                self.labels.extend([{
                    'color': color,
                    'shape': shape,
                    'index': len(self.data) - CONFIG["SAMPLES_PER_COMBO"] + i
                } for i in range(CONFIG["SAMPLES_PER_COMBO"])])
        
        self.data = np.array(self.data)
    
    def get_embedding(self, attribute):
        """
        Get prototype embedding for a single attribute (color or shape)
        """
        if attribute in self.colors:
            vec = np.zeros(CONFIG["DIM_INPUT"])
            vec[CONFIG["COLOR_DIM_START"]:CONFIG["COLOR_DIM_END"]] = self.color_vectors[attribute]
            return vec
        elif attribute in self.shapes:
            vec = np.zeros(CONFIG["DIM_INPUT"])
            vec[CONFIG["SHAPE_DIM_START"]:CONFIG["SHAPE_DIM_END"]] = self.shape_vectors[attribute]
            return vec
        else:
            raise ValueError(f"Unknown attribute: {attribute}")
    
    def get_ground_truth(self, color=None, shape=None):
        """
        Get indices of items matching the specified attributes
        """
        indices = []
        for label in self.labels:
            if color is not None and label['color'] != color:
                continue
            if shape is not None and label['shape'] != shape:
                continue
            indices.append(label['index'])
        return indices
    
    def get_prototype(self, color=None, shape=None):
        """
        Get mean embedding for items with specified attributes
        """
        indices = self.get_ground_truth(color=color, shape=shape)
        if len(indices) == 0:
            return None
        return np.mean(self.data[indices], axis=0)


def generate_clevr_lite():
    """
    Generate CLEVR-lite dataset
    
    Returns:
        CLEVRLiteDataset instance
    """
    return CLEVRLiteDataset(seed=CONFIG["RANDOM_SEED"])
