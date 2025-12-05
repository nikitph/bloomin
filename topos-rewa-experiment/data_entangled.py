"""
Generate entangled dataset for Experiment 4
Attributes are correlated: Metallic implies Grey/Gold, Matte implies Red/Blue/Green
"""

import numpy as np
from config import CONFIG


class EntangledDataset:
    """
    Dataset with correlated attributes (non-orthogonal)
    - Metallic objects: mostly Grey/Gold, high shininess
    - Matte objects: Red/Blue/Green, low shininess
    - NO "Metallic Red" objects in training data
    """
    
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        
        # Define entangled attribute combinations
        self.metallic_colors = ["grey", "gold"]
        self.matte_colors = ["red", "blue", "green"]
        self.all_colors = self.metallic_colors + self.matte_colors
        
        self.data = []
        self.labels = []
        self._generate_entangled_data()
    
    def _create_entangled_vector(self, color, material, shininess):
        """
        Create vector with entangled attributes
        - Color dims: 0-15
        - Material dims: 16-25
        - Shininess dims: 26-30
        - Noise dims: 31+
        """
        vec = np.zeros(CONFIG["DIM_INPUT"])
        
        # Color encoding (different patterns for each color)
        color_idx = self.all_colors.index(color)
        vec[color_idx * 2:color_idx * 2 + 2] = 1.0
        
        # Material encoding
        if material == "metallic":
            vec[16:20] = 1.0
        else:  # matte
            vec[21:25] = 1.0
        
        # Shininess encoding (correlated with material)
        vec[26:30] = shininess
        
        # Noise
        vec[31:] = np.random.randn(CONFIG["DIM_INPUT"] - 31) * 0.01
        
        return vec
    
    def _generate_entangled_data(self):
        """
        Generate entangled clusters
        """
        samples_per_combo = CONFIG["SAMPLES_PER_COMBO"]
        
        # Cluster A: Metallic objects (Grey/Gold, high shininess)
        for color in self.metallic_colors:
            base_vec = self._create_entangled_vector(
                color, 
                material="metallic",
                shininess=0.9  # High shininess
            )
            
            # Generate cluster with correlation
            cluster = np.random.randn(samples_per_combo, CONFIG["DIM_INPUT"]) * 0.15 + base_vec
            
            self.data.extend(cluster)
            self.labels.extend([{
                'color': color,
                'material': 'metallic',
                'shininess': 'high',
                'index': len(self.data) - samples_per_combo + i
            } for i in range(samples_per_combo)])
        
        # Cluster B: Matte objects (Red/Blue/Green, low shininess)
        for color in self.matte_colors:
            base_vec = self._create_entangled_vector(
                color,
                material="matte",
                shininess=0.1  # Low shininess
            )
            
            # Generate cluster
            cluster = np.random.randn(samples_per_combo, CONFIG["DIM_INPUT"]) * 0.15 + base_vec
            
            self.data.extend(cluster)
            self.labels.extend([{
                'color': color,
                'material': 'matte',
                'shininess': 'low',
                'index': len(self.data) - samples_per_combo + i
            } for i in range(samples_per_combo)])
        
        self.data = np.array(self.data)
        
        print(f"Generated entangled dataset:")
        print(f"  Metallic objects: {len([l for l in self.labels if l['material'] == 'metallic'])}")
        print(f"  Matte objects: {len([l for l in self.labels if l['material'] == 'matte'])}")
        print(f"  Total: {len(self.data)}")
    
    def get_embedding(self, attribute):
        """Get prototype embedding for an attribute"""
        if attribute == "metallic":
            vec = np.zeros(CONFIG["DIM_INPUT"])
            vec[16:20] = 1.0
            vec[26:30] = 0.9
            return vec
        elif attribute == "matte":
            vec = np.zeros(CONFIG["DIM_INPUT"])
            vec[21:25] = 1.0
            vec[26:30] = 0.1
            return vec
        elif attribute in self.all_colors:
            color_idx = self.all_colors.index(attribute)
            vec = np.zeros(CONFIG["DIM_INPUT"])
            vec[color_idx * 2:color_idx * 2 + 2] = 1.0
            return vec
        else:
            raise ValueError(f"Unknown attribute: {attribute}")
    
    def get_ground_truth(self, color=None, material=None):
        """Get indices matching specified attributes"""
        indices = []
        for label in self.labels:
            if color is not None and label['color'] != color:
                continue
            if material is not None and label['material'] != material:
                continue
            indices.append(label['index'])
        return indices
    
    def get_prototype(self, color=None, material=None):
        """Get mean embedding for items with specified attributes"""
        indices = self.get_ground_truth(color=color, material=material)
        if len(indices) == 0:
            return None
        return np.mean(self.data[indices], axis=0)


def generate_entangled_data():
    """Generate entangled dataset"""
    return EntangledDataset(seed=CONFIG["RANDOM_SEED"])
