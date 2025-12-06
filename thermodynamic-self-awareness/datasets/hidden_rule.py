"""
Hidden Rule Dataset Generator

Creates datasets with implicit rules (e.g., red → large) that are not
explicitly labeled but must be discovered through correlation.
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class HiddenRuleExample:
    """Single example with attributes and hidden rule"""
    item_id: str
    attributes: Dict[str, str]  # e.g., {'color': 'red', 'shape': 'cube'}
    features: np.ndarray  # Dense feature vector
    hidden_properties: Dict[str, str]  # e.g., {'size': 'large'} - not given to agent


class HiddenRuleDataset:
    """
    Generate synthetic dataset with hidden correlational rules.
    
    The agent observes explicit attributes (color, shape, material)
    but must discover hidden properties (size) through correlation.
    
    Rule: red → large (with high probability)
    """
    
    def __init__(
        self,
        n_samples: int = 10000,
        rule_strength: float = 0.95,  # P(large | red)
        feature_dim: int = 64,
        seed: int = 42
    ):
        self.n_samples = n_samples
        self.rule_strength = rule_strength
        self.feature_dim = feature_dim
        self.rng = np.random.RandomState(seed)
        
        # Attribute vocabularies
        self.colors = ['red', 'blue', 'green', 'yellow', 'purple']
        self.shapes = ['cube', 'sphere', 'cylinder', 'pyramid', 'torus']
        self.materials = ['metal', 'wood', 'plastic', 'glass']
        self.sizes = ['small', 'medium', 'large']
        
        # Feature embeddings for each attribute value
        self.attribute_embeddings = self._create_attribute_embeddings()
        
    def _create_attribute_embeddings(self) -> Dict[str, np.ndarray]:
        """Create random embeddings for each attribute value"""
        embeddings = {}
        
        for color in self.colors:
            embeddings[f'color_{color}'] = self.rng.randn(self.feature_dim)
        
        for shape in self.shapes:
            embeddings[f'shape_{shape}'] = self.rng.randn(self.feature_dim)
        
        for material in self.materials:
            embeddings[f'material_{material}'] = self.rng.randn(self.feature_dim)
        
        # Normalize
        for key in embeddings:
            embeddings[key] = embeddings[key] / (np.linalg.norm(embeddings[key]) + 1e-10)
        
        return embeddings
    
    def generate(self) -> List[HiddenRuleExample]:
        """
        Generate dataset with hidden rule: red → large
        
        Returns:
            List of examples with explicit and hidden attributes
        """
        examples = []
        
        for i in range(self.n_samples):
            # Sample explicit attributes
            color = self.rng.choice(self.colors)
            shape = self.rng.choice(self.shapes)
            material = self.rng.choice(self.materials)
            
            # Apply hidden rule: red → large
            if color == 'red':
                # High probability of being large
                if self.rng.random() < self.rule_strength:
                    size = 'large'
                else:
                    size = self.rng.choice(['small', 'medium'])
            else:
                # Random size for non-red objects
                size = self.rng.choice(self.sizes)
            
            # Create feature vector (combination of attribute embeddings)
            features = (
                self.attribute_embeddings[f'color_{color}'] +
                self.attribute_embeddings[f'shape_{shape}'] +
                self.attribute_embeddings[f'material_{material}']
            )
            
            # Add noise
            features = features + self.rng.randn(self.feature_dim) * 0.1
            features = features / (np.linalg.norm(features) + 1e-10)
            
            example = HiddenRuleExample(
                item_id=f'item_{i}',
                attributes={
                    'color': color,
                    'shape': shape,
                    'material': material
                },
                features=features,
                hidden_properties={'size': size}
            )
            
            examples.append(example)
        
        return examples
    
    def create_test_queries(self, n_queries: int = 100) -> List[Tuple[Dict, str]]:
        """
        Create test queries to evaluate rule discovery.
        
        Returns:
            List of (query_attributes, expected_size) tuples
        """
        queries = []
        
        for _ in range(n_queries):
            color = self.rng.choice(self.colors)
            shape = self.rng.choice(self.shapes)
            material = self.rng.choice(self.materials)
            
            # Expected size based on rule
            if color == 'red':
                expected_size = 'large'
            else:
                expected_size = 'unknown'  # No rule for other colors
            
            query = {
                'color': color,
                'shape': shape,
                'material': material
            }
            
            queries.append((query, expected_size))
        
        return queries
    
    def evaluate_rule_discovery(
        self,
        discovered_rules: Dict,
        test_queries: List[Tuple[Dict, str]]
    ) -> float:
        """
        Evaluate whether discovered rules match ground truth.
        
        Args:
            discovered_rules: Dictionary of discovered rules
            test_queries: Test queries with expected answers
            
        Returns:
            Accuracy on test queries
        """
        if 'red_implies_large' not in discovered_rules:
            return 0.0
        
        correct = 0
        for query, expected_size in test_queries:
            if query['color'] == 'red' and expected_size == 'large':
                correct += 1
        
        return correct / len(test_queries)


def create_hidden_rule_dataset(
    n_train: int = 8000,
    n_test: int = 2000,
    rule_strength: float = 0.95,
    seed: int = 42
) -> Tuple[List[HiddenRuleExample], List[HiddenRuleExample], List[Tuple[Dict, str]]]:
    """
    Create train/test split for hidden rule experiment.
    
    Returns:
        (train_examples, test_examples, test_queries)
    """
    # Training set
    train_dataset = HiddenRuleDataset(
        n_samples=n_train,
        rule_strength=rule_strength,
        seed=seed
    )
    train_examples = train_dataset.generate()
    
    # Test set
    test_dataset = HiddenRuleDataset(
        n_samples=n_test,
        rule_strength=rule_strength,
        seed=seed + 1
    )
    test_examples = test_dataset.generate()
    
    # Test queries
    test_queries = test_dataset.create_test_queries(n_queries=200)
    
    return train_examples, test_examples, test_queries
