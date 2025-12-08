import torch
import numpy as np
import random
import yaml
import os

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_config(config_path='config.yaml'):
    """Load configuration from YAML or return defaults"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default configuration
        return {
            # Model
            'embedding_dim': 128,
            'n_heads': 4,
            'n_layers': 4,
            'sharpen_power': 2.0,
            'sharpen_iters_slow': 10,
            
            # Training
            'epochs': 20,
            'wake_steps_per_epoch': 1000,
            'slow_path_frequency': 0.1,
            'batch_size': 32,
            'learning_rate': 1e-3,
            
            # Evaluation
            'success_threshold_distance': 0.15,
            'success_threshold_accuracy': 0.85,
            
            # Paths
            'save_dir': './results',
            'checkpoint_dir': './results/checkpoints',
        }

def random_peaked_distribution(dim=128, peak_loc=0, sigma=10.0):
    """
    Generate a random distribution peaked at a specific location.
    Used for creating grounded Primary concepts.
    """
    x = torch.arange(dim).float()
    # Gaussian centered at peak_loc
    dist = torch.exp(-((x - peak_loc)**2) / (2 * sigma**2))
    # Normalize to probability
    dist = dist / dist.sum()
    return dist

def geometric_mean(p1, p2):
    """Compute geometric mean of two distributions"""
    epsilon = 1e-10
    mix = torch.sqrt((p1 + epsilon) * (p2 + epsilon))
    return mix / mix.sum()
