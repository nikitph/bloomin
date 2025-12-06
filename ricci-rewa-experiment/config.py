"""
Configuration for Ricci-REWA Self-Healing Experiment
"""

CONFIG = {
    # Data Generation
    "N_SAMPLES": 2000,          # Database size
    "DIM_INPUT": 128,           # Raw data dimension
    "DIM_EMBED": 64,            # Witness manifold dimension (m)
    "N_CLUSTERS": 20,           # To create structure/curvature
    
    # Training
    "TEMP": 0.1,                # Contrastive temperature
    "LEARNING_RATE": 1e-3,
    "BATCH_SIZE": 256,
    "GENESIS_EPOCHS": 100,      # Training epochs for Phase 1
    
    # Perturbation & Healing
    "PERTURBATION_SCALE": 0.5,  # Magnitude of damage
    "HEALING_STEPS": 500,       # Steps to recover
    
    # Monitoring
    "SNAPSHOT_INTERVAL": 10,    # How often to measure geometry
    
    # Random Seed
    "SEED": 42
}
