"""
Optimized Configuration for Ricci-REWA Self-Healing Experiment
Based on analysis of baseline results (67.4% recovery)
"""

CONFIG = {
    # Data Generation (unchanged)
    "N_SAMPLES": 2000,
    "DIM_INPUT": 128,
    "DIM_EMBED": 64,
    "N_CLUSTERS": 20,
    
    # Training - Stronger Genesis
    "TEMP": 0.1,
    "LEARNING_RATE": 1e-3,
    "BATCH_SIZE": 256,
    "GENESIS_EPOCHS": 150,      # Increased from 100 for stronger convergence
    
    # Perturbation & Healing - Optimized
    "PERTURBATION_SCALE": 0.3,  # Reduced from 0.5 for less severe damage
    "HEALING_STEPS": 1000,      # Increased from 500 for more recovery time
    "HEALING_BATCH_SIZE": 512,  # Larger batches for stable gradients
    "HEALING_LR_START": 5e-3,   # Higher initial LR for aggressive recovery
    "HEALING_LR_END": 1e-4,     # Decay to fine-tune
    
    # Monitoring
    "SNAPSHOT_INTERVAL": 10,
    
    # Random Seed
    "SEED": 42
}
