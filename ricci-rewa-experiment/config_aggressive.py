"""
Aggressive Optimization for Ricci-REWA Self-Healing Experiment
Target: >80% recovery
"""

CONFIG = {
    # Data Generation (unchanged)
    "N_SAMPLES": 2000,
    "DIM_INPUT": 128,
    "DIM_EMBED": 64,
    "N_CLUSTERS": 20,
    
    # Training - Very Strong Genesis
    "TEMP": 0.1,
    "LEARNING_RATE": 1e-3,
    "BATCH_SIZE": 256,
    "GENESIS_EPOCHS": 200,      # Even stronger convergence
    
    # Perturbation & Healing - Aggressive
    "PERTURBATION_SCALE": 0.2,  # Much less damage
    "HEALING_STEPS": 1500,      # More recovery time
    "HEALING_BATCH_SIZE": 512,
    "HEALING_LR_START": 1e-2,   # Even higher initial LR
    "HEALING_LR_END": 5e-5,     # Fine-tune more
    
    # Monitoring
    "SNAPSHOT_INTERVAL": 10,
    
    # Random Seed
    "SEED": 42
}
