"""
Configuration for Topos-REWA Logic & Consistency Experiment
"""

CONFIG = {
    "N_SAMPLES": 1000,
    "DIM_INPUT": 64,
    "N_WITNESSES": 128,
    "FISHER_BETA": 1.0,         # Softmax temp for witness distribution
    "NEIGHBORHOOD_RADIUS": 0.5, # Size of Open Sets (U) in Fisher metric
    "CONSISTENCY_THRESHOLD": 0.1, # Max KL divergence for valid gluing
    
    # Data generation
    "CLUSTER_SIGMA": 0.1,       # Gaussian noise for attribute clusters
    "SAMPLES_PER_COMBO": 100,   # Samples per color-shape combination
    
    # Dimension allocation for attributes
    "COLOR_DIM_START": 0,
    "COLOR_DIM_END": 10,
    "SHAPE_DIM_START": 11,
    "SHAPE_DIM_END": 20,
    "NOISE_DIM_START": 21,
    
    # Truth maintenance
    "TM_LEARNING_RATE": 0.1,
    "TM_MAX_STEPS": 100,
    
    # Autopoietic invention
    "MANIFOLD_TEMP": 0.1,       # Sharpness of existing concepts
    "DREAM_TEMP": 1.0,          # High temp for creative mixing
    "DISSONANCE_THRESHOLD": 0.5, # How much logical pain before learning?
    "HEALING_STEPS": 200,       # Steps to stabilize new concept
    
    # Output
    "RESULTS_DIR": "results",
    "RANDOM_SEED": 42,
}
