"""Configuration for Semantic RG Flow Experiment."""

CONFIG = {
    "N_SAMPLES": 5000,          # Database size
    "DIM_INPUT": 128,           # Raw Euclidean dimension
    "L_MICRO": 256,             # Initial microscopic witness count (Level 0)
    "RG_STEPS": 5,              # Number of coarse-graining steps
    "COMPRESSION_FACTOR": 0.5,  # Each step reduces witnesses by 50%
    "K_HASHES": 4,              # For REWA encoding evaluation
    "HIERARCHY_DEPTH": 3,       # For synthetic data generation
    "CLUSTER_BRANCHING": 5,     # Branching factor of hierarchy
    "TOP_K_WITNESSES": None,    # Will be set to L_MICRO // 10
}

# Derived parameters
CONFIG["TOP_K_WITNESSES"] = CONFIG["L_MICRO"] // 10
