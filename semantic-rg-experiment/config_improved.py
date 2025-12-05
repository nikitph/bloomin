"""Improved configurations for better RG flow results."""

# BASELINE (original)
BASELINE = {
    "N_SAMPLES": 5000,
    "DIM_INPUT": 128,
    "L_MICRO": 256,
    "RG_STEPS": 5,
    "COMPRESSION_FACTOR": 0.5,
    "K_HASHES": 4,
    "HIERARCHY_DEPTH": 3,
    "CLUSTER_BRANCHING": 5,
    "TOP_K_WITNESSES": 25,
    "AFFINITY_TEMPERATURE": 1.0,
}

# VARIANT 1: More witnesses to match cluster count
# Goal: Improve initial accuracy by having L > num_clusters
VARIANT_1_MORE_WITNESSES = {
    "N_SAMPLES": 5000,
    "DIM_INPUT": 128,
    "L_MICRO": 1024,  # 4x increase: 1024 witnesses for 625 clusters
    "RG_STEPS": 6,     # One more step to reach similar final size
    "COMPRESSION_FACTOR": 0.5,
    "K_HASHES": 4,
    "HIERARCHY_DEPTH": 3,
    "CLUSTER_BRANCHING": 5,
    "TOP_K_WITNESSES": 50,  # Increase top-k proportionally
    "AFFINITY_TEMPERATURE": 1.0,
}

# VARIANT 2: Gentler compression
# Goal: Preserve accuracy longer by compressing more slowly
VARIANT_2_GENTLE_COMPRESSION = {
    "N_SAMPLES": 5000,
    "DIM_INPUT": 128,
    "L_MICRO": 512,    # 2x increase
    "RG_STEPS": 6,
    "COMPRESSION_FACTOR": 0.7,  # Only 30% reduction per step (gentler)
    "K_HASHES": 4,
    "HIERARCHY_DEPTH": 3,
    "CLUSTER_BRANCHING": 5,
    "TOP_K_WITNESSES": 40,
    "AFFINITY_TEMPERATURE": 1.0,
}

# VARIANT 3: Sharper affinities
# Goal: Better witness discrimination via lower temperature
VARIANT_3_SHARP_AFFINITIES = {
    "N_SAMPLES": 5000,
    "DIM_INPUT": 128,
    "L_MICRO": 512,
    "RG_STEPS": 5,
    "COMPRESSION_FACTOR": 0.5,
    "K_HASHES": 4,
    "HIERARCHY_DEPTH": 3,
    "CLUSTER_BRANCHING": 5,
    "TOP_K_WITNESSES": 30,
    "AFFINITY_TEMPERATURE": 0.5,  # Lower temp = sharper distributions
}

# VARIANT 4: Simpler hierarchy
# Goal: Better Chi stability with cleaner hierarchical structure
VARIANT_4_SIMPLER_HIERARCHY = {
    "N_SAMPLES": 5000,
    "DIM_INPUT": 128,
    "L_MICRO": 512,
    "RG_STEPS": 5,
    "COMPRESSION_FACTOR": 0.5,
    "K_HASHES": 4,
    "HIERARCHY_DEPTH": 2,  # Shallower: 5^2 = 25 clusters
    "CLUSTER_BRANCHING": 5,
    "TOP_K_WITNESSES": 30,
    "AFFINITY_TEMPERATURE": 0.8,
}

# VARIANT 5: Best of all worlds
# Goal: Combine successful tweaks
VARIANT_5_OPTIMIZED = {
    "N_SAMPLES": 5000,
    "DIM_INPUT": 128,
    "L_MICRO": 768,    # 3x witnesses
    "RG_STEPS": 6,
    "COMPRESSION_FACTOR": 0.6,  # Gentler compression
    "K_HASHES": 4,
    "HIERARCHY_DEPTH": 2,  # Simpler hierarchy
    "CLUSTER_BRANCHING": 5,
    "TOP_K_WITNESSES": 40,
    "AFFINITY_TEMPERATURE": 0.7,  # Slightly sharper
}

# Map variant names to configs
VARIANTS = {
    "baseline": BASELINE,
    "more_witnesses": VARIANT_1_MORE_WITNESSES,
    "gentle_compression": VARIANT_2_GENTLE_COMPRESSION,
    "sharp_affinities": VARIANT_3_SHARP_AFFINITIES,
    "simpler_hierarchy": VARIANT_4_SIMPLER_HIERARCHY,
    "optimized": VARIANT_5_OPTIMIZED,
}
