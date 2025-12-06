# Semantic Renormalization Group (RG) Flow Experiment

This experiment empirically validates the **RG Flow equations** by demonstrating that coarse-graining witnesses (abstraction) flows towards specific fixed points while preserving retrieval capabilities.

## Overview

The experiment implements a Renormalization Group operator $\mathcal{R}_s$ that iteratively merges microscopic witnesses into macroscopic semantic packets. It tracks the flow of geometric couplings ($K, \rho, L$) and demonstrates that retrieval accuracy is preserved even as the representation size compresses logarithmically.

## Key Concepts

- **Witnesses**: Semantic prototypes that represent data points
- **RG Flow**: Iterative coarse-graining process that merges similar witnesses
- **Fixed Point**: Scale-invariant state where thermodynamic variable χ stabilizes
- **Abstraction**: Compression that preserves retrieval accuracy

## Project Structure

```
semantic-rg-experiment/
├── config.py              # Experiment configuration
├── data_generation.py     # Hierarchical data generation
├── witness_init.py        # Witness initialization via k-means
├── rg_operator.py         # Core RG blocking operator
├── measurements.py        # Observable measurements
├── experiment.py          # Main experiment orchestration
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Installation

```bash
cd semantic-rg-experiment
pip install -r requirements.txt
```

## Usage

Run the complete experiment:

```bash
python experiment.py
```

This will:
1. Generate hierarchical data (5000 samples, 128D)
2. Initialize 256 microscopic witnesses
3. Apply 5 RG coarse-graining steps (50% compression each)
4. Measure observables at each scale
5. Generate visualization plots

## Expected Results

### Fixed Point Behavior
- **χ(s) stabilizes**: Even though L (bits) drops, ρ (signal) increases
- Since χ ∝ m·Δ², if m drops by 2x and Δ increases by √2x, χ is invariant
- This proves **Scale Invariance**

### Abstraction Preservation
- Accuracy stays high (flat line) for initial steps
- Drops only when block size exceeds natural cluster size
- This proves **Optimal Compression**

## Configuration

Edit `config.py` to adjust parameters:

```python
CONFIG = {
    "N_SAMPLES": 5000,          # Database size
    "DIM_INPUT": 128,           # Raw Euclidean dimension
    "L_MICRO": 256,             # Initial witness count
    "RG_STEPS": 5,              # Number of coarse-graining steps
    "COMPRESSION_FACTOR": 0.5,  # Compression per step
    "HIERARCHY_DEPTH": 3,       # Data hierarchy depth
    "CLUSTER_BRANCHING": 5      # Branching factor
}
```

## Output

The experiment generates:
- `rg_flow_results.png`: 6-panel visualization showing:
  - Witness multiplicity flow L(s)
  - Signal concentration ρ(s)
  - Fixed point behavior χ(s)
  - Retrieval accuracy preservation
  - One-shot generalization (compression vs accuracy)
  - Geometric curvature flow K(s)

- Console output with detailed measurements at each scale

## Theory

This experiment validates the theoretical framework where:
- **Microscopic scale**: Many witnesses, low signal overlap
- **RG flow**: Witnesses merge based on semantic affinity
- **Macroscopic scale**: Fewer witnesses, higher signal concentration
- **Fixed point**: χ remains stable across scales

The key insight is that semantic information is preserved during coarse-graining, enabling efficient compression without loss of retrieval capability.
