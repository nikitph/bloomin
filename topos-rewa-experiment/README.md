# Topos-REWA Logic & Consistency Experiment

Implementation of sheaf-theoretic logic over witness manifolds, validating:
1. **Compositional Reasoning** via Sheaf Gluing
2. **Truth Maintenance** via KL-Projection

## Overview

This experiment validates the core claims of sheaf-theoretic approaches to logical reasoning on manifolds:
- **Sheaf Gluing** outperforms standard vector arithmetic for compositional queries
- **KL-Projection** can resolve logical contradictions geometrically

## Structure

```
topos-rewa-experiment/
├── config.py                          # Configuration parameters
├── utils.py                           # Helper functions (KL, Fisher distance, metrics)
├── data_generation.py                 # CLEVR-lite synthetic dataset
├── witness_manifold.py                # WitnessManifold class (Fisher geometry)
├── semantic_sheaf.py                  # SemanticSheaf class (sheaf operations)
├── experiment_composition.py          # Phase 1: Composition via Gluing
├── experiment_truth_maintenance.py    # Phase 2: Truth Maintenance
├── run_experiments.py                 # Main orchestration script
├── requirements.txt                   # Dependencies
└── results/                           # Generated plots and outputs
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run all experiments:
```bash
python run_experiments.py
```

Run individual experiments:
```bash
python experiment_composition.py
python experiment_truth_maintenance.py
```

## Expected Results

### Phase 1: Composition via Gluing
- **Topos (Sheaf Gluing)**: Near 100% precision for compositional queries like "Red Squares"
- **Baseline (Vector Arithmetic)**: Lower precision due to averaging artifacts (retrieves "Red Circles" or "Blue Squares")

### Phase 2: Truth Maintenance
- Smooth KL convergence curve showing geometric resolution of contradictions
- Minimal Fisher distance movement while resolving logical inconsistency
- Demonstrates "closest possible" semantic transformation

## Configuration

Key parameters in `config.py`:
- `N_WITNESSES`: Number of witness prototypes (default: 128)
- `FISHER_BETA`: Temperature for witness distribution (default: 1.0)
- `NEIGHBORHOOD_RADIUS`: Size of open sets (default: 0.5)
- `CONSISTENCY_THRESHOLD`: Max KL divergence for valid gluing (default: 0.1)

## Outputs

Results are saved to `results/`:
- `composition_comparison.png`: Precision/Recall/F1 comparison
- `truth_maintenance_dynamics.png`: KL convergence and geometric path
