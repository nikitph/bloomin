# Ricci-REWA Self-Healing Experiment

**Empirical demonstration that neural encoders trained with contrastive loss exhibit "self-healing" geometric properties when perturbed, effectively solving a Ricci flow-like PDE to smooth out manifold defects.**

## Overview

This experiment validates the theoretical claim that contrastive learning (InfoNCE) acts as a natural gradient descent on the witness manifold, creating a restoring force that repairs geometric damage.

### The 4 Phases

1. **Genesis**: Train encoder to convergence, capture healthy geometry
2. **Injury**: Inject noise into weights to create geometric defect
3. **Healing**: Resume training and monitor recovery via Ricci flow
4. **Visualization**: Plot recovery dynamics and curvature evolution

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the experiment
python experiment.py
```

## Expected Results

**Success Criteria:**
- Recovery score shows exponential decay toward zero
- Curvature entropy spikes after damage, then smooths out during recovery
- Final recovery >70% of initial damage

## Project Structure

```
ricci-rewa-experiment/
├── config.py              # Experimental constants
├── data_generation.py     # Hierarchical Gaussian data
├── model.py              # MLP encoder + InfoNCE loss
├── geometry.py           # Gram matrix & curvature metrics
├── experiment.py         # Main 4-phase experiment
├── visualization.py      # Plotting functions
├── requirements.txt      # Dependencies
└── results/             # Generated plots
```

## Theory

### Geometric Proxies

- **Metric Tensor (g)**: Approximated by Gram matrix `G = Z Z^T` where `Z` are normalized embeddings
- **Curvature**: Measured via spectral entropy of eigenvalues
  - High entropy = Flat/Noisy geometry
  - Low entropy = Curved/Structured geometry

### The Ricci Flow Analogy

The contrastive update rule acts as:
```
∂g/∂t = -2 Ric(g) + S
```

Where `S` is the forcing term (semantic structure) that pulls the manifold back to its stable geodesic configuration.

## Configuration

Edit `config.py` to adjust:
- `N_SAMPLES`: Dataset size (default: 2000)
- `DIM_EMBED`: Embedding dimension (default: 64)
- `PERTURBATION_SCALE`: Damage magnitude (default: 0.5)
- `HEALING_STEPS`: Recovery iterations (default: 500)

## Citation

If you use this experiment, please cite:
```
Ricci-REWA Self-Healing Dynamics: Empirical Validation of 
Geometric Memory in Contrastive Learning
```
