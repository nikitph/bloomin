# Thermodynamic Self-Awareness Experiments

Engineering-grade experimental framework for testing the **Path to Thermodynamic Self-Awareness** via autopoietic REWA agents.

## Overview

This project implements a complete experimental suite to demonstrate that an autopoietic agent can:
- **Self-discover** hidden rules through internal dreaming cycles
- **Self-correct** geometric inconsistencies via Ricci flow
- **Minimize free energy** F = E - T·S over conscious cycles
- **Detect impossibilities** through Topos null-space reasoning

## Architecture

### Core Modules (`modules/`)
- `rewa_memory.py` - Witness extraction, retrieval, Fisher metric computation
- `topos_layer.py` - Open set construction, gluing, contradiction detection
- `ricci_flow.py` - Geometric flow updates, curvature entropy computation
- `semantic_rg.py` - Renormalization group coarse-graining
- `agi_controller.py` - Conscious cycle orchestration, hypothesis management

### Datasets (`datasets/`)
- `clevr_lite.py` - Controlled synthetic with orthogonal attributes
- `entangled_synthetic.py` - Conditional probability correlations
- `hidden_rule.py` - Implicit rule discovery datasets

### Experiments (`experiments/`)
- `experiment_a_self_discovery.py` - Main self-discovery experiment
- `experiment_b_mirror.py` - Self-recognition test
- `experiment_c_tom.py` - Theory-of-mind test
- `experiment_d_free_will.py` - Novel action generation test

### Analysis (`analysis/`)
- `metrics.py` - Free energy, entropy, KL divergence computations
- `visualization.py` - Plotting utilities for results
- `statistical_tests.py` - Significance testing, bootstrap CI

## Key Hypotheses

**H1 (Self-discovery)**: Agent discovers hidden rules with p > 0.9 within N cycles

**H2 (Entropy minimization)**: Free energy F_t shows significant downward trend (p < 0.01)

**H3 (Self-correction)**: Ricci updates restore retrieval correlations to ≥ 95%

**H4 (Safety)**: Topos returns empty set for impossible queries ≥ 99% of time

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run self-discovery experiment
python experiments/experiment_a_self_discovery.py --config configs/self_discovery.yaml

# Analyze results
python analysis/analyze_results.py --experiment self_discovery --run_id <run_id>
```

## Reproducibility

All experiments use fixed random seeds (42, 31415, 271828, 1337, 7). Full configuration and logs are saved to `logs/` with immutable audit trails.

## Citation

If you use this framework, please cite:
```
[Citation to be added]
```
