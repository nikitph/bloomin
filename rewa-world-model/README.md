# Multiscale Geometric REWA World-Model PoC

A proof-of-concept implementation integrating multiple REWA frameworks into a unified world-model for semantic search, reasoning, and self-healing.

## Architecture Overview

This system combines:
1. **REWA Encoding**: Witness-based compression with capacity guarantees
2. **Fisher Geometry**: Information-geometric metric learning
3. **Semantic RG**: Multiscale coarse-graining and abstraction
4. **Topos Logic**: Local propositions with gluing consistency
5. **Ricci-REWA**: Metric evolution and self-healing dynamics

## Modules

### 1. Witness Extraction (`witnesses/`)
- Boolean witnesses (tokens, keywords)
- Natural witnesses (counts, frequencies)
- Real witnesses (embeddings, scores)
- Tropical witnesses (graph distances, min-plus)

### 2. REWA Encoder (`encoding/`)
- Capacity-based index sizing (Shannon formula)
- Multi-hash aggregation
- Monoid operations (Boolean/Natural/Real/Tropical)

### 3. Neural Encoder (`neural/`)
- Contrastive learning (InfoNCE)
- Fisher-Rao metric approximation
- Natural gradient optimization

### 4. Geometry Module (`geometry/`)
- Fisher metric computation
- Scalar curvature diagnostics
- Geodesic distance estimation

### 5. Semantic RG (`semantic_rg/`)
- Witness blocking and clustering
- Multiscale coarse-graining
- Renormalized metric computation

### 6. Topos Module (`topos/`)
- Local proposition extraction
- Restriction maps
- Gluing consistency
- KL-projection truth maintenance

### 7. Ricci-REWA (`ricci/`)
- Ricci tensor computation
- Lichnerowicz Laplacian
- Discrete PDE evolution
- Self-healing dynamics

### 8. Retrieval & Reasoning (`retrieval/`)
- Multiscale REWA retrieval
- Fisher-metric refinement
- Topological consistency checks

### 9. Evaluation (`evaluation/`)
- Retrieval metrics (Recall@K, MRR)
- Information metrics (capacity, MI)
- Geometric metrics (curvature, distortion)
- Logical metrics (consistency, QA accuracy)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run basic REWA encoding test
python examples/01_basic_rewa.py

# Train contrastive encoder
python examples/02_train_encoder.py

# Run self-healing experiment
python examples/03_ricci_healing.py

# Full evaluation suite
python examples/04_full_evaluation.py
```

## Project Structure

```
rewa-world-model/
├── src/
│   ├── witnesses/          # Witness extraction
│   ├── encoding/           # REWA encoding
│   ├── neural/             # Contrastive encoder
│   ├── geometry/           # Fisher geometry
│   ├── semantic_rg/        # Semantic RG
│   ├── topos/              # Topos logic
│   ├── ricci/              # Ricci-REWA
│   ├── retrieval/          # Retrieval + reasoning
│   └── evaluation/         # Metrics
├── examples/               # Example scripts
├── tests/                  # Unit tests
├── data/                   # Datasets
└── results/                # Experiment results
```

## Key Papers

- REWA Framework: Witness encoding and capacity
- Shannon-REWA: Information-theoretic bounds
- Fisher Geometry: Curvature diagnostics
- Ricci-REWA: Metric evolution and self-healing
- Semantic RG: Multiscale coarse-graining
- Topos-REWA: Local logic and gluing

## Development Roadmap

- **Week 1**: Core infrastructure (witnesses, encoding, retrieval)
- **Week 2**: Geometry (Fisher metric, curvature)
- **Week 3**: Semantic RG (coarse-graining, compression)
- **Week 4**: Topos (propositions, gluing, QA)
- **Week 5-6**: Ricci-REWA (evolution, self-healing)
- **Week 7**: Integration and evaluation

## License

MIT
