# Ricci-REWA Continual Learning Experiment Summary

## What We Built

A complete experimental framework for testing the Ricci-REWA hypothesis in continual learning:

```
ricci-continual-learning/
├── src/
│   ├── ricci_curvature.py       # Ollivier-Ricci curvature computation
│   ├── ricci_curvature_v2.py    # Improved local geometry regularizers
│   ├── curvature_focused.py     # Scale-invariant curvature measures
│   ├── models.py                # MLP and ConvNet architectures
│   ├── continual_learning.py    # Baseline, EWC, and Ricci-Reg methods
│   ├── continual_learning_v2.py # Improved geometry regularization
│   ├── continual_learning_v3.py # Focused curvature preservation
│   └── experiment.py            # Main experiment runner
├── run_experiment.py            # Basic experiment
├── run_improved_experiment.py   # With lambda sweep
├── run_focused_experiment.py    # Scale-invariant approach
└── requirements.txt
```

## The Hypothesis

**Ricci-REWA predicts:** Preserving Ricci curvature during continual learning should preserve task performance, even when weights change dramatically.

**Concrete prediction:** A network trained on MNIST, then on FashionMNIST with curvature regularization, should retain MNIST accuracy better than EWC, while allowing larger weight changes.

## Results (Quick Tests)

| Method        | MNIST↓    | Fashion   | Weight Δ | Forgetting |
|---------------|-----------|-----------|----------|------------|
| Baseline      | ~20%      | ~82%      | ~8.4     | ~70%       |
| EWC (λ=5000)  | ~35-40%   | ~80%      | ~3.8     | ~55%       |
| Ricci (v1)    | ~25%      | ~78%      | ~9.5     | ~65%       |
| Ricci (v2)    | ~22%      | ~81%      | ~9.2     | ~70%       |
| Ricci (v3)    | ~26%      | ~81%      | ~9.3     | ~67%       |

**Key Finding:** EWC consistently outperforms our Ricci curvature methods.

## Analysis: Why Isn't It Working?

### 1. The Curvature-Classification Gap
The curvature of the embedding space doesn't directly control classification. Classification happens in the final linear layer, which partitions the embedding space with hyperplanes. Preserving curvature doesn't necessarily preserve these partitions.

### 2. Local vs Global Structure
Our curvature measures are local (k-NN based), but task performance may depend on global structure. EWC captures importance at the parameter level, which affects the entire computation.

### 3. Gradient Signal Strength
The curvature loss provides a relatively weak gradient signal. Even with high λ, it doesn't strongly constrain learning. EWC's quadratic penalty on parameter changes provides a stronger, more direct signal.

### 4. What Curvature Are We Actually Preserving?
We implemented:
- Ollivier-Ricci curvature (graph-based)
- Local distance distributions
- Angular relationships between neighbors
- Second-order neighborhood overlap

None of these may capture the "functionally relevant" curvature that Ricci-REWA refers to.

## What We Learned

### Positive Findings:
1. **Weight changes are larger with Ricci-Reg** (~9.3 vs ~3.8 for EWC), showing it allows more parameter plasticity
2. **Fashion accuracy is maintained** (~81% vs ~80% for EWC), showing no loss of plasticity
3. **The framework works** - we can successfully regularize curvature during training

### Negative Findings:
1. **MNIST retention is worse** than EWC despite curvature preservation
2. **The curvature-performance link is unclear** - preserving our curvature measures doesn't preserve task performance

## Next Steps: What Would Make This Work?

### 1. Class-Conditional Curvature
Instead of preserving global curvature, preserve curvature **within each class cluster**. The intra-class geometry may be more task-relevant.

```python
def class_conditional_curvature(embeddings, labels):
    """Compute curvature separately for each class."""
    curvatures = {}
    for c in unique(labels):
        class_emb = embeddings[labels == c]
        curvatures[c] = compute_ricci(class_emb)
    return curvatures
```

### 2. Fisher-Weighted Curvature
Use Fisher information to identify which **directions** in embedding space matter, then preserve curvature along those directions.

### 3. Decision Boundary Geometry
Preserve the geometry of decision boundaries, not embeddings. This directly relates to classification.

### 4. Representation Similarity Analysis
Use methods like CKA or RSA to measure representation similarity, which may better capture functional equivalence.

### 5. Theoretical Refinement
The original Ricci-REWA theory may need refinement:
- What exactly is the "curvature" that matters?
- Is it the curvature of the loss landscape, not the embedding space?
- Should we preserve the Fisher-Rao metric on parameter space?

## Running the Experiments

```bash
# Quick test
python run_focused_experiment.py --ricci-lambda 30

# Lambda sweep
python run_improved_experiment.py --quick --sweep

# Full experiment
PYTORCH_ENABLE_MPS_FALLBACK=1 python run_experiment.py --full
```

## Conclusion

The Ricci-REWA hypothesis - that preserving geometric curvature can prevent catastrophic forgetting - is **not confirmed** by these experiments. However, this doesn't falsify the theory; it indicates our curvature proxies may not capture the relevant geometric structure.

The key insight: **there is a gap between the curvature we can compute (Ollivier-Ricci, local distances, angles) and the "functional curvature" that determines task performance.**

Future work should focus on:
1. Better understanding what geometric structure determines classification
2. Developing curvature measures that capture this structure
3. Investigating whether the relevant geometry is in parameter space, not activation space
