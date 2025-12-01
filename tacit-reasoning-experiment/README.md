# Tacit Reasoning Experiments

This directory contains experiments validating "tacit reasoning" via the REWA framework—multi-step reasoning through hierarchical witness extraction and tropical aggregation, without explicit token-by-token verbalization.

## Experiments

### Experiment 1: Baseline
- **Task**: Hierarchical graph navigation (predict shortest path distance)
- **Models**: Direct MLP vs Tacit Reasoner (hierarchical witnesses + tropical aggregation)
- **Result**: Both achieve ~87-88% accuracy on in-distribution data
- **Conclusion**: Tacit architecture is expressive enough

### Experiment 2-4: Out-of-Distribution Tests
- **Test 1**: Path length extrapolation (train 1-4, test 5-8)
  - Baseline: 0% accuracy
- **Test 2**: Novel cluster combinations (train A-B, B-C; test A-C)
  - Baseline: 18.75% → With auxiliary losses: 40.50% (+21.75%)
- **Test 3**: Witness capacity scaling
  - Baseline: 52.85% → With auxiliary losses: 66.35% (+13.50%)

**Key Finding**: Auxiliary losses (Triangle Inequality + Witness Consistency) significantly improve compositional generalization.

### Experiment 5: Curriculum Learning
- **Method**: Progressive training on path lengths 1→2→3→...→8
- **Result**: 76.72% accuracy on length 5-6 extrapolation (vs 0% baseline)
- **Witness Correlation**: 0.09 (short paths), -0.02 (long paths)
- **Conclusion**: Curriculum forces compositional learning; witnesses encode implicit features that compose correctly

## Key Insights

### 1. Witness Extraction Requires Structure Access
The MLP-based witness extractor failed because it lacked access to graph structure. It had to infer topology from distance supervision alone—a hard inverse problem.

### 2. Curriculum Reveals Structure Implicitly
Progressive training forced the model to learn that longer paths decompose into shorter ones, revealing compositional structure.

### 3. LLM Hidden States ARE the Structure
For language tasks, LLM hidden states encode semantic relationships. Witness extraction from LLM hidden states should be analogous to extraction from a known graph—the hard work is already done.

### 4. The Theory-Practice Gap
REWA theory assumes "good witnesses" exist. Our experiments show:
- **Expressiveness**: ✅ The architecture can represent solutions
- **Learning**: ❌ Standard training finds degenerate solutions
- **Inductive Biases**: ✅ Auxiliary losses + curriculum enable structural learning

## Files

- `data_gen.py`: Hierarchical graph generator
- `models.py`: TacitReasoner, DirectModel
- `models_with_losses.py`: Enhanced model with auxiliary losses
- `train.py`: Basic training loop
- `train_with_losses.py`: Training with auxiliary losses
- `run_experiments.py`: Baseline OOD tests
- `run_experiment_4.py`: Auxiliary loss experiments
- `curriculum_learning.py`: Curriculum training and analysis
- `analyze_witnesses.py`: Witness-distance correlation diagnostic
- `INSIGHTS.md`: Detailed analysis and LLM parallel

## Running Experiments

```bash
# Baseline
python3 train.py

# OOD tests
python3 run_experiments.py

# Auxiliary losses
python3 run_experiment_4.py

# Curriculum learning
python3 curriculum_learning.py
```

## Next Steps

**Experiment 6**: LLM-based Tacit Reasoning
- Implement witness extraction from LLM hidden states
- Test on multi-hop QA or logical reasoning
- Measure witness overlap gap
- Validate whether LLM hidden states satisfy REWA conditions

See `INSIGHTS.md` for detailed architecture proposal.
