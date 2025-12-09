# Intrinsic Dimension Sweep Experiment

## Overview
This experiment measures the intrinsic dimensionality of embeddings across different models, similar to the universal curvature experiment structure.

## Files Created
1. **`experiment_intrinsic_dim_sweep.py`** - Main experiment class
2. **`run_intrinsic_dim_sweep.py`** - Runner script with CLI
3. **`plot_intrinsic_dim_sweep.py`** - Visualization script

## Usage

### Quick Test (2 models)
```bash
cd universal-curvature-experiment
python3 run_intrinsic_dim_sweep.py --quick
```

### Full Sweep (all models)
```bash
python3 run_intrinsic_dim_sweep.py
```

### Skip Uncached Models (faster)
Only run on models that are already downloaded:
```bash
python3 run_intrinsic_dim_sweep.py --skip-download
```

### Custom Thresholds
```bash
python3 run_intrinsic_dim_sweep.py --thresholds 0.90 0.95 0.99
```

### Generate Plots
```bash
python3 plot_intrinsic_dim_sweep.py --results-dir ./results
```

## Measurements

The experiment measures:
- **Intrinsic Dimension** at 90%, 95%, and 99% variance thresholds
- **Compression Ratio** (full_dim / intrinsic_dim)
- **Effective Rank** (participation ratio)
- **Variance Explained** by top PCs
- **Full Variance Spectrum** (first 100 components)

## Initial Results (Quick Test)

| Model | Full Dim | Intrinsic Dim (95%) | Compression | Effective Rank |
|-------|----------|---------------------|-------------|----------------|
| sentence-bert | 384 | 33 | 11.64x | 29.80 |
| bert-base | 768 | 39 | 19.69x | 22.13 |

**Key Findings:**
- Mean intrinsic dimension: 36.0 ± 4.2
- Mean compression potential: 15.66x
- Both models show high compressibility (~10-20x)
- BERT-base has lower effective rank despite higher dimensionality

## Output Files
- `results/intrinsic_dim_results.csv` - Full results table
- `results/intrinsic_dim_summary.json` - Summary statistics
- `results/variance_spectrum_{model}.json` - Per-model variance spectra
- `results/intrinsic_dim_comparison.png` - Comparison plots
- `results/variance_spectra.png` - Variance curves

## Next Steps
Run the full sweep on all models in `model_configs.py` to get comprehensive results across architectures, dimensions, and domains.
