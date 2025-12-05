# Quick Reference: Running Improved Experiments

## Available Variants

### 1. More Witnesses (BEST ACCURACY)
```bash
python3 experiment_improved.py more_witnesses
```
- Initial accuracy: **0.86**
- Best for: High-quality retrieval
- L₀ = 1024, compression = 0.5

### 2. Optimized (BALANCED)
```bash
python3 experiment_improved.py optimized
```
- Better sustained accuracy
- Best for: Robust compression
- L₀ = 768, compression = 0.6

### 3. Gentle Compression
```bash
python3 experiment_improved.py gentle_compression
```
- Slower compression (0.7 factor)
- Best for: Preserving accuracy longer

### 4. Sharp Affinities
```bash
python3 experiment_improved.py sharp_affinities
```
- Lower temperature (0.5)
- Best for: Sparse representations

### 5. Simpler Hierarchy
```bash
python3 experiment_improved.py simpler_hierarchy
```
- Depth = 2 (125 clusters)
- Best for: Cleaner structure

## Compare Multiple Variants

```bash
# Compare baseline vs improved
python3 experiment_improved.py baseline more_witnesses

# Compare all variants
python3 experiment_improved.py baseline optimized more_witnesses gentle_compression

# Run all
python3 experiment_improved.py baseline optimized more_witnesses gentle_compression sharp_affinities simpler_hierarchy
```

## Key Insight

**Optimal witness-to-cluster ratio: 1.5-2.0**

If your data has N clusters, set:
```python
L_MICRO = int(N * 1.75)  # Sweet spot
```

## Files

- `config_improved.py` - All variant configurations
- `experiment_improved.py` - Improved experiment runner
- `IMPROVEMENT_ANALYSIS.md` - Detailed analysis
- `variant_comparison.png` - Visual comparison
