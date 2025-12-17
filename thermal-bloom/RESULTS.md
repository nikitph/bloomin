# Thermal Bloom Filter - PoC Results

## Executive Summary

The Thermal Bloom Filter concept shows **exceptional performance on inherently low-dimensional data** but does not compete with state-of-the-art methods like FAISS on high-dimensional embeddings.

## Key Results

### Low-Dimensional Data (2D Clustered Points)

| Method | Recall@1 | QPS | Status |
|--------|----------|-----|--------|
| Discrete Bloom | 17.6% | 29.8M | Baseline |
| **Thermal Bloom V2** | **99.6%** | 127K | **HOLY SHIT** |

- **5.7x improvement** over discrete bloom
- Gradient ascent converges in ~12 steps
- Passes "Holy Shit" threshold (>85% recall)

### High-Dimensional Embeddings (384D)

| Method | Recall@1 | QPS |
|--------|----------|-----|
| FAISS Flat (exact) | 100% | 16.8K |
| FAISS HNSW (ef=256) | 62.6% | 3.1K |
| Best Thermal | 17.3% | 104 |

- Thermal methods achieve only 1-17% recall
- The fundamental issue: **projecting 384D → 2D loses too much information**

## Why It Works on 2D, Not 384D

### The Math

1. **Johnson-Lindenstrauss Lemma**: To preserve pairwise distances within ε error, you need O(log n / ε²) dimensions. For 50K points at 10% error, that's ~100 dimensions minimum.

2. **Intrinsic Dimensionality**: Sentence embeddings have high intrinsic dimensionality (~50-100). A 2D projection cannot capture this structure.

3. **Neighborhood Preservation**: In 2D projections, points that are neighbors in 384D may map to distant locations, and vice versa.

### Visual Intuition

```
384D Space:          2D Projection:
  A●--B●              A●    B●
   \  /
    ●C                    C●

In 384D: A, B, C are all equidistant
In 2D: C appears closer to neither A nor B
```

## Where Thermal Bloom Excels

### 1. Geospatial Applications
- Location-based search
- Map routing
- Spatial indexing

### 2. Image Feature Maps
- After CNN projection to low-D
- Visual similarity search on compressed representations

### 3. 2D/3D Scientific Data
- Molecular docking (2D surfaces)
- Point cloud processing
- Simulation data

### 4. Hierarchical Refinement
- As a fast filter within clusters
- Combined with coarse-grained methods

## Recommendations

### Use Thermal Bloom When:
- Data is inherently 2D or 3D
- You have strong cluster structure
- You need a lightweight, cache-friendly index
- Memory efficiency is critical

### Use FAISS/HNSW When:
- Data is high-dimensional embeddings
- You need general-purpose ANN search
- Recall is more important than novelty

## Technical Innovation Value

While not suitable for general high-D ANN, the Thermal Bloom concept introduces valuable ideas:

1. **Continuous Fields for Discrete Data**: Using diffusion to create navigable gradient landscapes
2. **Gradient-Based Search**: Following heat gradients instead of graph traversal
3. **Natural Hierarchy**: Thermal diffusion creates implicit multi-scale structure

## Future Directions

To make Thermal Bloom competitive on high-D data:

1. **Learned Projections**: Train neural networks to project embeddings to 2D while preserving neighborhoods
2. **Product Quantization + Thermal**: Apply thermal to subspaces
3. **Hierarchical Thermal**: Multiple layers of thermal fields at different scales
4. **Hyperbolic Embeddings**: Map to hyperbolic space first, then apply thermal diffusion

## Files Generated

```
thermal-bloom/
├── src/main.rs                    # Rust implementation
├── faiss_benchmark.py             # FAISS comparison
├── faiss_benchmark_v2.py          # Advanced methods
├── visualize.py                   # Plotting script
├── thermal_bloom_v2_results.json  # 2D benchmark results
├── faiss_benchmark_results.json   # High-D results
├── comparison_bar.png             # Method comparison
├── parameter_heatmap.png          # Parameter sensitivity
├── recall_vs_sigma.png            # Sigma analysis
└── speed_accuracy_tradeoff.png    # Pareto frontier
```

## Conclusion

**The Thermal Bloom Filter is the Mona Lisa of 2D spatial indexing** - achieving near-perfect recall (99.6%) with elegant simplicity. For high-dimensional embeddings, stick with FAISS/HNSW.

The core insight - using thermal diffusion to create continuous, navigable gradient fields - remains beautiful and may find applications in domains with inherently low-dimensional structure.
