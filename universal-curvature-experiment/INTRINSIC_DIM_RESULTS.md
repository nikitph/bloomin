# Intrinsic Dimension Sweep Results (Real Data)

## Key Findings

### Synthetic vs Real Data Comparison

| Model | Synthetic Data (95%) | Real Wikipedia (95%) | Difference |
|-------|---------------------|---------------------|------------|
| BERT-base | 39D | **361D** | **9.3x higher!** |
| RoBERTa | 47D | **236D** | **5.0x higher** |
| DistilBERT | 46D | **264D** | **5.7x higher** |
| GPT-2 | 3D | **12D** | **4.0x higher** |

**Critical Discovery**: Synthetic template-based data produces **artificially low intrinsic dimensions** that don't reflect real-world embedding behavior!

## Full Results (Real Wikipedia Data)

| Model | Full Dim | Intrinsic (95%) | Compression | Effective Rank | 1st PC |
|-------|----------|----------------|-------------|----------------|--------|
| **BERT-base** | 768 | 361 | 2.13x | 105.6 | 18.8% |
| **RoBERTa** | 768 | 236 | 3.25x | 94.9 | 17.9% |
| **DistilBERT** | 768 | 264 | 2.91x | 106.3 | 9.6% |
| **GPT-2** | 768 | 12 | 64.0x | 4.4 | 69.3% |
| **CodeBERT** | 768 | 8 | 96.0x | 6.3 | 35.9% |
| **SciBERT** | 768 | 4 | 192.0x | 3.6 | 41.9% |

## Insights

### 1. **Real Data Matters**
- Synthetic templates create repetitive patterns
- Real Wikipedia sentences are much more diverse
- Intrinsic dimension measurements on synthetic data are **not representative**

### 2. **Model Architecture Differences**
- **Encoder models (BERT, RoBERTa, DistilBERT)**: High intrinsic dimension (236-361D)
  - Capture rich contextual information
  - More distributed representations
- **Decoder model (GPT-2)**: Low intrinsic dimension (12D)
  - More concentrated variance
  - First PC explains 69% of variance!

### 3. **Domain-Specific Models**
- **CodeBERT & SciBERT**: Very low intrinsic dimension (4-8D)
  - Note: These used synthetic fallback data due to HuggingFace dataset deprecation
  - Real code/science data would likely show higher dimensions

### 4. **Compression Potential**
- Encoder models: 2-5x compression at 95% variance
- GPT-2: 64x compression (highly compressible!)
- Domain-specific: 96-192x (but synthetic data caveat)

## Recommendations

1. **Always use real data** for intrinsic dimension analysis
2. **GPT-2's low dimensionality** is interesting - worth investigating why decoder models concentrate variance more
3. **Update code/science loaders** to work with newer HuggingFace datasets API
4. **Consider testing on multiple datasets** per model to see variance

## Files Generated
- `results/intrinsic_dim_results.csv` - Full results table
- `results/intrinsic_dim_summary.json` - Summary statistics
- `results/intrinsic_dim_comparison.png` - Comparison plots
- `results/variance_spectra.png` - Variance curves
