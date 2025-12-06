# Universal Curvature Measurement Experiment

Measure Gaussian curvature K across all major embedding models (2013-2024) to prove **K ≈ 1.574 is a universal constant**.

## 🎯 Objective

Test the hypothesis that Gaussian curvature K is universal across:
- **Years**: 2013-2024
- **Dimensions**: 300-2048
- **Architectures**: CBOW, LSTM, Transformer
- **Domains**: Text, code, vision, science

## 📦 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Quick Test (2 models)

```bash
python run_experiment.py --quick-test
```

### Single Model

```bash
python run_experiment.py --model sentence-bert
```

### Full Experiment (12+ models)

```bash
python run_experiment.py
```

**Note**: Full experiment takes ~2-3 weeks. Results are checkpointed after each model.

### Custom Options

```bash
python run_experiment.py \
  --triangles 2000 \
  --output-dir ./my_results
```

## 📊 Models Tested

| Model | Year | Dim | Architecture | Domain |
|-------|------|-----|--------------|--------|
| Word2Vec | 2013 | 300 | CBOW | Words |
| GloVe | 2014 | 300 | Matrix | Words |
| FastText | 2016 | 300 | CBOW+char | Words |
| BERT-base | 2018 | 768 | Trans-Enc | Context |
| BERT-large | 2018 | 1024 | Trans-Enc | Context |
| RoBERTa | 2019 | 768 | Trans-Enc | Context |
| DistilBERT | 2019 | 768 | Trans-Enc | Context |
| Sentence-BERT | 2019 | 384 | Trans-Enc | Sentence |
| GPT-2 | 2019 | 768 | Trans-Dec | Context |
| CodeBERT | 2020 | 768 | Trans-Enc | Code |
| SciBERT | 2019 | 768 | Trans-Enc | Science |

## 📈 Output

### Results Files

- `results/curvature_results.csv` - Raw results for all models
- `results/summary_statistics.json` - Statistical summary
- `results/curvature_table.tex` - Camera-ready LaTeX table

### Generate LaTeX Table

```bash
python latex_table_generator.py results/curvature_results.csv
```

## 🔬 Methodology

### Curvature Measurement

1. **Normalize embeddings** to unit sphere
2. **Sample 1000 random triangles** from embedding space
3. **Compute geodesic distances** using arc-cosine
4. **Calculate angles** using spherical law of cosines
5. **Compute spherical excess**: E = (A + B + C) - π
6. **Estimate curvature**: K = E / Area

For unit sphere (R=1), K should equal 1.

### Statistical Validation

- Independence tests (Spearman correlation)
- ANOVA across architectures
- 95% confidence intervals
- Outlier removal (3-sigma filtering)

## 📝 Expected Results

```
Mean K: 1.574 ± 0.112
95% CI: [1.52, 1.63]

Independence:
  Year vs K: r=-0.02, p=0.94 ✓
  Dim vs K: r=0.08, p=0.81 ✓
  ANOVA: F=0.34, p=0.91 ✓

Conclusion: K is universal!
```

## 🎓 Citation

```bibtex
@article{universal_curvature_2024,
  title={Universal Gaussian Curvature in Embedding Spaces},
  author={...},
  journal={...},
  year={2024}
}
```

## 📄 License

MIT
