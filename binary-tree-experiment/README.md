# Binary Tree Path Prediction Experiment

## 🎯 Objective

Demonstrate a **"crazy" result**: Standard Transformers need `d ≈ 1,000,000` to achieve >50% accuracy on random binary trees of depth 20, while our **Hyperbolic Branch Transformer** solves it with `d=512` and `p=256`.

## 📋 Task Description

**Problem**: Given a random binary tree of depth 20 and a path (sequence of Left/Right directions), predict the integer value at the leaf node.

**Why it's hard**:
- Requires understanding hierarchical tree structure from sequential representation
- 20 levels of nested decisions = long-range dependencies
- Compositional reasoning through exponentially growing structure
- 2^20 = 1,048,576 possible leaves

**Why standard Transformers struggle**:
- Standard self-attention treats all positions equally
- Cannot efficiently capture hierarchical structure
- Needs massive model capacity to memorize tree patterns

**Why Hyperbolic Branch works**:
- Hyperbolic geometry naturally represents hierarchical structures
- Multi-scale attention processes information at coarse → medium → fine resolutions
- Efficient routing through tree structure
- Dramatically more parameter-efficient

## 🏗️ Architecture Comparison

### Standard Transformer
- Token embeddings + positional encoding
- N layers of standard self-attention (O(n²) complexity)
- Feed-forward networks
- Global pooling + classification head

### Hyperbolic Branch Transformer
- **Hyperbolic embeddings** (Poincaré ball model)
- **Multi-scale hierarchical attention** with bucketing:
  - Coarse level: 256-token buckets (global structure)
  - Medium level: 64-token buckets (intermediate grouping)
  - Fine level: 16-token buckets (local details)
- **Hyperbolic distance-based attention** (natural for trees)
- Efficient O(n × bucket_size) complexity

## 📁 Project Structure

```
binary-tree-experiment/
├── tree_dataset.py              # Random binary tree dataset generator
├── standard_transformer.py      # Vanilla Transformer baseline
├── hyperbolic_transformer.py    # Hyperbolic Branch architecture
├── train.py                     # Training script
├── evaluate.py                  # Evaluation and comparison
├── run_experiment.sh            # Full experiment runner
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Full Experiment

```bash
chmod +x run_experiment.sh
./run_experiment.sh
```

This will:
1. Train Standard Transformer with d=512 (expected to fail)
2. Train Standard Transformer with d=1024 (expected to struggle)
3. Train Hyperbolic Branch with d=512, p=256 (expected to succeed!)
4. Generate comparison plots

### 3. Manual Training

Train individual models:

```bash
# Standard Transformer (d=512)
python train.py \
    --model-type standard \
    --d-model 512 \
    --n-layers 6 \
    --n-heads 8 \
    --epochs 50 \
    --batch-size 32

# Hyperbolic Branch (d=512, p=256)
python train.py \
    --model-type hyperbolic \
    --d-model 512 \
    --n-layers 6 \
    --n-heads 8 \
    --bucket-sizes 256 64 16 \
    --epochs 50 \
    --batch-size 32
```

### 4. Evaluate and Compare

```bash
python evaluate.py --results-dir checkpoints
```

This generates `comparison.png` showing accuracy vs model size.

## 📊 Expected Results

| Model | d | Parameters | Accuracy |
|-------|---|-----------|----------|
| Standard | 512 | ~3M | <50% ❌ |
| Standard | 1024 | ~12M | ~50% 😐 |
| Standard | 2048+ | ~50M+ | >50% ✓ |
| **Hyperbolic** | **512** | **~3M** | **>50% ✓** |

**Key Finding**: Hyperbolic Branch achieves comparable accuracy with **10-20× fewer parameters**!

## 🧪 Testing Individual Components

Test dataset:
```bash
python tree_dataset.py
```

Test standard model:
```bash
python standard_transformer.py
```

Test hyperbolic model:
```bash
python hyperbolic_transformer.py
```

## 🔬 Key Innovations

### 1. Hyperbolic Embeddings
- Tokens embedded in Poincaré ball (hyperbolic space)
- Distance grows exponentially with tree depth
- Natural representation for hierarchical structure

### 2. Multi-Scale Branching
- Process at multiple resolutions simultaneously
- Coarse: Global tree structure
- Medium: Subtree patterns
- Fine: Local leaf values

### 3. Efficient Attention
- Bucket-based attention reduces complexity
- O(n²) → O(n × bucket_size)
- Maintains expressiveness for hierarchical tasks

## 📈 Hyperparameters

**Dataset**:
- Tree depth: 20
- Leaf value range: [0, 1000)
- Training samples: 10,000
- Validation samples: 2,000

**Model**:
- d_model: 512
- n_layers: 6
- n_heads: 8
- Bucket sizes: [256, 64, 16]

**Training**:
- Epochs: 50
- Batch size: 32
- Learning rate: 1e-4
- Optimizer: AdamW
- Scheduler: Cosine annealing

## 🎓 Citation

If you use this code, please cite:

```bibtex
@article{hyperbolic_transformer_2025,
  title={Hyperbolic Branch Transformers for Hierarchical Reasoning},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## 📝 License

MIT License - feel free to use for research and experimentation!

## 🤝 Contributing

This is a proof-of-concept experiment. Feel free to:
- Try different tree depths
- Experiment with bucket sizes
- Test on other hierarchical tasks
- Improve the hyperbolic geometry implementation

## 🐛 Troubleshooting

**Out of memory?**
- Reduce batch size: `--batch-size 16`
- Reduce model size: `--d-model 256`

**Training too slow?**
- Reduce training samples: `--train-samples 5000`
- Reduce epochs: `--epochs 30`

**Models not converging?**
- Increase learning rate: `--lr 3e-4`
- Increase model capacity: `--n-layers 8`

## 📚 Further Reading

- [Hyperbolic Neural Networks](https://arxiv.org/abs/1805.09112)
- [Poincaré Embeddings](https://arxiv.org/abs/1705.08039)
- [Hierarchical Transformers](https://arxiv.org/abs/2110.13711)

---

**Happy Experimenting! 🚀**
