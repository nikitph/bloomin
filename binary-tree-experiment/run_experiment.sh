#!/bin/bash
# Run comparison experiment: Standard vs Hyperbolic Branch Transformers
# on binary tree path prediction (depth 20)

echo "=========================================="
echo "Binary Tree Experiment"
echo "Comparing Standard vs Hyperbolic Branch"
echo "=========================================="
echo ""

# Create checkpoints directory
mkdir -p checkpoints

# Train Standard Transformer with different sizes
echo "Training Standard Transformers..."
echo ""

# d=512 (expected to fail)
echo "→ Standard Transformer: d=512"
python3 train.py \
    --model-type standard \
    --d-model 512 \
    --n-layers 6 \
    --n-heads 8 \
    --d-ff 2048 \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --train-samples 10000 \
    --val-samples 2000

# d=1024 (expected to struggle)
echo ""
echo "→ Standard Transformer: d=1024"
python3 train.py \
    --model-type standard \
    --d-model 1024 \
    --n-layers 6 \
    --n-heads 8 \
    --d-ff 4096 \
    --epochs 50 \
    --batch-size 16 \
    --lr 1e-4 \
    --train-samples 10000 \
    --val-samples 2000

# Train Hyperbolic Branch Transformer
echo ""
echo "Training Hyperbolic Branch Transformers..."
echo ""

# d=512, p=256 (expected to succeed!)
echo "→ Hyperbolic Branch: d=512, p=256"
python3 train.py \
    --model-type hyperbolic \
    --d-model 512 \
    --n-layers 6 \
    --n-heads 8 \
    --d-ff 1024 \
    --bucket-sizes 256 64 16 \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --train-samples 10000 \
    --val-samples 2000

# Evaluate and compare
echo ""
echo "Generating comparison plots..."
python3 evaluate.py --results-dir checkpoints --output-plot comparison.png

echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "See comparison.png for results"
echo "=========================================="
