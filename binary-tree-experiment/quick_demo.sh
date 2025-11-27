#!/bin/bash
# Quick demo: Train a small hyperbolic model to verify it works

echo "🚀 Quick Demo: Hyperbolic Branch Transformer"
echo "=============================================="
echo ""
echo "Training a small model on depth-10 trees..."
echo "(This is faster than the full depth-20 experiment)"
echo ""

python3 train.py \
    --model-type hyperbolic \
    --d-model 256 \
    --n-layers 4 \
    --n-heads 4 \
    --d-ff 512 \
    --bucket-sizes 128 32 8 \
    --depth 10 \
    --epochs 20 \
    --batch-size 64 \
    --train-samples 2000 \
    --val-samples 500 \
    --lr 3e-4

echo ""
echo "✅ Demo complete! Check the results above."
echo ""
echo "To run the full experiment (depth-20), use:"
echo "  ./run_experiment.sh"
