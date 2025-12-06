#!/bin/bash
# Comprehensive experiment: Prove the paradigm shift
# 
# 1. Depth-5 with 20k samples (should get validation accuracy)
# 2. Depth-8 with 20k samples (prove exponential gap)

echo "=========================================="
echo "Comprehensive Binary Tree Experiment"
echo "Proving Hierarchical >> Standard"
echo "=========================================="
echo ""

mkdir -p results_comprehensive

# ============================================
# Depth-5: High Data Regime
# ============================================
echo "PART 1: Depth-5 Trees with 20k Training Samples"
echo "Expected: Both models learn, hierarchical much better"
echo ""

echo "→ Training Standard Transformer (d=128, depth=5, 20k samples)..."
python3 train.py \
    --model-type standard \
    --d-model 128 \
    --n-layers 3 \
    --n-heads 4 \
    --d-ff 256 \
    --depth 5 \
    --epochs 30 \
    --batch-size 64 \
    --train-samples 20000 \
    --val-samples 2000 \
    --lr 5e-4 \
    --save-dir results_comprehensive

echo ""
echo "→ Training Hierarchical Transformer (d=128, depth=5, 20k samples)..."
python3 train.py \
    --model-type hyperbolic \
    --d-model 128 \
    --n-layers 3 \
    --n-heads 4 \
    --d-ff 256 \
    --bucket-sizes 64 16 4 \
    --depth 5 \
    --epochs 30 \
    --batch-size 64 \
    --train-samples 20000 \
    --val-samples 2000 \
    --lr 5e-4 \
    --save-dir results_comprehensive

# ============================================
# Depth-8: Exponential Gap Test
# ============================================
echo ""
echo "PART 2: Depth-8 Trees with 20k Training Samples"
echo "Expected: Standard fails (~0%), Hierarchical succeeds"
echo "This proves the exponential gap!"
echo ""

echo "→ Training Standard Transformer (d=128, depth=8, 20k samples)..."
python3 train.py \
    --model-type standard \
    --d-model 128 \
    --n-layers 3 \
    --n-heads 4 \
    --d-ff 256 \
    --depth 8 \
    --epochs 30 \
    --batch-size 32 \
    --train-samples 20000 \
    --val-samples 2000 \
    --lr 5e-4 \
    --save-dir results_comprehensive

echo ""
echo "→ Training Hierarchical Transformer (d=128, depth=8, 20k samples)..."
python3 train.py \
    --model-type hyperbolic \
    --d-model 128 \
    --n-layers 3 \
    --n-heads 4 \
    --d-ff 256 \
    --bucket-sizes 128 32 8 \
    --depth 8 \
    --epochs 30 \
    --batch-size 32 \
    --train-samples 20000 \
    --val-samples 2000 \
    --lr 5e-4 \
    --save-dir results_comprehensive

# ============================================
# Generate Comparison
# ============================================
echo ""
echo "Generating comparison plots..."
python3 evaluate.py --results-dir results_comprehensive --output-plot comprehensive_comparison.png

echo ""
echo "=========================================="
echo "Comprehensive Experiment Complete!"
echo "See comprehensive_comparison.png"
echo "=========================================="
