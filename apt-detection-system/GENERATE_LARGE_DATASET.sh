#!/bin/bash
#
# Generate Large-Scale Test Dataset (5-10GB)
#
# This script generates a large, realistic test dataset with embedded APT campaigns
# for high-fidelity performance testing.
#
# Usage:
#   ./GENERATE_LARGE_DATASET.sh [size_in_gb] [num_campaigns]
#
# Examples:
#   ./GENERATE_LARGE_DATASET.sh 5 50      # 5GB with 50 campaigns
#   ./GENERATE_LARGE_DATASET.sh 10 100    # 10GB with 100 campaigns
#

# Default values
SIZE=${1:-5.0}
CAMPAIGNS=${2:-50}

echo "======================================================================="
echo "LARGE-SCALE TEST DATASET GENERATOR"
echo "======================================================================="
echo ""
echo "Configuration:"
echo "  Size: ${SIZE} GB"
echo "  APT Campaigns: ${CAMPAIGNS}"
echo "  Output: data/test_logs/large_dataset_${SIZE}gb.json"
echo ""
echo "Estimated time: 15-30 minutes"
echo "Estimated disk space: ${SIZE} GB"
echo ""
echo "======================================================================="
echo ""

# Prompt for confirmation
read -p "Proceed with generation? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Cancelled."
    exit 1
fi

echo ""
echo "Starting generation..."
echo ""

# Set PYTHONPATH
export PYTHONPATH=/Users/truckx/PycharmProjects/bloomin/apt-detection-system

# Generate dataset
python tests/generate_large_dataset.py \
  --size ${SIZE} \
  --campaigns ${CAMPAIGNS} \
  --density 0.001 \
  --output data/test_logs/large_dataset_${SIZE}gb.json

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "SUCCESS!"
    echo "======================================================================="
    echo ""
    echo "Dataset created: data/test_logs/large_dataset_${SIZE}gb.json"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Run high-fidelity test:"
    echo "   PYTHONPATH=. python tests/run_high_fidelity_test.py \\"
    echo "     --dataset data/test_logs/large_dataset_${SIZE}gb.json"
    echo ""
    echo "2. Or run with sample size:"
    echo "   PYTHONPATH=. python tests/run_high_fidelity_test.py \\"
    echo "     --dataset data/test_logs/large_dataset_${SIZE}gb.json \\"
    echo "     --sample 100000"
    echo ""
    echo "======================================================================="
else
    echo ""
    echo "======================================================================="
    echo "ERROR: Dataset generation failed"
    echo "======================================================================="
    exit 1
fi
