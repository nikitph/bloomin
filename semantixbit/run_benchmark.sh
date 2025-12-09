#!/bin/bash
# Quick start script for SemantixBit Wikipedia benchmark

set -e

echo "=== SemantixBit Wikipedia Benchmark Setup ==="
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

# Check if Rust is available
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust/Cargo is required but not found"
    exit 1
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -q -r requirements.txt

# Prepare Wikipedia data
echo
echo "📚 Preparing Wikipedia dataset (this may take 10-20 minutes)..."
echo "   - Downloading 100k Wikipedia articles"
echo "   - Chunking into passages"
echo "   - Generating MiniLM embeddings"
echo
python scripts/prepare_wiki_data.py

# Run SemantixBit benchmark
echo
echo "🚀 Running SemantixBit benchmark..."
cargo run --example benchmark_wiki --release

# Run FAISS baseline
echo
echo "📊 Running FAISS baseline for comparison..."
python scripts/baseline_faiss.py

echo
echo "✅ Benchmark complete!"
echo
echo "Results:"
echo "  - SemantixBit: See output above"
echo "  - FAISS: results/faiss_baseline.json"
echo
echo "Next steps:"
echo "  - Compare recall@k between SemantixBit and FAISS"
echo "  - Analyze speed vs. accuracy tradeoffs"
echo "  - Test on your own dataset"
