# SemantixBit Project Structure

```
semantixbit/
├── src/                          # Core Rust library
│   ├── lib.rs                   # Main library interface & config
│   ├── quantizer.rs             # REWA quantization (SimHash)
│   ├── storage.rs               # Binary index storage
│   ├── search.rs                # Search engine (Hamming distance)
│   └── hybrid.rs                # Hybrid monoid encoder
│
├── examples/                     # Runnable examples
│   └── benchmark_wiki.rs        # Wikipedia benchmark
│
├── scripts/                      # Python utilities
│   ├── prepare_wiki_data.py    # Download & prepare Wikipedia
│   └── baseline_faiss.py       # FAISS baseline comparison
│
├── data/                         # Generated data (not in git)
│   └── wikipedia/
│       ├── embeddings.npy       # MiniLM embeddings
│       ├── passages.json        # Text passages
│       ├── metadata.json        # Document metadata
│       └── info.json            # Dataset info
│
├── results/                      # Benchmark results (not in git)
│   └── faiss_baseline.json     # FAISS performance metrics
│
├── Cargo.toml                   # Rust dependencies
├── requirements.txt             # Python dependencies
├── run_benchmark.sh             # Quick start script
└── README.md                    # Documentation
```

## Key Files

### Core Implementation

- **[quantizer.rs](file:///Users/truckx/PycharmProjects/bloomin/semantixbit/src/quantizer.rs)**: REWA quantizer implementing random projection and binarization
- **[storage.rs](file:///Users/truckx/PycharmProjects/bloomin/semantixbit/src/storage.rs)**: Compact binary index with serialization
- **[search.rs](file:///Users/truckx/PycharmProjects/bloomin/semantixbit/src/search.rs)**: Parallel search engine using Hamming distance
- **[hybrid.rs](file:///Users/truckx/PycharmProjects/bloomin/semantixbit/src/hybrid.rs)**: Compositional semantic + keyword search

### Benchmarking

- **[benchmark_wiki.rs](file:///Users/truckx/PycharmProjects/bloomin/semantixbit/examples/benchmark_wiki.rs)**: Rust benchmark testing different bit depths
- **[prepare_wiki_data.py](file:///Users/truckx/PycharmProjects/bloomin/semantixbit/scripts/prepare_wiki_data.py)**: Wikipedia dataset preparation
- **[baseline_faiss.py](file:///Users/truckx/PycharmProjects/bloomin/semantixbit/scripts/baseline_faiss.py)**: FAISS baseline for comparison

## Running the Project

### Quick Start
```bash
./run_benchmark.sh
```

### Manual Steps
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data
python scripts/prepare_wiki_data.py

# 3. Run benchmark
cargo run --example benchmark_wiki --release

# 4. Compare with FAISS
python scripts/baseline_faiss.py
```

### Development
```bash
# Run tests
cargo test

# Build library
cargo build --release

# Check documentation
cargo doc --open
```
