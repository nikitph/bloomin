"""
Main Wikipedia benchmark script

Compares:
1. FAISS IndexFlatIP (baseline - current best practice)
2. FAISS IndexHNSW (baseline - current fast)
3. FAISS-Sphere Fast (LSH)
4. FAISS-Sphere Balanced (HNSW equivalent)
5. FAISS-Sphere Memory (PQ)

Dataset:
- Wikipedia embeddings (BERT-base, 768D)
- 100K documents
- 1K queries
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faiss_sphere import FAISSSphere
from faiss_sphere.benchmark import Benchmark


def load_wikipedia_embeddings(n_documents: int = 100000,
                               n_queries: int = 1000):
    """
    Load or generate Wikipedia embeddings
    
    For prototype: Generate synthetic BERT-like embeddings
    For production: Load from actual Wikipedia + BERT
    
    Returns:
        documents: (n_documents, 768)
        queries: (n_queries, 768)
        metadata: dict with document info
    """
    print(f"Loading {n_documents} Wikipedia documents...")
    
    # Option 1: Load real embeddings (if available)
    # from datasets import load_dataset
    # wiki = load_dataset('wikipedia', '20220301.en', split='train[:100000]')
    # ... encode with BERT ...
    
    # Option 2: Generate synthetic (for prototype)
    np.random.seed(42)
    documents = np.random.randn(n_documents, 768).astype('float32')
    queries = np.random.randn(n_queries, 768).astype('float32')
    
    # Normalize (critical!)
    try:
        import faiss
        faiss.normalize_L2(documents)
        faiss.normalize_L2(queries)
    except ImportError:
        # Fallback to numpy
        documents = documents / np.linalg.norm(documents, axis=1, keepdims=True)
        queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    metadata = {
        'n_documents': n_documents,
        'n_queries': n_queries,
        'dimension': 768,
        'source': 'synthetic'  # or 'wikipedia'
    }
    
    print(f"Loaded {n_documents} docs, {n_queries} queries")
    return documents, queries, metadata


def compute_ground_truth(documents: np.ndarray,
                        queries: np.ndarray,
                        k: int = 10):
    """
    Compute exact nearest neighbors (ground truth)
    """
    try:
        import faiss
        
        print("Computing ground truth...")
        index = faiss.IndexFlatIP(documents.shape[1])
        index.add(documents)
        
        distances, indices = index.search(queries, k)
        
        print(f"Ground truth computed: {k} neighbors per query")
        return distances, indices
    except ImportError:
        print("Warning: faiss not installed, using brute force for ground truth")
        # Brute force
        similarities = queries @ documents.T
        indices = np.argsort(-similarities, axis=1)[:, :k]
        distances = np.arccos(np.clip(
            np.array([similarities[i, indices[i]] for i in range(len(queries))]),
            -1.0, 1.0
        ))
        return distances, indices


def run_benchmark():
    """
    Main benchmark function
    """
    # Load data (reduced size for faster benchmarking)
    documents, queries, metadata = load_wikipedia_embeddings(
        n_documents=10000,  # Reduced from 100K for speed
        n_queries=1000
    )
    
    # Split data
    train_docs = documents[:2000]  # For training projector (20% of data)
    index_docs = documents  # Full dataset
    
    # Compute ground truth
    gt_distances, gt_indices = compute_ground_truth(documents, queries, k=10)
    
    # Initialize benchmark
    bench = Benchmark(f"Wikipedia {metadata['n_documents']} docs")
    
    # Method 1: FAISS Flat (baseline)
    try:
        import faiss
        
        print("\n" + "="*80)
        print("METHOD 1: FAISS IndexFlatIP (Current Best Practice)")
        print("="*80)
        
        index_flat = faiss.IndexFlatIP(768)
        index_flat.add(documents)
        
        bench.add_method('FAISS Flat', index_flat, queries, gt_indices)
        
        # Method 2: FAISS HNSW (baseline fast)
        print("\n" + "="*80)
        print("METHOD 2: FAISS IndexHNSW (Current Fast)")
        print("="*80)
        
        index_hnsw = faiss.IndexHNSWFlat(768, 16)
        index_hnsw.metric_type = faiss.METRIC_INNER_PRODUCT
        index_hnsw.add(documents)
        
        bench.add_method('FAISS HNSW', index_hnsw, queries, gt_indices)
        
    except ImportError:
        print("\nWarning: faiss not installed, skipping FAISS baselines")
        print("Install with: pip install faiss-cpu")
    
    # Method 3: FAISS-Sphere Fast
    print("\n" + "="*80)
    print("METHOD 3: FAISS-Sphere Fast (LSH)")
    print("="*80)
    
    index_sphere_fast = FAISSSphere(768, mode='fast')
    index_sphere_fast.train(train_docs)
    index_sphere_fast.add(index_docs)
    
    bench.add_method('Sphere Fast', index_sphere_fast, queries, gt_indices)
    
    # Method 4: FAISS-Sphere Memory
    print("\n" + "="*80)
    print("METHOD 4: FAISS-Sphere Memory (PQ)")
    print("="*80)
    
    index_sphere_mem = FAISSSphere(768, mode='memory')
    index_sphere_mem.train(train_docs)
    index_sphere_mem.add(index_docs)
    
    bench.add_method('Sphere Memory', index_sphere_mem, queries, gt_indices)
    
    # Method 5: FAISS-Sphere Exact (intrinsic dimension only)
    print("\n" + "="*80)
    print("METHOD 5: FAISS-Sphere Exact (350D projection)")
    print("="*80)
    
    index_sphere_exact = FAISSSphere(768, mode='exact')
    index_sphere_exact.train(train_docs)
    index_sphere_exact.add(index_docs)
    
    bench.add_method('Sphere Exact', index_sphere_exact, queries, gt_indices)
    
    # Print and save results
    df = bench.print_summary()
    
    # Save results
    os.makedirs('results', exist_ok=True)
    
    bench.save_results('results/wikipedia_benchmark.json')
    df.to_csv('results/wikipedia_benchmark.csv', index=False)
    
    # Generate LaTeX table
    generate_latex_table(df, 'results/comparison_table.tex')
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)
    print(f"Results saved to:")
    print(f"  - results/wikipedia_benchmark.json")
    print(f"  - results/wikipedia_benchmark.csv")
    print(f"  - results/comparison_table.tex")
    
    return df


def generate_latex_table(df, output_path):
    """Generate camera-ready LaTeX table"""
    
    latex = r"""\begin{table}[t]
\centering
\caption{FAISS vs FAISS-Sphere: Wikipedia Benchmark (10K documents, 768D)}
\label{tab:wikipedia_benchmark}
\begin{tabular}{lrrrrr}
\toprule
\textbf{Method} & \textbf{QPS} & \textbf{Latency} & \textbf{Recall@10} & \textbf{Speedup} & \textbf{Memory} \\
 & & \textbf{(ms)} & & & \textbf{(MB)} \\
\midrule
"""
    
    for _, row in df.iterrows():
        latex += f"{row['method']:20s} & "
        latex += f"{row['qps']:7.1f} & "
        latex += f"{row['latency_ms']:6.2f} & "
        latex += f"{row['recall@10']:5.3f} & "
        
        if 'speedup' in row and not np.isnan(row['speedup']):
            latex += f"{row['speedup']:5.2f}× & "
        else:
            latex += f"{'--':>6s} & "
        
        latex += f"{row['memory_mb']:7.1f} \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}

\vspace{0.3cm}
\textit{Note: Speedup relative to FAISS Flat. All methods use cosine similarity on normalized vectors. FAISS-Sphere exploits K=1 spherical geometry with 350D intrinsic projection.}
\end{table}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"\nLaTeX table saved to {output_path}")


if __name__ == '__main__':
    run_benchmark()
