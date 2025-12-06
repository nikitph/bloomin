"""
Wikipedia benchmark with REAL BERT embeddings

This version loads pre-generated Wikipedia + BERT embeddings
instead of using synthetic data.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from faiss_sphere import FAISSSphere
from faiss_sphere.benchmark import Benchmark


def load_real_embeddings(data_dir='data'):
    """Load pre-generated Wikipedia + BERT embeddings"""
    doc_path = os.path.join(data_dir, 'wikipedia_documents.npy')
    query_path = os.path.join(data_dir, 'wikipedia_queries.npy')
    
    if not os.path.exists(doc_path) or not os.path.exists(query_path):
        print("ERROR: Embeddings not found!")
        print(f"Expected files:")
        print(f"  - {doc_path}")
        print(f"  - {query_path}")
        print("\nRun: python3 generate_wikipedia_embeddings.py")
        sys.exit(1)
    
    print("Loading real Wikipedia + BERT embeddings...")
    documents = np.load(doc_path)
    queries = np.load(query_path)
    
    metadata = {
        'n_documents': len(documents),
        'n_queries': len(queries),
        'dimension': documents.shape[1],
        'source': 'wikipedia+bert'
    }
    
    print(f"Loaded {len(documents)} docs, {len(queries)} queries ({documents.shape[1]}D)")
    return documents, queries, metadata


def compute_ground_truth(documents, queries, k=10):
    """Compute exact nearest neighbors"""
    try:
        import faiss
        
        print("Computing ground truth...")
        index = faiss.IndexFlatIP(documents.shape[1])
        index.add(documents)
        
        distances, indices = index.search(queries, k)
        
        print(f"Ground truth computed: {k} neighbors per query")
        return distances, indices
    except ImportError:
        print("Warning: faiss not installed, using brute force")
        similarities = queries @ documents.T
        indices = np.argsort(-similarities, axis=1)[:, :k]
        distances = np.arccos(np.clip(
            np.array([similarities[i, indices[i]] for i in range(len(queries))]),
            -1.0, 1.0
        ))
        return distances, indices


def run_benchmark():
    """Main benchmark with real embeddings"""
    # Load real embeddings
    documents, queries, metadata = load_real_embeddings()
    
    # Split data
    train_docs = documents[:2000]  # 20% for training
    index_docs = documents
    
    # Compute ground truth
    gt_distances, gt_indices = compute_ground_truth(documents, queries, k=10)
    
    # Initialize benchmark
    bench = Benchmark(f"Wikipedia+BERT {metadata['n_documents']} docs")
    
    # Method 1: FAISS Flat (baseline)
    try:
        import faiss
        
        print("\n" + "="*80)
        print("METHOD 1: FAISS IndexFlatIP (Baseline)")
        print("="*80)
        
        index_flat = faiss.IndexFlatIP(768)
        index_flat.add(documents)
        
        bench.add_method('FAISS Flat', index_flat, queries, gt_indices)
        
        # Method 2: FAISS HNSW
        print("\n" + "="*80)
        print("METHOD 2: FAISS IndexHNSW")
        print("="*80)
        
        index_hnsw = faiss.IndexHNSWFlat(768, 16)
        index_hnsw.metric_type = faiss.METRIC_INNER_PRODUCT
        index_hnsw.add(documents)
        
        bench.add_method('FAISS HNSW', index_hnsw, queries, gt_indices)
        
    except ImportError:
        print("\nWarning: faiss not installed")
    
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
    
    # Method 5: FAISS-Sphere Exact
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
    
    bench.save_results('results/wikipedia_bert_benchmark.json')
    df.to_csv('results/wikipedia_bert_benchmark.csv', index=False)
    
    # Generate LaTeX table
    generate_latex_table(df, 'results/wikipedia_bert_table.tex')
    
    print("\n" + "="*80)
    print("REAL WIKIPEDIA+BERT BENCHMARK COMPLETE!")
    print("="*80)
    print(f"Results saved to:")
    print(f"  - results/wikipedia_bert_benchmark.json")
    print(f"  - results/wikipedia_bert_benchmark.csv")
    print(f"  - results/wikipedia_bert_table.tex")
    
    return df


def generate_latex_table(df, output_path):
    """Generate LaTeX table"""
    latex = r"""\begin{table}[t]
\centering
\caption{FAISS vs FAISS-Sphere: Real Wikipedia+BERT Benchmark (10K documents, 768D)}
\label{tab:wikipedia_bert_benchmark}
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
\textit{Note: Real Wikipedia articles encoded with BERT-base. FAISS-Sphere exploits K=1 spherical geometry with 350D intrinsic projection. Recall should be significantly higher than synthetic data.}
\end{table}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"\nLaTeX table saved to {output_path}")


if __name__ == '__main__':
    run_benchmark()
