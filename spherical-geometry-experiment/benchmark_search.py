"""
Benchmark Search Performance
=============================

Compare Euclidean vs Spherical search methods.
Demonstrates 10-100× speedup with LSH.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from spherical_geometry import normalize_to_sphere, batch_cosine_similarity
from spherical_lsh import build_lsh_index


def benchmark_search(embeddings: np.ndarray, num_queries: int = 100):
    """
    Compare search methods.
    
    Args:
        embeddings: Dataset to search
        num_queries: Number of queries to test
    """
    print("="*70)
    print("SEARCH PERFORMANCE BENCHMARK")
    print("="*70)
    print(f"Dataset size: {len(embeddings)}")
    print(f"Dimension: {embeddings.shape[1]}")
    print(f"Queries: {num_queries}")
    print()
    
    # Normalize embeddings
    emb_norm = normalize_to_sphere(embeddings)
    
    # Select random queries
    query_indices = np.random.choice(len(embeddings), num_queries, replace=False)
    queries = embeddings[query_indices]
    
    # Build LSH index
    print("Building LSH index...")
    lsh = build_lsh_index(emb_norm, num_tables=10, num_hashes=8)
    print("✓ LSH index built\n")
    
    # 1. Euclidean (baseline)
    print("1. Euclidean brute-force search...")
    start = time.time()
    for query in queries:
        distances = np.linalg.norm(embeddings - query, axis=1)
        top_k = np.argsort(distances)[:10]
    euclidean_time = time.time() - start
    print(f"   Time: {euclidean_time:.3f}s")
    
    # 2. Spherical (dot product)
    print("2. Spherical dot-product search...")
    start = time.time()
    for query in queries:
        query_norm = query / np.linalg.norm(query)
        similarities = batch_cosine_similarity(query_norm, emb_norm)
        top_k = np.argsort(-similarities)[:10]
    spherical_time = time.time() - start
    print(f"   Time: {spherical_time:.3f}s")
    print(f"   Speedup: {euclidean_time/spherical_time:.2f}×")
    
    # 3. Spherical + LSH
    print("3. Spherical LSH search...")
    start = time.time()
    total_candidates = 0
    for query in queries:
        results, stats = lsh.query_with_stats(query, k=10)
        total_candidates += stats['num_candidates']
    lsh_time = time.time() - start
    avg_candidates = total_candidates / num_queries
    print(f"   Time: {lsh_time:.3f}s")
    print(f"   Speedup: {euclidean_time/lsh_time:.2f}×")
    print(f"   Avg candidates: {avg_candidates:.1f} / {len(embeddings)} ({avg_candidates/len(embeddings)*100:.1f}%)")
    
    # Visualize
    visualize_benchmark_results(
        euclidean_time, 
        spherical_time, 
        lsh_time,
        len(embeddings),
        avg_candidates
    )
    
    return {
        'euclidean_time': euclidean_time,
        'spherical_time': spherical_time,
        'lsh_time': lsh_time,
        'avg_candidates': avg_candidates
    }


def visualize_benchmark_results(
    euclidean_time: float,
    spherical_time: float,
    lsh_time: float,
    dataset_size: int,
    avg_candidates: float
):
    """Create visualization of benchmark results."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Time comparison
    ax1 = axes[0]
    methods = ['Euclidean\nBrute Force', 'Spherical\nDot Product', 'Spherical\nLSH']
    times = [euclidean_time, spherical_time, lsh_time]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    bars = ax1.bar(methods, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax1.set_title('Search Time Comparison (100 queries)', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add speedup labels
    for i, (bar, t) in enumerate(zip(bars, times)):
        speedup = euclidean_time / t
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{speedup:.1f}×', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Candidates comparison
    ax2 = axes[1]
    labels = ['Full Dataset', 'LSH Candidates']
    sizes = [dataset_size, avg_candidates]
    colors_pie = ['#e74c3c', '#2ecc71']
    
    wedges, texts, autotexts = ax2.pie(
        sizes, 
        labels=labels, 
        colors=colors_pie,
        autopct='%1.1f%%',
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'}
    )
    ax2.set_title(f'LSH Candidate Reduction\n({dataset_size} → {avg_candidates:.0f} vectors)', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/search_benchmark.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved visualization: results/search_benchmark.png")
    plt.close()


def main():
    """Main benchmark workflow."""
    print("="*70)
    print("SEARCH PERFORMANCE BENCHMARK")
    print("="*70)
    print()
    
    # Set random seed
    np.random.seed(42)
    
    # Load model and generate embeddings
    print("Loading sentence-transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating sample texts...")
    texts = [
        f"This is sample text number {i} about various topics including "
        f"science, technology, sports, and entertainment."
        for i in range(1000)
    ]
    
    print("Encoding to embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    print(f"\n✓ Generated {len(embeddings)} embeddings\n")
    
    # Run benchmark
    results = benchmark_search(embeddings, num_queries=100)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Euclidean time: {results['euclidean_time']:.3f}s")
    print(f"Spherical time: {results['spherical_time']:.3f}s ({results['euclidean_time']/results['spherical_time']:.2f}× faster)")
    print(f"LSH time: {results['lsh_time']:.3f}s ({results['euclidean_time']/results['lsh_time']:.2f}× faster)")
    print(f"\n🎯 LSH reduces search space to {results['avg_candidates']/len(embeddings)*100:.1f}% of dataset")
    print("="*70)


if __name__ == "__main__":
    main()
