#!/usr/bin/env python3
"""
Benchmark: Witness-LDPC vs FAISS

This is the "Flash Retriever" experiment - proving that Witness-LDPC codes
can match FAISS accuracy with dramatically better speed and memory.

Datasets:
1. Random embeddings (controlled baseline)
2. Sentence embeddings from sentence-transformers (realistic)

Metrics:
- Query latency (ms)
- Memory usage (MB)
- Recall@K (accuracy)
- Build time (s)

Expected Results:
- Witness-LDPC: 50-100x faster queries, 50-100x less memory
- FAISS: Better accuracy (but Witness-LDPC ~95% is good enough)
"""

import numpy as np
import time
import sys
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from witness_codes import WitnessLDPC, WitnessLDPCCompact


@dataclass
class BenchmarkResult:
    """Stores results for a single benchmark run."""
    method: str
    n_vectors: int
    dim: int
    build_time_s: float
    memory_mb: float
    avg_query_time_ms: float
    p50_query_time_ms: float
    p99_query_time_ms: float
    recall_at_1: float
    recall_at_10: float
    recall_at_100: float
    params: Dict


def compute_exact_neighbors(vectors: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Compute exact k-nearest neighbors using brute force."""
    print(f"  Computing exact neighbors (brute force)...")

    # Normalize for cosine similarity
    vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)

    n_queries = len(queries)
    exact_neighbors = np.zeros((n_queries, k), dtype=np.int64)

    batch_size = 100
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        batch_queries = queries_norm[start:end]

        # Compute all similarities
        similarities = batch_queries @ vectors_norm.T  # (batch, n_vectors)

        # Get top-k
        top_k_indices = np.argsort(similarities, axis=1)[:, -k:][:, ::-1]
        exact_neighbors[start:end] = top_k_indices

    return exact_neighbors


def compute_recall(retrieved: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """Compute recall@k."""
    n_queries = len(retrieved)
    recalls = []

    for i in range(n_queries):
        gt_set = set(ground_truth[i, :k])
        ret_set = set(retrieved[i, :k])
        overlap = len(gt_set & ret_set)
        recalls.append(overlap / k)

    return np.mean(recalls)


def benchmark_witness_ldpc(
    vectors: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    code_length: int = 2048,
    num_hashes: int = 4,
    num_witnesses: int = 64,
    compact: bool = False
) -> BenchmarkResult:
    """Benchmark Witness-LDPC index."""
    n, dim = vectors.shape
    n_queries = len(queries)

    print(f"\n[Witness-LDPC] code_length={code_length}, num_hashes={num_hashes}, num_witnesses={num_witnesses}")
    if compact:
        print("  (COMPACT mode - no vector storage)")

    # Build index
    if compact:
        index = WitnessLDPCCompact(
            dim=dim,
            code_length=code_length,
            num_hashes=num_hashes,
            num_witnesses=num_witnesses
        )
    else:
        index = WitnessLDPC(
            dim=dim,
            code_length=code_length,
            num_hashes=num_hashes,
            num_witnesses=num_witnesses
        )

    start = time.time()
    index.add(vectors, verbose=True)
    build_time = time.time() - start

    # Memory
    mem = index.memory_usage()
    memory_mb = mem['codes_mb'] + mem['inverted_index_mb']
    if not compact:
        memory_mb = mem['total_mb']

    # Query timing
    print(f"  Running {n_queries} queries...")
    query_times = []
    all_results = np.zeros((n_queries, 100), dtype=np.int64)

    for i, query in enumerate(queries):
        start = time.time()
        indices, scores = index.search(query, k=100, rerank=not compact)
        query_times.append((time.time() - start) * 1000)  # ms

        n_results = min(len(indices), 100)
        all_results[i, :n_results] = indices[:n_results]

    # Compute recalls
    recall_1 = compute_recall(all_results, ground_truth, k=1)
    recall_10 = compute_recall(all_results, ground_truth, k=10)
    recall_100 = compute_recall(all_results, ground_truth, k=100)

    result = BenchmarkResult(
        method="Witness-LDPC" + (" (Compact)" if compact else ""),
        n_vectors=n,
        dim=dim,
        build_time_s=build_time,
        memory_mb=memory_mb,
        avg_query_time_ms=np.mean(query_times),
        p50_query_time_ms=np.percentile(query_times, 50),
        p99_query_time_ms=np.percentile(query_times, 99),
        recall_at_1=recall_1,
        recall_at_10=recall_10,
        recall_at_100=recall_100,
        params={
            'code_length': code_length,
            'num_hashes': num_hashes,
            'num_witnesses': num_witnesses,
            'compact': compact
        }
    )

    print(f"  Build: {build_time:.2f}s, Memory: {memory_mb:.1f}MB")
    print(f"  Query: avg={np.mean(query_times):.2f}ms, p50={np.percentile(query_times, 50):.2f}ms, p99={np.percentile(query_times, 99):.2f}ms")
    print(f"  Recall: @1={recall_1:.3f}, @10={recall_10:.3f}, @100={recall_100:.3f}")

    return result


def benchmark_faiss(
    vectors: np.ndarray,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    index_type: str = 'flat'
) -> BenchmarkResult:
    """Benchmark FAISS index."""
    try:
        import faiss
    except ImportError:
        print("\n[FAISS] Not installed. Install with: pip install faiss-cpu")
        return None

    n, dim = vectors.shape
    n_queries = len(queries)

    print(f"\n[FAISS] index_type={index_type}")

    vectors = vectors.astype(np.float32)
    queries = queries.astype(np.float32)

    # Normalize for cosine similarity
    faiss.normalize_L2(vectors)
    faiss.normalize_L2(queries)

    # Build index
    start = time.time()

    if index_type == 'flat':
        # Exact search
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
    elif index_type == 'ivf':
        # IVF with 100 clusters
        nlist = min(100, n // 10)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
        index.add(vectors)
        index.nprobe = 10  # Search 10 clusters
    elif index_type == 'ivf_pq':
        # IVF with Product Quantization
        nlist = min(100, n // 10)
        m = 8  # Number of subvectors
        nbits = 8  # Bits per subvector
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)
        index.train(vectors)
        index.add(vectors)
        index.nprobe = 10
    elif index_type == 'hnsw':
        # HNSW graph-based
        M = 16  # Number of neighbors
        index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
        index.add(vectors)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    build_time = time.time() - start

    # Memory (approximate)
    if index_type == 'flat':
        memory_mb = vectors.nbytes / 1024**2
    elif index_type == 'ivf':
        memory_mb = vectors.nbytes / 1024**2 * 1.1  # Overhead
    elif index_type == 'ivf_pq':
        # PQ: m bytes per vector + centroids
        memory_mb = (n * 8 + n * dim * 0.1) / 1024**2
    elif index_type == 'hnsw':
        # HNSW: vectors + graph
        memory_mb = vectors.nbytes / 1024**2 * 2

    # Query timing
    print(f"  Running {n_queries} queries...")
    query_times = []
    all_results = np.zeros((n_queries, 100), dtype=np.int64)

    for i in range(n_queries):
        query = queries[i:i+1]
        start = time.time()
        D, I = index.search(query, 100)
        query_times.append((time.time() - start) * 1000)
        all_results[i] = I[0]

    # Compute recalls
    recall_1 = compute_recall(all_results, ground_truth, k=1)
    recall_10 = compute_recall(all_results, ground_truth, k=10)
    recall_100 = compute_recall(all_results, ground_truth, k=100)

    result = BenchmarkResult(
        method=f"FAISS-{index_type}",
        n_vectors=n,
        dim=dim,
        build_time_s=build_time,
        memory_mb=memory_mb,
        avg_query_time_ms=np.mean(query_times),
        p50_query_time_ms=np.percentile(query_times, 50),
        p99_query_time_ms=np.percentile(query_times, 99),
        recall_at_1=recall_1,
        recall_at_10=recall_10,
        recall_at_100=recall_100,
        params={'index_type': index_type}
    )

    print(f"  Build: {build_time:.2f}s, Memory: {memory_mb:.1f}MB")
    print(f"  Query: avg={np.mean(query_times):.2f}ms, p50={np.percentile(query_times, 50):.2f}ms, p99={np.percentile(query_times, 99):.2f}ms")
    print(f"  Recall: @1={recall_1:.3f}, @10={recall_10:.3f}, @100={recall_100:.3f}")

    return result


def generate_clustered_data(n_vectors: int, dim: int, n_clusters: int = 100) -> np.ndarray:
    """Generate clustered data (more realistic than uniform random)."""
    vectors_per_cluster = n_vectors // n_clusters

    vectors = []
    for i in range(n_clusters):
        # Random cluster center
        center = np.random.randn(dim) * 5
        # Samples around center
        samples = center + np.random.randn(vectors_per_cluster, dim) * 0.5
        vectors.append(samples)

    vectors = np.vstack(vectors).astype(np.float32)

    # Shuffle
    np.random.shuffle(vectors)

    # Normalize
    vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)

    return vectors


def print_summary_table(results: List[BenchmarkResult]):
    """Print a nicely formatted summary table."""
    print("\n" + "="*100)
    print("BENCHMARK SUMMARY")
    print("="*100)

    print(f"\n{'Method':<25} {'Memory (MB)':>12} {'Query (ms)':>12} {'Recall@1':>10} {'Recall@10':>10} {'Recall@100':>10}")
    print("-"*100)

    for r in results:
        if r is None:
            continue
        print(f"{r.method:<25} {r.memory_mb:>12.1f} {r.avg_query_time_ms:>12.3f} {r.recall_at_1:>10.3f} {r.recall_at_10:>10.3f} {r.recall_at_100:>10.3f}")

    print("-"*100)

    # Compute speedups vs FAISS-flat
    faiss_flat = next((r for r in results if r and 'FAISS-flat' in r.method), None)
    if faiss_flat:
        print("\nSpeedups vs FAISS-flat:")
        for r in results:
            if r is None or 'FAISS-flat' in r.method:
                continue
            speedup = faiss_flat.avg_query_time_ms / r.avg_query_time_ms
            memory_ratio = faiss_flat.memory_mb / r.memory_mb
            print(f"  {r.method}: {speedup:.1f}x faster, {memory_ratio:.1f}x less memory")


def save_results(results: List[BenchmarkResult], filename: str):
    """Save results to JSON."""
    import json

    data = []
    for r in results:
        if r is None:
            continue
        data.append({
            'method': r.method,
            'n_vectors': r.n_vectors,
            'dim': r.dim,
            'build_time_s': r.build_time_s,
            'memory_mb': r.memory_mb,
            'avg_query_time_ms': r.avg_query_time_ms,
            'p50_query_time_ms': r.p50_query_time_ms,
            'p99_query_time_ms': r.p99_query_time_ms,
            'recall_at_1': r.recall_at_1,
            'recall_at_10': r.recall_at_10,
            'recall_at_100': r.recall_at_100,
            'params': r.params
        })

    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to {filename}")


def run_benchmark(
    n_vectors: int = 100000,
    dim: int = 768,
    n_queries: int = 1000,
    data_type: str = 'clustered'
):
    """Run the full benchmark suite."""
    print("="*100)
    print("WITNESS-LDPC vs FAISS BENCHMARK")
    print("="*100)
    print(f"\nConfiguration:")
    print(f"  - Vectors: {n_vectors:,}")
    print(f"  - Dimensions: {dim}")
    print(f"  - Queries: {n_queries}")
    print(f"  - Data type: {data_type}")

    # Generate data
    print(f"\nGenerating {data_type} data...")
    np.random.seed(42)

    if data_type == 'random':
        vectors = np.random.randn(n_vectors, dim).astype(np.float32)
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    elif data_type == 'clustered':
        vectors = generate_clustered_data(n_vectors, dim, n_clusters=100)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    # Queries
    query_indices = np.random.choice(n_vectors, n_queries, replace=False)
    queries = vectors[query_indices].copy()

    # Add some noise to queries (more realistic)
    queries = queries + np.random.randn(*queries.shape).astype(np.float32) * 0.1
    queries = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)

    # Ground truth
    print("\nComputing ground truth (exact k-NN)...")
    ground_truth = compute_exact_neighbors(vectors, queries, k=100)

    # Benchmark methods
    results = []

    # FAISS baselines
    results.append(benchmark_faiss(vectors.copy(), queries.copy(), ground_truth, index_type='flat'))
    results.append(benchmark_faiss(vectors.copy(), queries.copy(), ground_truth, index_type='ivf'))
    results.append(benchmark_faiss(vectors.copy(), queries.copy(), ground_truth, index_type='hnsw'))

    # Witness-LDPC variants
    results.append(benchmark_witness_ldpc(vectors, queries, ground_truth,
                                          code_length=1024, num_hashes=4, num_witnesses=32))
    results.append(benchmark_witness_ldpc(vectors, queries, ground_truth,
                                          code_length=2048, num_hashes=4, num_witnesses=64))
    results.append(benchmark_witness_ldpc(vectors, queries, ground_truth,
                                          code_length=4096, num_hashes=8, num_witnesses=128))

    # Compact version (no vector storage)
    results.append(benchmark_witness_ldpc(vectors, queries, ground_truth,
                                          code_length=4096, num_hashes=8, num_witnesses=128, compact=True))

    # Summary
    print_summary_table(results)

    # Save results
    save_results(results, f"benchmark_results_{n_vectors}_{dim}.json")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Witness-LDPC vs FAISS")
    parser.add_argument('--n_vectors', type=int, default=100000, help='Number of vectors')
    parser.add_argument('--dim', type=int, default=768, help='Vector dimension')
    parser.add_argument('--n_queries', type=int, default=500, help='Number of queries')
    parser.add_argument('--data_type', type=str, default='clustered', choices=['random', 'clustered'])

    args = parser.parse_args()

    run_benchmark(
        n_vectors=args.n_vectors,
        dim=args.dim,
        n_queries=args.n_queries,
        data_type=args.data_type
    )
