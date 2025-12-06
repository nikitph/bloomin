"""
Comprehensive Optimization Benchmark
=====================================

Tests all optimization variants:
1. FAISS Flat 768D (baseline)
2. Sphere NumPy (current)
3. Sphere BLAS (optimized)
4. Sphere Numba (fully optimized)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import faiss
import time
import pandas as pd

from optimized_projection import OptimizedIntrinsicProjector


def comprehensive_benchmark(n_samples=50000, n_queries=100, k=10):
    """
    Benchmark all optimization variants.
    """
    print("="*80)
    print("COMPREHENSIVE OPTIMIZATION BENCHMARK")
    print("="*80)
    print(f"Samples: {n_samples:,}")
    print(f"Queries: {n_queries}")
    print(f"k: {k}")
    print("="*80)
    print()
    
    # Generate data with TRUE 350D intrinsic structure
    print("Generating data with 350D intrinsic structure...")
    D = 768
    d_true_intrinsic = 350
    
    # Step 1: Generate in intrinsic space
    intrinsic_data = np.random.randn(n_samples, d_true_intrinsic).astype('float32')
    intrinsic_data = np.ascontiguousarray(intrinsic_data)
    faiss.normalize_L2(intrinsic_data)
    
    # Step 2: Random projection to 768D
    projection_matrix = np.random.randn(d_true_intrinsic, D).astype('float32') / np.sqrt(d_true_intrinsic)
    embeddings = (intrinsic_data @ projection_matrix).astype('float32')
    embeddings = np.ascontiguousarray(embeddings)
    faiss.normalize_L2(embeddings)
    
    print(f"✓ Generated {len(embeddings)} embeddings")
    print(f"  True intrinsic dimension: {d_true_intrinsic}D")
    print(f"  Ambient dimension: {D}D")
    
    # Query embeddings
    query_embeddings = embeddings[:n_queries]
    
    results = []
    
    # Ground truth
    print("\nComputing ground truth...")
    index_gt = faiss.IndexFlatIP(D)
    index_gt.add(embeddings)
    _, gt_indices = index_gt.search(query_embeddings, k)
    
    # Method 1: FAISS Flat (baseline)
    print("\n" + "="*80)
    print("1. FAISS Flat 768D (baseline)")
    print("="*80)
    
    start = time.time()
    _, idx1 = index_gt.search(query_embeddings, k)
    time1 = (time.time() - start) * 1000
    
    memory1 = n_samples * D * 4 / 1e6
    
    print(f"Search time: {time1:.2f}ms ({time1/n_queries:.3f}ms per query)")
    print(f"Memory: {memory1:.1f} MB")
    
    results.append({
        'method': 'FAISS Flat 768D',
        'dimension': D,
        'time_ms': time1,
        'time_per_query_ms': time1 / n_queries,
        'memory_mb': memory1,
        'recall': 1.000,
        'speedup': 1.000
    })
    
    # Method 2: Sphere NumPy
    print("\n" + "="*80)
    print("2. FAISS-Sphere (NumPy projection)")
    print("="*80)
    
    projector = OptimizedIntrinsicProjector(D)
    projector.train(embeddings[:5000])
    
    d_opt = projector.d_intrinsic
    print(f"Selected dimension: {d_opt}D")
    
    # Project database
    data_proj = projector.project_numpy(embeddings)
    index2 = faiss.IndexFlatIP(d_opt)
    index2.add(data_proj)
    
    # Search
    start = time.time()
    queries_proj = projector.project_numpy(query_embeddings)
    _, idx2 = index2.search(queries_proj, k)
    time2 = (time.time() - start) * 1000
    
    recall2 = np.mean([
        len(set(idx2[i]) & set(gt_indices[i])) / k
        for i in range(n_queries)
    ])
    
    memory2 = n_samples * d_opt * 4 / 1e6
    
    print(f"Search time: {time2:.2f}ms ({time2/n_queries:.3f}ms per query)")
    print(f"Memory: {memory2:.1f} MB")
    print(f"Recall: {recall2:.3f}")
    
    results.append({
        'method': f'Sphere NumPy ({d_opt}D)',
        'dimension': d_opt,
        'time_ms': time2,
        'time_per_query_ms': time2 / n_queries,
        'memory_mb': memory2,
        'recall': recall2,
        'speedup': time1 / time2
    })
    
    # Method 3: Sphere BLAS
    print("\n" + "="*80)
    print("3. FAISS-Sphere (BLAS projection)")
    print("="*80)
    
    start = time.time()
    queries_proj_blas = projector.project_blas(query_embeddings)
    _, idx3 = index2.search(queries_proj_blas, k)
    time3 = (time.time() - start) * 1000
    
    recall3 = np.mean([
        len(set(idx3[i]) & set(gt_indices[i])) / k
        for i in range(n_queries)
    ])
    
    print(f"Search time: {time3:.2f}ms ({time3/n_queries:.3f}ms per query)")
    print(f"Memory: {memory2:.1f} MB")
    print(f"Recall: {recall3:.3f}")
    
    results.append({
        'method': f'Sphere BLAS ({d_opt}D)',
        'dimension': d_opt,
        'time_ms': time3,
        'time_per_query_ms': time3 / n_queries,
        'memory_mb': memory2,
        'recall': recall3,
        'speedup': time1 / time3
    })
    
    # Method 4: Sphere Numba
    print("\n" + "="*80)
    print("4. FAISS-Sphere (Numba projection)")
    print("="*80)
    
    # Warmup Numba
    print("Warming up Numba JIT...")
    _ = projector.project_numba(query_embeddings[:10])
    
    start = time.time()
    queries_proj_numba = projector.project_numba(query_embeddings)
    _, idx4 = index2.search(queries_proj_numba, k)
    time4 = (time.time() - start) * 1000
    
    recall4 = np.mean([
        len(set(idx4[i]) & set(gt_indices[i])) / k
        for i in range(n_queries)
    ])
    
    print(f"Search time: {time4:.2f}ms ({time4/n_queries:.3f}ms per query)")
    print(f"Memory: {memory2:.1f} MB")
    print(f"Recall: {recall4:.3f}")
    
    results.append({
        'method': f'Sphere Numba ({d_opt}D)',
        'dimension': d_opt,
        'time_ms': time4,
        'time_per_query_ms': time4 / n_queries,
        'memory_mb': memory2,
        'recall': recall4,
        'speedup': time1 / time4
    })
    
    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION COMPARISON")
    print("="*80)
    
    df = pd.DataFrame(results)
    print(df[['method', 'dimension', 'time_per_query_ms', 'memory_mb', 'recall', 'speedup']].to_string(index=False))
    
    # Highlight best
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    best_speedup = df.loc[df['speedup'].idxmax()]
    print(f"\n🏆 FASTEST: {best_speedup['method']}")
    print(f"   Speedup: {best_speedup['speedup']:.2f}×")
    print(f"   Time per query: {best_speedup['time_per_query_ms']:.3f}ms")
    print(f"   Recall: {best_speedup['recall']:.3f}")
    print(f"   Memory: {best_speedup['memory_mb']:.1f} MB ({memory1/best_speedup['memory_mb']:.2f}× smaller)")
    
    # Compare optimizations
    numpy_speedup = df[df['method'].str.contains('NumPy')]['speedup'].values[0]
    blas_speedup = df[df['method'].str.contains('BLAS')]['speedup'].values[0]
    numba_speedup = df[df['method'].str.contains('Numba')]['speedup'].values[0]
    
    print(f"\n📊 OPTIMIZATION GAINS:")
    print(f"   NumPy baseline: {numpy_speedup:.2f}×")
    print(f"   + BLAS: {blas_speedup:.2f}× (+{(blas_speedup/numpy_speedup-1)*100:.1f}%)")
    print(f"   + Numba: {numba_speedup:.2f}× (+{(numba_speedup/numpy_speedup-1)*100:.1f}%)")
    
    if numba_speedup >= 2.0:
        print(f"\n✅ TARGET ACHIEVED: {numba_speedup:.2f}× speedup (goal: 2.0×)")
    else:
        print(f"\n⚠️  Close to target: {numba_speedup:.2f}× (goal: 2.0×)")
    
    return df


if __name__ == '__main__':
    results_df = comprehensive_benchmark(
        n_samples=50000,
        n_queries=100,
        k=10
    )
    
    results_df.to_csv('optimization_comparison.csv', index=False)
    print("\n✓ Results saved to optimization_comparison.csv")
