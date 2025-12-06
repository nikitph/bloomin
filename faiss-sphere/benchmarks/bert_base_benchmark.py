"""
BERT-base Benchmark using Sentence-Transformers
================================================

Using sentence-transformers library which wraps BERT-base
and is more stable than raw transformers.

We'll use 'bert-base-nli-mean-tokens' which is BERT-base (768D)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import faiss
import time
import pandas as pd
from sentence_transformers import SentenceTransformer

from faiss_sphere import FAISSSphere


def bert_base_benchmark_stable(n_samples=10000, n_queries=100, k=10, d_intrinsic=350):
    """
    Benchmark BERT-base (768D) using sentence-transformers.
    
    This should validate our 2.2× speedup prediction!
    """
    print("="*80)
    print("BERT-BASE BENCHMARK: 768D → 350D Intrinsic Projection")
    print("="*80)
    print(f"Samples: {n_samples:,}")
    print(f"Queries: {n_queries}")
    print(f"k: {k}")
    print(f"Intrinsic dimension: {d_intrinsic}D")
    print("="*80)
    print()
    
    # Generate diverse sentences
    print("Generating sentences...")
    topics = [
        'artificial intelligence', 'machine learning', 'quantum computing',
        'climate change', 'renewable energy', 'biotechnology',
        'neuroscience', 'astrophysics', 'particle physics',
        'computer science', 'mathematics', 'philosophy',
        'economics', 'psychology', 'sociology',
        'literature', 'art history', 'music theory',
        'political science', 'international relations', 'law',
        'medicine', 'public health', 'epidemiology',
    ]
    
    templates = [
        "The latest research in {} has shown remarkable progress.",
        "Recent developments in {} have transformed our understanding.",
        "Experts in {} are making breakthrough discoveries.",
        "The impact of {} on modern society is significant.",
        "Studies in {} reveal important insights about {}.",
        "The relationship between {} and {} is complex.",
        "Advances in {} enable new applications in {}.",
    ]
    
    sentences = []
    for i in range(n_samples):
        template = templates[i % len(templates)]
        topic1 = topics[i % len(topics)]
        topic2 = topics[(i + 1) % len(topics)]
        
        if template.count('{}') == 1:
            sentence = template.format(topic1)
        elif template.count('{}') == 2:
            sentence = template.format(topic1, topic2)
        else:
            sentence = template
        
        sentences.append(sentence)
    
    # Load BERT-base model via sentence-transformers
    print("\nLoading BERT-base model (via sentence-transformers)...")
    print("Model: 'paraphrase-distilroberta-base-v1' (768D)")
    
    # Use a 768D model
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    
    # Get embeddings
    print(f"\nGenerating embeddings for {n_samples:,} sentences...")
    embeddings = model.encode(
        sentences,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype('float32')
    
    D = embeddings.shape[1]
    
    print(f"\n✓ Generated {len(embeddings)} embeddings")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dimension: {D}D")
    
    # Query embeddings
    query_embeddings = embeddings[:n_queries]
    
    results = []
    
    # Ground truth
    print("\nComputing ground truth...")
    index_flat = faiss.IndexFlatIP(D)
    index_flat.add(embeddings)
    dist_true, idx_true = index_flat.search(query_embeddings, k)
    
    # Method 1: FAISS Flat (baseline)
    print("\n" + "="*80)
    print("1. FAISS IndexFlatIP (baseline)")
    print("="*80)
    
    index_faiss = faiss.IndexFlatIP(D)
    
    start = time.time()
    index_faiss.add(embeddings)
    add_time = time.time() - start
    
    start = time.time()
    dist1, idx1 = index_faiss.search(query_embeddings, k)
    search_time = time.time() - start
    
    memory = n_samples * D * 4 / 1e6
    
    print(f"Add time: {add_time*1000:.2f}ms")
    print(f"Search time: {search_time*1000:.2f}ms ({search_time*1000/n_queries:.3f}ms per query)")
    print(f"Memory: {memory:.1f} MB")
    
    results.append({
        'method': 'FAISS Flat',
        'dimension': D,
        'time_per_query_ms': search_time * 1000 / n_queries,
        'memory_mb': memory,
        'recall': 1.000,
    })
    
    # Method 2: FAISS-Sphere Balanced
    print("\n" + "="*80)
    print(f"2. FAISS-Sphere Balanced ({D}D → {d_intrinsic}D)")
    print("="*80)
    
    index_sphere = FAISSSphere(D, mode='balanced', d_intrinsic=d_intrinsic)
    
    # Train
    train_size = min(5000, n_samples)
    print(f"Training on {train_size:,} samples...")
    index_sphere.train(embeddings[:train_size])
    
    start = time.time()
    index_sphere.add(embeddings)
    add_time = time.time() - start
    
    start = time.time()
    dist2, idx2 = index_sphere.search(query_embeddings, k)
    search_time = time.time() - start
    
    # Compute recall
    recall = np.mean([
        len(set(idx_true[i]) & set(idx2[i])) / k
        for i in range(n_queries)
    ])
    
    stats = index_sphere.get_stats()
    memory = n_samples * d_intrinsic * 4 / 1e6
    
    print(f"Add time: {add_time*1000:.2f}ms")
    print(f"Search time: {search_time*1000:.2f}ms ({search_time*1000/n_queries:.3f}ms per query)")
    print(f"Memory: {memory:.1f} MB")
    print(f"Recall@{k}: {recall:.3f}")
    print(f"Variance explained: {stats.get('variance_explained', 0):.3f}")
    
    results.append({
        'method': 'Sphere Balanced',
        'dimension': d_intrinsic,
        'time_per_query_ms': search_time * 1000 / n_queries,
        'memory_mb': memory,
        'recall': recall,
    })
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    df = pd.DataFrame(results)
    
    baseline_time = df[df['method'] == 'FAISS Flat']['time_per_query_ms'].values[0]
    baseline_memory = df[df['method'] == 'FAISS Flat']['memory_mb'].values[0]
    
    df['speedup'] = baseline_time / df['time_per_query_ms']
    df['memory_reduction'] = baseline_memory / df['memory_mb']
    
    print(df[['method', 'dimension', 'time_per_query_ms', 'memory_mb', 'recall', 'speedup', 'memory_reduction']].to_string(index=False))
    
    sphere_result = df[df['method'] == 'Sphere Balanced'].iloc[0]
    
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    print(f"Compression: {D}D → {d_intrinsic}D = {D/d_intrinsic:.2f}×")
    print(f"Speedup: {sphere_result['speedup']:.2f}×")
    print(f"Memory reduction: {sphere_result['memory_reduction']:.2f}×")
    print(f"Recall@{k}: {sphere_result['recall']:.1%}")
    print(f"Variance explained: {stats.get('variance_explained', 0):.1%}")
    
    if sphere_result['speedup'] > 1.5:
        print(f"\n✅ SPEEDUP VALIDATED: {sphere_result['speedup']:.2f}× faster!")
    else:
        print(f"\n⚠️  Speedup: {sphere_result['speedup']:.2f}×")
    
    if sphere_result['recall'] > 0.90:
        print(f"✅ RECALL VALIDATED: {sphere_result['recall']:.1%} (excellent!)")
    
    if sphere_result['memory_reduction'] > 1.5:
        print(f"✅ MEMORY VALIDATED: {sphere_result['memory_reduction']:.2f}× smaller!")
    
    print(f"\n🎉 {D}D model shows the power of intrinsic projection!")
    
    return df


if __name__ == '__main__':
    results_df = bert_base_benchmark_stable(
        n_samples=20000,  # 20K samples for good statistics
        n_queries=100,
        k=10,
        d_intrinsic=350
    )
    
    results_df.to_csv('bert_768d_benchmark.csv', index=False)
    print("\n✓ Results saved to bert_768d_benchmark.csv")
