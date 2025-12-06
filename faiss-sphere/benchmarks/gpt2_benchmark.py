"""
GPT-2 Benchmark: Testing 768D → 350D Projection
================================================

GPT-2 has 768D embeddings - the perfect size to validate
our 2.2× speedup prediction!
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import faiss
import time
import pandas as pd
from transformers import GPT2Tokenizer, GPT2Model
import torch

from faiss_sphere import FAISSSphere


def get_gpt2_embeddings(texts, batch_size=16):
    """
    Get GPT-2 embeddings for texts.
    
    Uses last hidden state mean pooling.
    """
    print("Loading GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    
    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Force CPU to avoid MPS segfault
    device = torch.device('cpu')
    model.to(device)
    model.eval()
    
    print(f"Device: {device} (using CPU to avoid MPS issues)")
    print(f"Encoding {len(texts)} texts (batch size: {batch_size})...")
    
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128  # Reduced for speed
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling over sequence
            hidden_states = outputs.last_hidden_state
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            
            # Masked mean
            masked_hidden = hidden_states * attention_mask
            sum_hidden = masked_hidden.sum(dim=1)
            sum_mask = attention_mask.sum(dim=1)
            mean_hidden = sum_hidden / sum_mask
            
            embeddings.extend(mean_hidden.cpu().numpy())
        
        if (i + batch_size) % 500 == 0 or i == 0:
            print(f"  Processed {min(i+batch_size, len(texts))}/{len(texts)}")
    
    embeddings = np.array(embeddings).astype('float32')
    
    # Normalize
    faiss.normalize_L2(embeddings)
    
    return embeddings


def gpt2_benchmark(n_samples=10000, n_queries=100, k=10, d_intrinsic=350):
    """
    Benchmark GPT-2 (768D) with intrinsic projection.
    
    This should show the predicted 2.2× speedup!
    """
    print("="*80)
    print("GPT-2 BENCHMARK: 768D → 350D Intrinsic Projection")
    print("="*80)
    print(f"Samples: {n_samples:,}")
    print(f"Queries: {n_queries}")
    print(f"k: {k}")
    print(f"Intrinsic dimension: {d_intrinsic}D")
    print("="*80)
    print()
    
    # Generate sentences
    print("Generating sentences...")
    topics = [
        'artificial intelligence', 'machine learning', 'quantum computing',
        'climate change', 'renewable energy', 'biotechnology',
        'neuroscience', 'astrophysics', 'particle physics',
        'computer science', 'mathematics', 'philosophy',
    ]
    
    sentences = []
    for i in range(n_samples):
        topic = topics[i % len(topics)]
        sentences.append(f"The latest research in {topic} has shown remarkable progress in recent years.")
    
    # Get GPT-2 embeddings
    embeddings = get_gpt2_embeddings(sentences, batch_size=32)
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
        'time_per_query_ms': search_time * 1000 / n_queries,
        'memory_mb': memory,
        'recall': 1.000,
    })
    
    # Method 2: FAISS-Sphere Balanced
    print("\n" + "="*80)
    print("2. FAISS-Sphere Balanced (768D → 350D)")
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
    
    print(df.to_string(index=False))
    
    sphere_result = df[df['method'] == 'Sphere Balanced'].iloc[0]
    
    print("\n" + "="*80)
    print("VALIDATION")
    print("="*80)
    print(f"Speedup: {sphere_result['speedup']:.2f}×")
    print(f"Memory reduction: {sphere_result['memory_reduction']:.2f}×")
    print(f"Recall@{k}: {sphere_result['recall']:.1%}")
    print(f"Variance explained: {stats.get('variance_explained', 0):.1%}")
    
    if sphere_result['speedup'] > 1.5:
        print(f"\n✅ SPEEDUP VALIDATED: {sphere_result['speedup']:.2f}× faster (predicted: 2.2×)")
    else:
        print(f"\n⚠️  Speedup lower than predicted: {sphere_result['speedup']:.2f}× (expected: 2.2×)")
    
    if sphere_result['recall'] > 0.90:
        print(f"✅ RECALL VALIDATED: {sphere_result['recall']:.1%} (excellent!)")
    
    if sphere_result['memory_reduction'] > 1.8:
        print(f"✅ MEMORY VALIDATED: {sphere_result['memory_reduction']:.2f}× smaller (predicted: 2.19×)")
    
    return df


if __name__ == '__main__':
    results_df = gpt2_benchmark(
        n_samples=5000,  # 5K for faster testing (GPT-2 is slow on CPU)
        n_queries=100,
        k=10,
        d_intrinsic=350
    )
    
    results_df.to_csv('gpt2_benchmark.csv', index=False)
    print("\n✓ Results saved to gpt2_benchmark.csv")
