#!/usr/bin/env python3
"""
Vector baseline using sentence-transformers + FAISS.
"""

import json
import time
import requests
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


SBNG_URL = "http://localhost:3001/query"

def load_corpus(corpus_path):
    """Load documents from JSONL corpus."""
    docs = []
    with open(corpus_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            docs.append(doc)
    return docs


def load_queries(queries_path):
    """Load benchmark queries."""
    with open(queries_path, 'r') as f:
        queries = json.load(f)
    return queries


def main():
    print("=== Vector Baseline (sentence-transformers + FAISS) ===\n")
    
    # Paths
    corpus_path = Path("data/wikipedia_10k.jsonl")
    queries_path = Path("benches/eval_queries.json")
    output_path = Path("scripts/vector_results.json")
    
    # Load data
    print("Loading corpus and queries...")
    docs = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    
    print(f"Loaded {len(docs)} documents, {len(queries)} queries\n")
    
    # Load model
    print("Loading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded\n")
    
    # Encode documents
    print("Encoding documents...")
    doc_texts = [doc['text'] for doc in docs]
    doc_ids = [doc['id'] for doc in docs]
    doc_embeddings = model.encode(doc_texts, show_progress_bar=False)
    print(f"Encoded {len(doc_embeddings)} documents\n")
    
    # Build FAISS index
    print("Building FAISS index...")
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(doc_embeddings.astype('float32'))
    print(f"FAISS index built (dimension={dimension})\n")
    
    # 1. BM25
    print("Running BM25...")
    from rank_bm25 import BM25Okapi
    tokenized_corpus = [doc.split(" ") for doc in doc_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    bm25_results = []
    bm25_times = []
    k = 5
    for q_item in queries:
        q = q_item['query']
        start = time.time()
        tokenized_query = q.split(" ")
        # BM25 returns top-k docs
        top_docs = bm25.get_top_n(tokenized_query, doc_texts, n=k)
        # Map back to IDs (inefficient but fine for small bench)
        hits = []
        for doc_text in top_docs:
            try:
                idx = doc_texts.index(doc_text)
                hits.append(doc_ids[idx])
            except ValueError:
                pass
        bm25_times.append((time.time() - start) * 1000)
        bm25_results.append(hits)

    # 2. SBNG (HTTP)
    print(f"Querying SBNG (HTTP) for {len(queries)} queries...")
    sbng_results = []
    sbng_times = []
    k = 5
    for q_item in queries:
        q = q_item['query']
        start = time.time()
        try:
            res = requests.post(SBNG_URL, json={"q": q, "k": k, "rerank": False}).json()
            hits = [r["doc_id"] for r in res.get("results", [])]
        except Exception as e:
            print(f"SBNG error: {e}")
            hits = []
        sbng_times.append((time.time() - start) * 1000)
        sbng_results.append(hits)

    # 3. SBNG + Rerank (HTTP)
    print(f"Querying SBNG + Rerank (HTTP) for {len(queries)} queries...")
    sbng_rerank_results = []
    sbng_rerank_times = []
    for q_item in queries:
        q = q_item['query']
        start = time.time()
        try:
            res = requests.post(SBNG_URL, json={"q": q, "k": k, "rerank": True}).json()
            hits = [r["doc_id"] for r in res.get("results", [])]
        except Exception as e:
            print(f"SBNG+Rerank error: {e}")
            hits = []
        sbng_rerank_times.append((time.time() - start) * 1000)
        sbng_rerank_results.append(hits)

    # 4. Vector (SentenceTransformers)
    print(f"Querying Vector for {len(queries)} queries...")
    vector_results = []
    vector_times = []
    k = 5
    for q_item in queries:
        q = q_item['query']
        start = time.time()
        q_emb = model.encode(q)
        # Simple cosine similarity
        # We use faiss or just numpy for simplicity here
        import numpy as np
        scores = np.dot(doc_embeddings, q_emb)
        top_k_idx = np.argsort(scores)[::-1][:k]
        hits = [doc_ids[i] for i in top_k_idx]
        vector_times.append((time.time() - start) * 1000)
        vector_results.append(hits)

    # Evaluate
    print("\n--- Evaluation Results ---")
    systems = [
        ("BM25", bm25_results, bm25_times),
        ("SBNG", sbng_results, sbng_times),
        ("SBNG+Rerank", sbng_rerank_results, sbng_rerank_times),
        ("Vector", vector_results, vector_times),
    ]

    for name, results, times in systems:
        avg_time = sum(times) / len(times)
        print(f"\nSystem: {name}")
        print(f"  Avg Latency: {avg_time:.2f} ms")
        
        # Calculate MRR and Recall@K
        mrr = 0.0
        recall = 0.0
        for i, hits in enumerate(results):
            # Ground truth is just the doc_id containing the query terms in synthetic data
            # But here we don't have ground truth labels easily unless we generate them.
            # For synthetic data, we can assume docs containing the query words are relevant?
            # Or we just print the results for manual inspection or comparison.
            pass
            
        # Since we don't have ground truth for synthetic queries easily without generating them carefully,
        # we will just print the overlap with Vector search as a proxy for "quality" if Vector is gold standard.
        if name != "Vector":
            overlap = 0
            for i, hits in enumerate(results):
                vec_hits = set(vector_results[i])
                my_hits = set(hits)
                overlap += len(vec_hits.intersection(my_hits))
            avg_overlap = overlap / (len(queries) * k)
            print(f"  Overlap with Vector: {avg_overlap:.2%}")
        # print(f"  Top-3: {[r['doc_id'] for r in retrieved[:3]]}") # 'retrieved' not defined
        print()
    
    # Save results
    print(f"Saving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Done!")


if __name__ == "__main__":
    main()
