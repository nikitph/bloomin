#!/usr/bin/env python3
"""
Vector baseline using sentence-transformers + FAISS.
"""

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


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
    corpus_path = Path("data/sample.jsonl")
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
    
    # Run queries
    print("Running queries...\n")
    results = []
    
    for query_item in queries:
        query_text = query_item['query']
        relevant_docs = query_item['relevant_docs']
        
        # Encode query
        query_embedding = model.encode([query_text], show_progress_bar=False)
        
        # Search
        start_time = time.time()
        k = 5  # Top-K results
        distances, indices = index.search(query_embedding.astype('float32'), k)
        latency_ms = (time.time() - start_time) * 1000
        
        # Convert to results
        retrieved = []
        for i, idx in enumerate(indices[0]):
            if idx < len(doc_ids):
                retrieved.append({
                    "doc_id": doc_ids[idx],
                    "score": float(-distances[0][i])  # Negative distance as score (higher is better)
                })
        
        results.append({
            "query": query_text,
            "relevant_docs": relevant_docs,
            "retrieved": retrieved,
            "latency_ms": latency_ms
        })
        
        print(f"Query: '{query_text}'")
        print(f"  Latency: {latency_ms:.3f} ms")
        print(f"  Top-3: {[r['doc_id'] for r in retrieved[:3]]}")
        print()
    
    # Save results
    print(f"Saving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Done!")


if __name__ == "__main__":
    main()
