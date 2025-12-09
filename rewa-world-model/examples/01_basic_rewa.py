"""
Example 1: Basic REWA Encoding and Retrieval

Demonstrates:
- Witness extraction from documents
- REWA encoding with capacity estimation
- Basic retrieval and evaluation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from witnesses import WitnessExtractor, WitnessType
from encoding import REWAConfig, REWAEncoder
from retrieval import REWARetriever, evaluate_retrieval
from data import HierarchicalGaussianGenerator

def main():
    print("=== Basic REWA Encoding and Retrieval ===")
    print()
    
    # 1. Generate synthetic dataset
    print("1. Generating hierarchical Gaussian dataset...")
    generator = HierarchicalGaussianGenerator(
        n_levels=2,
        branching_factor=3,
        dim=32
    )
    documents = generator.generate(docs_per_leaf=5)
    print(f"   Generated {len(documents)} documents")
    print()
    
    # 2. Extract witnesses
    print("2. Extracting Boolean witnesses (tokens)...")
    extractor = WitnessExtractor([WitnessType.BOOLEAN])
    
    doc_witnesses = {}
    for doc in documents:
        witnesses = extractor.extract({'id': doc.id, 'text': doc.text})
        doc_witnesses[doc.id] = witnesses
    
    print(f"   Extracted witnesses from {len(doc_witnesses)} documents")
    print()
    
    # 3. Estimate capacity and create encoder
    print("3. Creating REWA encoder...")
    N = len(documents)
    delta = 0.1  # Minimum gap
    K = 3  # Number of hash functions
    
    m = REWAEncoder.estimate_m(delta, N, K)
    print(f"   Capacity estimation: m = {m} positions for N = {N} documents")
    print("   (Shannon formula: m ~ 1/delta^2 * log N)")
    print()
    
    config = REWAConfig(
        input_dim=1000,  # Rough estimate of unique tokens
        num_positions=m,
        num_hashes=K,
        delta_gap=delta
    )
    encoder = REWAEncoder(config)
    
    # 4. Encode all documents
    print("4. Encoding documents...")
    retriever = REWARetriever(WitnessType.BOOLEAN)
    
    for doc in documents:
        witnesses = doc_witnesses[doc.id]
        signature = encoder.encode(witnesses)
        retriever.add(doc.id, signature)
    
    print(f"   Encoded {retriever.size()} documents")
    print(f"   Index size: {retriever.memory_usage() / 1024:.2f} KB")
    print()
    
    # 5. Test retrieval
    print("5. Testing retrieval...")
    
    # Use first 10 documents as queries
    queries = []
    ground_truth = {}
    
    for doc in documents[:10]:
        witnesses = doc_witnesses[doc.id]
        signature = encoder.encode(witnesses)
        queries.append((signature, doc.id))
        
        # Ground truth: documents in same cluster
        cluster_id = doc.metadata['cluster_id']
        relevant = [d.id for d in documents if d.metadata['cluster_id'] == cluster_id]
        ground_truth[doc.id] = relevant
    
    # Evaluate
    metrics = evaluate_retrieval(retriever, queries, ground_truth, k_values=[1, 5, 10])
    
    print("   Retrieval Metrics:")
    for metric, value in metrics.items():
        print(f"     {metric}: {value:.3f}")
    
    print()
    print("6. Sample retrieval:")
    query_doc = documents[0]
    query_sig = encoder.encode(doc_witnesses[query_doc.id])
    results = retriever.search(query_sig, k=5)
    
    print(f"   Query: {query_doc.text}")
    print(f"   Cluster: {query_doc.metadata['cluster_id']}")
    print()
    print("   Top-5 Results:")
    for i, result in enumerate(results):
        doc = next(d for d in documents if d.id == result.doc_id)
        print(f"     {i+1}. {doc.text}")
        print(f"        Cluster: {doc.metadata['cluster_id']}, Score: {result.score:.3f}")
    
    print()
    print("✅ Basic REWA encoding and retrieval complete!")

if __name__ == "__main__":
    main()
