"""
Example 10: Rigorous Baseline Comparison (S-BERT)

We test a standard dense retriever (all-MiniLM-L6-v2) against the structural queries
that REWA-Topos passed.

Tests:
1. Modifier Binding ("Red Bike")
2. Negation ("Not Red")
"""

import sys
from sentence_transformers import SentenceTransformer, util

def main():
    print("=== Rigorous Benchmark: S-BERT (all-MiniLM-L6-v2) ===")
    print("Loading model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Test 1: Modifier Binding
    print("\n--- Test 1: Modifier Binding ---")
    query = "Find a red bike"
    
    docs = [
        "A red car and a small blue bike.", # Distractor (has Red, has Bike, but wrong binding)
        "A red bike next to a bench."       # Target
    ]
    
    # Encode
    q_emb = model.encode(query, convert_to_tensor=True)
    d_embs = model.encode(docs, convert_to_tensor=True)
    
    # Compute Cosine Similarity
    scores = util.cos_sim(q_emb, d_embs)[0]
    
    print(f"Query: '{query}'")
    print(f"1. '{docs[0]}' -> Score: {scores[0]:.4f}")
    print(f"2. '{docs[1]}' -> Score: {scores[1]:.4f}")
    
    if scores[0] > scores[1]:
        print("❌ FAIL: S-BERT ranked Distractor higher.")
    elif scores[0] > 0.9 * scores[1]:
        print("⚠️ WEAK: S-BERT scores are dangerously close (Distractor is >90% of Target).")
    else:
        print("✅ PASS: S-BERT successfully distinguished them.")

    # Test 3: Negation
    print("\n--- Test 3: Negation ---")
    query = "The bike is red"
    
    docs = [
        "The bike is red.",      # Target
        "The bike is not red."   # Contradiction (Should have LOW similarity in semantic search?)
                                 # Actually, in vector space, negation often has HIGH similarity.
    ]
    
    q_emb = model.encode(query, convert_to_tensor=True)
    d_embs = model.encode(docs, convert_to_tensor=True)
    scores = util.cos_sim(q_emb, d_embs)[0]
    
    print(f"Query: '{query}'")
    print(f"1. '{docs[0]}' -> Score: {scores[0]:.4f}")
    print(f"2. '{docs[1]}' -> Score: {scores[1]:.4f}")
    
    if scores[1] > 0.8:
        print("❌ FAIL: S-BERT assigns high similarity to direct contradiction.")
    else:
        print("✅ PASS: S-BERT correctly penalizes negation.")

if __name__ == "__main__":
    main()
