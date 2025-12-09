"""
Example 5: Topos Compositional QA

Demonstrates:
- Local proposition extraction
- Gluing consistency checks
- Compositional question answering
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import CompositionalQAGenerator
from witnesses import estimate_witness_distribution
from topos import ToposLogic, CompositionalQA

def main():
    print("=== Topos Compositional QA ===")
    print()
    
    # 1. Generate compositional dataset
    print("1. Generating CLEVR-style dataset...")
    generator = CompositionalQAGenerator(n_objects=50)
    documents, qa_pairs = generator.generate()
    
    print(f"   Generated {len(documents)} objects")
    print(f"   Generated {len(qa_pairs)} QA pairs")
    print()
    
    # 2. Build Topos
    print("2. Building Topos (local propositions)...")
    topos = ToposLogic(confidence_threshold=0.7)
    
    # Create sections for each document
    for doc in documents:
        # Create witness distribution from metadata
        witness_dist = {}
        for key, value in doc.metadata.items():
            if key != 'object_id':
                witness_dist[f"{key}_{value}"] = 1.0
        
        # Normalize
        total = sum(witness_dist.values())
        if total > 0:
            witness_dist = {k: v/total for k, v in witness_dist.items()}
        
        topos.build_section(doc.id, witness_dist)
    
    print(f"   Built {len(topos.sections)} local sections")
    print()
    
    # 3. Check gluing consistency
    print("3. Checking gluing consistency...")
    sections = list(topos.sections.values())
    
    consistent_pairs = 0
    total_pairs = 0
    
    for i in range(min(10, len(sections))):
        for j in range(i + 1, min(i + 5, len(sections))):
            is_consistent, inconsistencies = topos.check_gluing_consistency(
                sections[i],
                sections[j]
            )
            total_pairs += 1
            if is_consistent:
                consistent_pairs += 1
    
    consistency_rate = consistent_pairs / total_pairs if total_pairs > 0 else 0
    print(f"   Consistency rate: {consistency_rate:.1%} ({consistent_pairs}/{total_pairs} pairs)")
    print()
    
    # 4. Run compositional QA
    print("4. Running compositional QA...")
    qa_system = CompositionalQA(topos)
    
    correct = 0
    total = 0
    
    for qa in qa_pairs[:10]:  # Test first 10
        query = qa['question']
        expected = set(qa['answer'])
        
        results = qa_system.answer_query(query, documents)
        predicted = set(results)
        
        # Check accuracy
        if len(expected) > 0:
            recall = len(predicted & expected) / len(expected)
            if recall > 0.5:  # Consider correct if >50% recall
                correct += 1
            total += 1
        
        print(f"   Q: {query}")
        print(f"      Expected: {len(expected)} objects, Got: {len(predicted)} objects")
        print(f"      Recall: {recall:.1%}")
    
    accuracy = correct / total if total > 0 else 0
    print()
    print(f"Overall QA Accuracy: {accuracy:.1%} ({correct}/{total})")
    print()
    
    # 5. Sample propositions
    print("5. Sample propositions from first section:")
    first_section = sections[0]
    print(f"   Region: {first_section.region_id}")
    print(f"   Propositions:")
    for prop in first_section.propositions[:5]:
        print(f"     - {prop.predicate} (confidence: {prop.confidence:.2f})")
    
    print()
    print("✅ Topos compositional QA complete!")

if __name__ == "__main__":
    main()
