from witness_inversion_poc.bloom import bloom_encode, bloom_decode
from witness_ts_codegen_poc.ts_synthesizer import synthesize_ts
import yaml
import os

def run_ts_demo():
    print("TypeScript Verified Codegen POC")
    print("=" * 40)
    
    # Load TS Vocabulary
    vocab_path = os.path.join(os.path.dirname(__file__), "witnesses_ts.yaml")
    with open(vocab_path, "r") as f:
        data = yaml.safe_load(f)
    vocabulary = []
    for cat, items in data.items():
        vocabulary.extend(items)
        
    m = 512
    k = 3
    
    # 1. BASE REQUIREMENT: Async Save API
    base_reqs = {"typed_interface", "async_await", "db_save"}
    print(f"\n🔹 Scenario 1: Base Requirements {base_reqs}")
    
    sketch = bloom_encode(base_reqs, m, k)
    decoded = bloom_decode(sketch, vocabulary, k)
    print(f"   Generated Code (Base):")
    print(synthesize_ts(decoded, with_proof=True))
    
    # 2. ALGEBRAIC EDIT: Add Safety & Generics
    print(f"\n🔹 Scenario 2: Algebraic Edit (+ Safety, + Generics)")
    safety_sketch = bloom_encode({"input_validation", "generic_type", "optional_field"}, m, k)
    
    # Binary OR = Semantic Addition
    enhanced_sketch = [s | sa for s, sa in zip(sketch, safety_sketch)]
    enhanced_decoded = bloom_decode(enhanced_sketch, vocabulary, k)
    
    print(f"   Generated Code (Enhanced):")
    print(synthesize_ts(enhanced_decoded, with_proof=True))

    # 3. INTERSECTION: Finding Common Type Semantics
    print(f"\n🔹 Scenario 3: Semantic Intersection (Finding Shared Patterns)")
    sketch_a = bloom_encode({"typed_interface", "optional_field", "async_await"}, m, k)
    sketch_b = bloom_encode({"typed_interface", "union_type", "loop"}, m, k)
    
    common_sketch = [a & b for a, b in zip(sketch_a, sketch_b)]
    common_decoded = bloom_decode(common_sketch, vocabulary, k)
    
    print(f"   Shared Semantics: {common_decoded}")
    print(synthesize_ts(common_decoded, with_proof=True))

if __name__ == "__main__":
    run_ts_demo()
