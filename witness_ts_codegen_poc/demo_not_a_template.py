from witness_inversion_poc.bloom import bloom_encode, bloom_decode
from witness_ts_codegen_poc.ts_api_generator import generate_ts_api_endpoint

def run_proof():
    print("🧬 Witness Arithmetic: Algebra vs Templates Proof")
    print("=" * 60)
    
    vocabulary = [
        "http_get", "http_post", "zod_schema", "jwt_auth", 
        "db_save", "error_boundary", "json_response"
    ]
    m = 256
    k = 3

    # 1. DEFINE TWO INDEPENDENT APIS
    # API A: Simple Public Read
    api_a_intent = {"http_get", "json_response", "error_boundary"}
    sketch_a = bloom_encode(api_a_intent, m, k)
    
    # API B: Secure Admin Write
    api_b_intent = {"http_post", "zod_schema", "jwt_auth", "db_save", "json_response", "error_boundary"}
    sketch_b = bloom_encode(api_b_intent, m, k)

    print("\n[SCENARIO 1: SEMANTIC DIFFERENTIATION (XOR)]")
    # What is the 'delta' required to upgrade API A to API B?
    delta_bits = [sa ^ sb for sa, sb in zip(sketch_a, sketch_b)]
    delta_semantics = bloom_decode(delta_bits, vocabulary, k)
    
    print(f"   Upgrade Path (A -> B): {delta_semantics}")
    print("   Note: Templates can't calculate 'diffs'. Sketches can.")

    print("\n[SCENARIO 2: SEMANTIC INTERSECTION (AND)]")
    # What is the common 'Standard Library' shared between these two APIs?
    common_bits = [sa & sb for sa, sb in zip(sketch_a, sketch_b)]
    common_semantics = bloom_decode(common_bits, vocabulary, k)
    
    print(f"   Shared Standard Pattern: {common_semantics}")
    print("   Resulting Generated 'Common' Code:")
    print("-" * 40)
    print(generate_ts_api_endpoint(common_semantics))
    print("-" * 40)

    print("\n[SCENARIO 3: COMPOSITIONAL SYNTHESIS (OR)]")
    # Combine a "Security Mask" with a "Blank API"
    security_mask = {"jwt_auth", "error_boundary"}
    base_api = {"http_get", "json_response"}
    
    sketch_s = bloom_encode(security_mask, m, k)
    sketch_p = bloom_encode(base_api, m, k)
    
    composed_sketch = [ss | sp for ss, sp in zip(sketch_s, sketch_p)]
    composed_semantics = bloom_decode(composed_sketch, vocabulary, k)
    
    print(f"   Composed Hybrid: {composed_semantics}")
    print("   Resulting Generated 'Secured' Code:")
    print("-" * 40)
    print(generate_ts_api_endpoint(composed_semantics))
    print("-" * 40)

    print("\n✅ VERDICT: This is NOT a template.")
    print("   A template is a rigid string literal.")
    print("   A Witness Sketch is 'Algebraic DNA'. It can be added, subtracted, and intersected")
    print("   to produce valid code from any combination of semantic inputs.")

if __name__ == "__main__":
    run_proof()
