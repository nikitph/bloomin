from witness_inversion_poc.bloom import bloom_encode, bloom_decode
from stress_tests.harness import print_test_header

def test_logic_and_contradictions():
    print_test_header("ST3/4 - Logic & Contradictions")

    vocabulary = ["pure_function", "side_effect", "http_get", "k8s_deployment", "error_boundary"]
    m = 256
    k = 3
    
    # ST3: Degenerate Intersection
    print("\nScenario: ST3 - Unrelated Domain Intersection (AND)")
    sketch_k8s = bloom_encode({"k8s_deployment", "error_boundary"}, m, k)
    sketch_api = bloom_encode({"http_get", "error_boundary"}, m, k)
    
    intersect_bits = [k8 & ap for k8, ap in zip(sketch_k8s, sketch_api)]
    intersect_decoded = bloom_decode(intersect_bits, vocabulary, k)
    
    print(f"   K8s ∩ API -> {intersect_decoded}")
    if intersect_decoded == {"error_boundary"}:
        print("   ✅ PASS: Correctly collapsed to shared 'error_boundary'.")
    else:
        print("   ❌ FAIL: Nonsensical intersection results.")

    # ST4: Overconstrained Sketch (Contradiction)
    print("\nScenario: ST4 - Semantic Contradiction (Pure vs SideEffect)")
    # A function can't be both pure and have a side effect
    contradiction = {"pure_function", "side_effect"}
    print(f"   Intent: {contradiction}")
    
    # Synthesis normally just joins them, but a hardened system should halt
    # Here we check if our "Proof Trace" exposes the conflict
    from witness_ts_codegen_poc.ts_api_generator import generate_ts_api_endpoint
    code = generate_ts_api_endpoint(contradiction, with_proof=True)
    
    if "pure_function" in code and "side_effect" in code:
        print("   🔴 OVERCONSTRAINED FAIL: System blindly generated code for contradictory witnesses.")
        print("   Resolution: Implement 'Unsat Witness Conflict' rules.")
    else:
        print("   ✅ SUCCESS: System correctly halted or flagged conflict.")

if __name__ == "__main__":
    test_logic_and_contradictions()
