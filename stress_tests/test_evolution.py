from witness_inversion_poc.bloom import bloom_encode, bloom_decode
from stress_tests.harness import print_test_header

def test_semantic_evolution():
    print_test_header("ST8 - Evolution Over Time")
    
    m, k = 256, 3
    
    # Version 1 Universe
    universe_v1 = ["http_get", "auth_basic", "json_response"]
    
    # Version 2 Universe (auth_basic is deprecated, joined by auth_jwt)
    universe_v2 = ["http_get", "auth_basic", "auth_jwt", "json_response", "pagination"]
    
    # Scenario: A sketch created in V1
    v1_intent = {"http_get", "auth_basic"}
    sketch_old = bloom_encode(v1_intent, m, k)
    
    print("Goal: Decode V1 sketch using a V2 vocabulary.")
    
    # Attempt decoding with V2
    decoded_in_v2 = bloom_decode(sketch_old, universe_v2, k)
    
    print(f"   V1 Sketch: {v1_intent}")
    print(f"   Decoded in V2: {decoded_in_v2}")
    
    if v1_intent.issubset(decoded_in_v2):
        print("✅ SUCCESS: Backward compatibility preserved.")
    else:
        print("❌ FAILURE: Old semantics lost in new vocabulary.")

if __name__ == "__main__":
    test_semantic_evolution()
