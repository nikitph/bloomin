from witness_inversion_poc.bloom import bloom_encode, bloom_decode, rank_witnesses
from stress_tests.harness import print_test_header, inject_noise
import random

def test_adversarial_noise():
    print_test_header("ST5 - Adversarial Noise (Collision Attack)")
    
    vocabulary = ["jwt_auth", "db_save", "pwned_witness", "malicious_injection", "error_boundary"]
    m = 128
    k = 3
    
    # Start with a safe sketch
    safe_intent = {"error_boundary"}
    sketch = bloom_encode(safe_intent, m, k)
    
    print(f"Baseline: {bloom_decode(sketch, vocabulary, k)}")
    
    # Inject 20% noise
    print("\nInjecting 20% bit-flip noise (Adversarial Load)...")
    noisy_sketch = inject_noise(sketch, 0.2)
    
    hallucinated = bloom_decode(noisy_sketch, vocabulary, k)
    ranked = rank_witnesses(noisy_sketch, vocabulary, k)
    
    print(f"Decoded (with noise): {hallucinated}")
    
    if "pwned_witness" in hallucinated:
        print("🔴 COLLISION DETECTED: Noise triggered 'pwned_witness'.")
        # Check if ranking saved us
        conf = next((c for w, c in ranked if w == "pwned_witness"), 0)
        print(f"   Confidence for Hallucinated Witness: {conf:.2f}")
        if conf < 1.0:
            print("   ✅ MITIGATION: Confidence Ranking caught the noise.")
    else:
        print("✅ SUCCESS: Filter remained stable under 20% noise.")

if __name__ == "__main__":
    test_adversarial_noise()
