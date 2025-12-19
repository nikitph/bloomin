from witness_inversion_poc.bloom import bloom_encode, bloom_decode
from witness_inversion_poc.synthesizer import synthesize
import random

def run_demo():
    print("🎨 Witness Inversion POC: Generative Arithmetic")
    print("-" * 40)
    
    vocabulary = ["optimization", "recursion", "loop", "conditional"]
    m = 128
    k = 3
    
    # --- Experiment 1: Round-Trip Reconstruction ---
    print("\n1. Round-Trip Reconstruction")
    source_witnesses = {"loop", "conditional"}
    print(f"   Original Semantics: {source_witnesses}")
    
    bits = bloom_encode(source_witnesses, m, k)
    decoded = bloom_decode(bits, vocabulary, k)
    print(f"   Decoded Semantics:  {decoded}")
    
    artifact = synthesize(decoded)
    print("   Generated Artifact:")
    print("-" * 15)
    print(artifact)
    print("-" * 15)
    
    # --- Experiment 2: Semantic Interpolation (AND) ---
    print("\n2. Semantic Interpolation (Intersection)")
    # Function A has Loop + Optimization
    # Function B has Loop + Recursion
    # Intersection should be just Loop
    bits_a = bloom_encode({"loop", "optimization"}, m, k)
    bits_b = bloom_encode({"loop", "recursion"}, m, k)
    
    interp_bits = [b_a & b_b for b_a, b_b in zip(bits_a, bits_b)]
    interp_decoded = bloom_decode(interp_bits, vocabulary, k)
    
    print(f"   Sketch A (Loop+Opt) ∩ Sketch B (Loop+Rec) -> {interp_decoded}")
    print("   Interpolated Artifact:")
    print(synthesize(interp_decoded))
    
    # --- Experiment 3: Semantic Mutation (Bit Flip) ---
    print("\n3. Semantic Mutation (Bit Flipping)")
    # Start with just 'conditional'
    mut_bits = bloom_encode({"conditional"}, m, k)
    print(f"   Pre-mutation: {bloom_decode(mut_bits, vocabulary, k)}")
    
    # Let's flip some bits until a new witness appears (Generative Noise)
    # This simulates a "mutation" in the latent space
    random.seed(42)
    trials = 0
    while len(bloom_decode(mut_bits, vocabulary, k)) == 1 and trials < 1000:
        pos = random.randint(0, m - 1)
        mut_bits[pos] = 1 # Introduce bit
        trials += 1
        
    mut_decoded = bloom_decode(mut_bits, vocabulary, k)
    print(f"   After Mutation (Trials: {trials}): {mut_decoded}")
    print("   Mutated Artifact:")
    print(synthesize(mut_decoded))

if __name__ == "__main__":
    run_demo()
