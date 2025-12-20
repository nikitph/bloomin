from witness_inversion_poc.bloom import bloom_encode, rank_witnesses, sample_witness_sets
from witness_inversion_poc.synthesizer import synthesize
import json

def run_advanced_demo():
    print("💎 Witness Inversion: Advanced Verified Generation")
    print("=" * 50)
    
    vocabulary = ["optimization", "recursion", "loop", "conditional", "hashmap", "binary_search"]
    m = 256
    k = 3
    
    # 1. THE SKETCH (Latent Space)
    # Target: A function that uses a loop and a conditional
    source = {"loop", "conditional"}
    sketch = bloom_encode(source, m, k)
    
    # 2. SEMANTIC SEARCH (Inverse Query)
    print("\n🔍 1. Semantic Search (Inverse Query)")
    print("   Query: What is inside this sketch?")
    ranked = rank_witnesses(sketch, vocabulary, k)
    for w, conf in ranked:
        if conf > 0.3:
            status = "✅ [Confirmed]" if conf == 1.0 else "❓ [Probable]"
            print(f"   - {w:15} | Confidence: {conf:.2f} | {status}")

    # 3. VERIFIED SYNTHESIS (Proof-Carrying Generation)
    print("\n📜 2. Verified Synthesis (Proof-Carrying)")
    # Filter for high confidence witnesses
    verified_witnesses = {w for w, conf in ranked if conf == 1.0}
    artifact = synthesize(verified_witnesses, with_proof=True)
    print("   Generated Artifact with Witness Traces:")
    print("-" * 40)
    print(artifact)
    print("-" * 40)

    # 4. MULTI-SAMPLE GENERATION (Creativity from one Sketch)
    print("\n🎨 3. Multi-sample Generation (One Sketch -> Many Snippets)")
    samples = sample_witness_sets(sketch, vocabulary, k, num_samples=3)
    for i, sample in enumerate(samples, 1):
        print(f"\n   Sample {i}: {sample}")
        print("   Snippet:")
        code = synthesize(sample)
        # Just show the first few lines
        first_line = code.split('\n')[0]
        print(f"   > {first_line} ...")

if __name__ == "__main__":
    run_advanced_demo()
