import torch
import numpy as np
from utils import set_seed, geometric_mean, random_peaked_distribution
from gauge_math import GaugeMath

def run_semantic_algebra():
    print("="*60)
    print("Experiment 2: Semantic Algebra (King - Man + Woman = Queen)")
    print("="*60)
    
    set_seed(42)
    dim = 128
    
    # 1. Generate Base Concepts (Latent Attributes)
    print("Generating base concepts...")
    base = {}
    base['Male'] = random_peaked_distribution(dim, peak_loc=20)
    base['Female'] = random_peaked_distribution(dim, peak_loc=40)
    base['Royal'] = random_peaked_distribution(dim, peak_loc=60)
    base['Commoner'] = random_peaked_distribution(dim, peak_loc=80)
    
    # 2. Generate Compound Concepts (Words)
    # Using geometric mean (Log-Linear Mixing)
    words = {}
    words['King'] = geometric_mean(base['Male'], base['Royal'])
    words['Queen'] = geometric_mean(base['Female'], base['Royal'])
    words['Man'] = geometric_mean(base['Male'], base['Commoner'])
    words['Woman'] = geometric_mean(base['Female'], base['Commoner'])
    
    print("Words generated: King, Queen, Man, Woman")
    
    # 3. Perform Algebra
    # Target: King - Man + Woman = Queen
    # In Log-Space: log(K) - log(M) + log(W)
    # = 0.5(Male+Royal) - 0.5(Male+Commoner) + 0.5(Female+Commoner)
    # = 0.5(Royal - Commoner + Female + Commoner)
    # = 0.5(Royal + Female)
    # = log(Queen)
    
    # Implementation in Probability Space:
    # Result = King * Woman / Man
    
    p_king = words['King']
    p_man = words['Man']
    p_woman = words['Woman']
    p_queen_true = words['Queen']
    
    # Algebra Operation
    # Add epsilon to avoid division by zero
    p_result = (p_king * p_woman) / (p_man + 1e-10)
    p_result = p_result / (p_result.sum() + 1e-10) # Renormalize
    
    # 4. Evaluation
    dist = GaugeMath.distance(p_result, p_queen_true).item()
    
    # Also compare with random baseline
    p_random = random_peaked_distribution(dim, peak_loc=100)
    dist_random = GaugeMath.distance(p_result, p_random).item()
    
    print(f"\nDistance to True Queen: {dist:.4f}")
    print(f"Distance to Random:     {dist_random:.4f}")
    
    # Cosine Similarity check
    cos_sim = torch.nn.functional.cosine_similarity(
        p_result.unsqueeze(0), 
        p_queen_true.unsqueeze(0)
    ).item()
    
    print(f"Cosine Similarity:      {cos_sim:.4f}")
    
    # Check if "Result" is closer to "Queen" than "King"
    dist_to_king = GaugeMath.distance(p_result, p_king).item()
    print(f"Distance to King:       {dist_to_king:.4f}")
    
    results = {
        "dist_target": dist,
        "dist_random": dist_random,
        "cosine": cos_sim,
        "success": dist < 0.1
    }
    
    # Save to file
    with open("ALGEBRA_RESULTS.md", "w") as f:
        f.write("# Semantic Algebra Results\n\n")
        f.write(f"- **Equation**: King - Man + Woman = Queen\n")
        f.write(f"- **Method**: Geometric Mean Algebra ($K \\cdot W / M$)\n")
        f.write(f"- **Recovery Error**: {dist:.4f}\n")
        f.write(f"- **Cosine Accuracy**: {cos_sim:.4f}\n")
        f.write(f"- **Baseline Distance**: {dist_random:.4f}\n\n")
        f.write("## Conclusion\n")
        if dist < 0.1:
            f.write("✅ **Valid.** The Concept Manifold supports vector-like algebra in log-space.\n")
        else:
            f.write("❌ **Invalid.** The algebraic relationship does not hold.\n")
            
    return results

if __name__ == "__main__":
    run_semantic_algebra()
