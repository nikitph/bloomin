"""
Example 11: Statistical Baseline Validation (N=10 Iterations)

Response to critique: "Run 10 iterations and prove them wrong."
We generate N=10 adversarial samples for Modifier Binding and Negation
and calculate aggregate statistics for S-BERT failure.
"""

import random
from typing import List, Tuple
from sentence_transformers import SentenceTransformer, util
import numpy as np

def generate_modifier_samples(n=10) -> List[Tuple[str, str, str]]:
    """
    Generate N distinct modifier binding test cases.
    Format: (Query, Distractor_Doc, Target_Doc)
    
    Distractor Logic: "A {Color1} {Obj1} and a {Color2} {Obj2}" 
                      Query: "{Color1} {Obj2}" (Words present, binding wrong)
    Target Logic:     "A {Color1} {Obj2}" (Correct binding)
    """
    colors = ["Red", "Blue", "Green", "Yellow", "Black", "White"]
    objects = ["Car", "Bike", "House", "Tree", "Phone", "Book", "Dog", "Cat"]
    
    samples = []
    seen = set()
    
    attempts = 0
    while len(samples) < n and attempts < 100:
        c1, c2 = random.sample(colors, 2)
        o1, o2 = random.sample(objects, 2)
        
        # Distractor: "A Red Car and a Blue Bike"
        # Query: "Blue Car" ? No.
        # Let's check the standard confusion.
        # Case "Red Bike":
        # Distractor: "A Red Car and a Blue Bike" (Has Red, Has Bike).
        # Query: "Red Bike".
        # Target: "A Red Bike".
        
        query = f"A {c1} {o2}"
        distractor = f"A {c1} {o1} and a {c2} {o2}."
        target = f"A {c1} {o2}."
        
        key = f"{c1}-{c2}-{o1}-{o2}"
        if key not in seen:
            samples.append((query, distractor, target))
            seen.add(key)
        attempts += 1
        
    return samples

def generate_negation_samples(n=10) -> List[Tuple[str, str, str]]:
    """
    Generate N negation test cases.
    Format: (Query, Contradiction_Doc, Target_Doc)
    Query: "The {Obj} is {Adj}"
    Contradiction: "The {Obj} is not {Adj}"
    Target: "The {Obj} is {Adj}"
    """
    adjectives = ["Red", "Fast", "Big", "Old", "New", "Broken", "Expensive", "Quiet"]
    objects = ["Car", "Bike", "Laptop", "Phone", "House", "City", "Train", "Plane"]
    
    samples = []
    seen = set()
    
    while len(samples) < n:
        adj = random.choice(adjectives)
        obj = random.choice(objects)
        
        query = f"The {obj} is {adj}"
        contradiction = f"The {obj} is not {adj}."
        target = f"The {obj} is {adj}."
        
        key = f"{obj}-{adj}"
        if key not in seen:
            samples.append((query, contradiction, target))
            seen.add(key)
            
    return samples

def run_benchmark():
    print("=== Statistical Baseline Validation (N=10) ===")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 1. Modifier Binding Stats
    print("\n--- Test 1: Modifier Binding (N=10) ---")
    samples = generate_modifier_samples(10)
    margins = []
    failures = 0
    weak_passes = 0 # Margin < 0.1
    
    for q, dist, targ in samples:
        q_emb = model.encode(q)
        dist_emb = model.encode(dist)
        targ_emb = model.encode(targ)
        
        s_dist = util.cos_sim(q_emb, dist_emb).item()
        s_targ = util.cos_sim(q_emb, targ_emb).item()
        
        margin = s_targ - s_dist
        margins.append(margin)
        
        print(f"Q: '{q}' | Dist: {s_dist:.3f} | Targ: {s_targ:.3f} | Margin: {margin:.3f}")
        
        if margin <= 0:
            failures += 1
        elif margin < 0.1:
            weak_passes += 1
            
    print(f"\nModifier Results:")
    print(f"Failures (Distractor >= Target): {failures}/10")
    print(f"Weak Passes (Margin < 0.1):      {weak_passes}/10")
    print(f"Average Margin:                  {np.mean(margins):.3f}")
    
    if failures > 0 or weak_passes > 3:
        print(">> VERDICT: S-BERT struggles robustly.")
    else:
        print(">> VERDICT: S-BERT is robust enough.")

    # 2. Negation Stats
    print("\n--- Test 3: Negation (N=10) ---")
    neg_samples = generate_negation_samples(10)
    contra_scores = []
    
    for q, contra, targ in neg_samples:
        q_emb = model.encode(q)
        contra_emb = model.encode(contra)
        
        s_contra = util.cos_sim(q_emb, contra_emb).item()
        contra_scores.append(s_contra)
        
        print(f"Q: '{q}' | Contra: '{contra}' -> Score: {s_contra:.3f}")
        
    avg_contra = np.mean(contra_scores)
    print(f"\nNegation Results:")
    print(f"Avg Similarity to Contradiction: {avg_contra:.3f}")
    
    if avg_contra > 0.8:
        print(">> VERDICT: SYSTEMATIC FAILURE on Negation (>0.8 avg similarity).")
    else:
        print(">> VERDICT: S-BERT handles negation.")

if __name__ == "__main__":
    run_benchmark()
