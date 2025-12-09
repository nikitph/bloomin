"""
Example 12: Hard Modifier Stress Test (Dense Adversarial Attributes)

Goal: Break S-BERT on dense modifier binding where basic embeddings usually succeed.
Method: Overload sentences with swappable attributes.

Cases:
1. "The expensive red car is fast. The cheap blue bike is slow."
   Query: "Cheap red bike" (Mixes attributes from both objects)
   Target: "The cheap red bike is..." (None here)
   Adversarial Distractor: The doc above.
"""

import sys
import os
from sentence_transformers import SentenceTransformer, util
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from topos import ToposLogic, LocalSection, Proposition

def run_tests():
    print("=== Phase 12: The 'Hard' Modifier Test ===")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 1. Dense Attribute Swap
    print("\n--- Test: Dense Attribute Swap ---")
    
    # Distractor: Has (Expensive, Red, Car, Fast) AND (Cheap, Blue, Bike, Slow)
    distractor_text = "The expensive red car is fast. The cheap blue bike is slow."
    
    # Target: Has (Cheap, Red, Bike)
    target_text = "The cheap red bike is broken."
    
    # Query: "Cheap red bike"
    # Logic:
    # "Cheap" -> matches "Blue Bike" in Distractor
    # "Red" -> matches "Car" in Distractor
    # "Bike" -> matches "Blue Bike" in Distractor
    # S-BERT sees "Cheap", "Red", "Bike" all present in Distractor. 
    # Can it distinguish the *binding*?
    
    query = "Cheap red bike"
    
    print(f"Query: '{query}'")
    print(f"Distractor: '{distractor_text}'")
    print(f"Target:     '{target_text}'")
    
    # S-BERT
    q_emb = model.encode(query, convert_to_tensor=True)
    d_emb = model.encode(distractor_text, convert_to_tensor=True)
    t_emb = model.encode(target_text, convert_to_tensor=True)
    
    score_d = util.cos_sim(q_emb, d_emb).item()
    score_t = util.cos_sim(q_emb, t_emb).item()
    
    print(f"S-BERT Distractor Score: {score_d:.4f}")
    print(f"S-BERT Target Score:     {score_t:.4f}")
    
    margin = score_t - score_d
    print(f"S-BERT Margin: {margin:.4f}")
    
    if margin < 0.1:
        print(">> S-BERT Result: WEAK / CONFUSED (Margin < 0.1)")
    elif score_d > score_t:
        print(">> S-BERT Result: FAIL (Ranked Distractor Higher)")
    else:
        print(">> S-BERT Result: PASS (Robust)")
        
    # Topos Logic
    print("\n--- Topos Logic Check ---")
    logic = ToposLogic()
    
    # Build Section for Distractor
    # "Expensive Red Car" (U1)
    s1 = LocalSection(
        region_id="d_s1",
        witness_ids={"car_1"},
        propositions=[
            Proposition("is_expensive", 1.0, {"car_1"}),
            Proposition("is_red", 1.0, {"car_1"}),
            Proposition("is_car", 1.0, {"car_1"})
        ]
    )
    # "Cheap Blue Bike" (U2)
    s2 = LocalSection(
        region_id="d_s2",
        witness_ids={"bike_1"},
        propositions=[
            Proposition("is_cheap", 1.0, {"bike_1"}),
            Proposition("is_blue", 1.0, {"bike_1"}),
            Proposition("is_bike", 1.0, {"bike_1"})
        ]
    )
    logic.sections[s1.region_id] = s1
    logic.sections[s2.region_id] = s2
    
    # Query logic: Find section with {is_cheap, is_red, is_bike}
    matches = []
    for s in logic.sections.values():
        preds = {p.predicate for p in s.propositions}
        if {"is_cheap", "is_red", "is_bike"}.issubset(preds):
            matches.append(s.region_id)
            
    print(f"Topos Matches for Distractor: {matches}")
    if not matches:
        print(">> Topos Result: PASS (Correct Rejection)")
    else:
        print(">> Topos Result: FAIL (Hallucination)")

if __name__ == "__main__":
    run_tests()
