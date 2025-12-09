"""
Example 16: Reasoning Layer Integration Demo

This script demonstrates the unified `ReasoningLayer` that consolidates:
1. Impossible Query Detection
2. Hard Modifier Handling
3. Advanced Inference (Transitivity, Instantiation)
4. Defeasible Logic
"""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from topos import ToposLogic, LocalSection, Proposition, ReasoningLayer

def run_integrated_demo():
    print("=== Example 16: Reasoning Layer Integration Demo ===")
    
    layer = ReasoningLayer()
    
    # --- 1. Impossible Query Detection ---
    print("\n[1] IMPOSSIBLE QUERY DETECTION")
    queries = [
        {"relation": "North", "target": "Berlin"}, # Valid
        {"relation": "North", "target": "North Pole"} # Invalid
    ]
    
    result = layer.check_query_feasibility(queries)
    if result.status == "impossible":
        print(f"✅ Correctly identified impossible query: {result.explanation}")
    else:
        print(f"❌ Failed detection: {result}")

    # --- 2. Hard Modifier Handling ---
    print("\n[2] HARD MODIFIER HANDLING")
    base_props = {"shoots": 1.0, "heavy": 1.0}
    
    # Test "Fake"
    name, props = layer.apply_modifier("Gun", "Fake", base_props)
    print(f"Applying 'Fake' to 'Gun': {name}")
    print(f"Properties: {props}")
    
    if props.get("shoots") == 0.0 and props.get("is_safe") == 1.0:
        print("✅ Modifier 'Fake' correctly negated 'shoots' and added 'is_safe'.")
    else:
        print("❌ Modifier failed.")

    # --- 3. Inference Engine (Transitivity) ---
    print("\n[3] INFERENCE ENGINE (Transitivity)")
    # A > B, B > C -> A > C
    sec_ab = LocalSection("rel_ab", {"A", "B"}, [Proposition("greater_than", 1.0, {"A", "B"})])
    sec_bc = LocalSection("rel_bc", {"B", "C"}, [Proposition("greater_than", 1.0, {"B", "C"})])
    
    result = layer.infer([sec_ab, sec_bc])
    print(f"Inference Result: {result.status}")
    print(f"Derived Facts: {result.derived_facts}")
    
    if result.status == "success":
        print("✅ Inference Successful (gluing worked).")
    else:
        print("❌ Inference Failed.")

    # --- 4. Advanced Logic (Defeasible) ---
    print("\n[4] DEFEASIBLE LOGIC")
    # Rule: Birds Fly (1.0). Tweety is Bird. Exception: Tweety !Fly (0.0).
    ctx = {
        "rule_property": 1.0,
        "exception_property": 0.0,
        "entity": "Tweety"
    }
    
    result = layer.solve_complex_query("defeasible", ctx)
    if result.status == "exception":
        print(f"✅ Correctly handled exception: {result.explanation}")
    else:
        print(f"❌ Failed defeasible logic: {result}")

    # --- 5. Conditional Logic ---
    print("\n[5] CONDITIONAL LOGIC")
    # Rain + Cold -> Snow
    ctx_cond = {
        "conditions": ["Rain", "Cold"],
        "facts": ["Rain", "Cold"],
        "consequent": "Snow"
    }
    result = layer.solve_complex_query("conditional", ctx_cond)
    if result.status == "success" and "Snow is True" in result.derived_facts:
         print(f"✅ Correctly handled conditional: {result.derived_facts}")
    else:
         print(f"❌ Failed conditional logic: {result}")


if __name__ == "__main__":
    run_integrated_demo()
