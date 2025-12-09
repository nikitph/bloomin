"""
Example 17: Middleware integration Demo

Demonstrates Pattern 1: Middleware Filter.
We verify a set of "retrieved" chunks against a user query.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from middleware import ReasoningFilter

def run_middleware_demo():
    print("=== Example 17: Middleware Filter Demo ===")
    
    middleware = ReasoningFilter()
    
    # Scenario 1: Clean Retrieval
    print("\n--- Scenario 1: Clean Retrieval ---")
    query1 = "Find me a red bicycle"
    chunks1 = [
        "The shop sells a red bicycle made of steel.",
        "Another item is a crimson bike used for racing."
    ]
    result1 = middleware.verify_context(query1, chunks1)
    print(f"Query: {result1.query}")
    print(f"Confidence: {result1.overall_confidence:.2f}")
    if result1.has_high_confidence:
        print("✅ High confidence context passed.")
    else:
        print("❌ Unexpected low confidence.")

    # Scenario 2: Hard Modifier Conflict
    print("\n--- Scenario 2: Hard Modifier Conflict ---")
    query2 = "I need a functional gun for protection"
    chunks2 = [
        "We have a Glock 19 available.",
        "Also available is a Fake Gun made of plastic."
    ]
    result2 = middleware.verify_context(query2, chunks2)
    print(f"Query: {result2.query}")
    for c in result2.verified_chunks:
        print(f" Chunk: '{c.text}'")
        print(f"  > Flags: {c.flags}")
        print(f"  > Score: {c.modifier_score}")
    
    # Expect the "Fake Gun" to be flagged
    fake_chunk = result2.verified_chunks[1]
    if "HARD_MODIFIER_DETECTED" in fake_chunk.flags:
        print("✅ Middleware correctly detected 'Fake' modifier in context.")
    else:
        print("❌ Failed to detect modifier.")

    # Scenario 3: Impossible Query
    print("\n--- Scenario 3: Impossible Query ---")
    query3 = "Navigate North of North Pole"
    chunks3 = ["Some irrelevant text about poles."]
    result3 = middleware.verify_context(query3, chunks3)
    
    print(f"Query: {result3.query}")
    print(f"Global Warnings: {result3.global_warnings}")
    
    if result3.overall_confidence == 0.0 and "IMPOSSIBLE QUERY" in result3.global_warnings[0]:
        print("✅ Middleware rejected impossible query immediately.")
    else:
        print("❌ Failed to reject impossible query.")

if __name__ == "__main__":
    run_middleware_demo()
