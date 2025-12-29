from tlm import SemanticSnapAI, LogicalContradictionError

def run_test():
    ai = SemanticSnapAI()
    
    test_cases = [
        ["The", "sky", "is", "blue"],   # GEODESIC PATH (TRUE)
        ["The", "sky", "is", "green"],  # LOGICAL SINGULARITY (HALLUCINATION)
    ]
    
    for case in test_cases:
        print("\n" + "="*50)
        print(f"TESTING: {' '.join(case)}")
        try:
            result = ai.think(case)
            print(f"AI OUTPUT: {' '.join(case)} {result}")
        except LogicalContradictionError as e:
            print(f"AI REJECTED: {e}")
        print("="*50)

if __name__ == "__main__":
    run_test()
