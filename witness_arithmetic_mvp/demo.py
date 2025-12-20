from witness_arithmetic_mvp.arithmetic import semantic_diff
import json

def run_demo():
    input_a = """
def sort(arr):
    return quicksort(arr)
"""

    input_b = """
def sort(arr):
    if len(arr) < 10:
        return insertion_sort(arr)
    return quicksort(arr)
"""

    result = semantic_diff(input_a, input_b)
    
    # Pretty print JSON Result
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    run_demo()
