from witness_arithmetic_mvp.ast_extractor import extract_witnesses

DATASET = [
    {
        "code": "if x > 0: return x",
        "expected": {"conditional", "gt", "early_return"} # Assuming simplistic matching, adjust if needed
    },
    {
        "code": "for i in range(10): pass",
        "expected": {"loop"}
    },
    {
        "code": "def foo(x): return foo(x-1)",
        "expected": {"recursion", "sub"}
    },
    {
        "code": "if len(arr) < 5: pass",
        "expected": {"conditional", "small_input_optimization", "lt"}
    }
]

def run_benchmark():
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for i, item in enumerate(DATASET):
        code = item["code"]
        expected = item["expected"]
        
        predicted = extract_witnesses(code)
        
        # Soft match: intersection over union or just sets?
        # Let's verify strict set matching for benchmark metrics
        
        tp = len(predicted.intersection(expected))
        fp = len(predicted - expected)
        fn = len(expected - predicted)
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        print(f"Snippet {i+1}:")
        print(f"  Expected: {expected}")
        print(f"  Predicted: {predicted}")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}")
        print("-" * 20)
        
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Overall Precision: {precision:.2f}")
    print(f"Overall Recall: {recall:.2f}")
    print(f"Overall F1 Score: {f1:.2f}")

if __name__ == "__main__":
    run_benchmark()
