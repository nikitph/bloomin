from witness_arithmetic_mvp.k8s_extractor import extract_witnesses_k8s
import re

def run_proof():
    print("🚀 Witness Arithmetic vs Regex Proof\n")

    # Case A: Commented Out (YAML parser handles this naturally, but Regex fails)
    case_a = """
apiVersion: v1
kind: Pod
metadata:
  name: commented-pod
spec:
  # encryption: true
  containers: []
"""

    # Case B: False Positive in Labels (The "Triviality" Trap)
    case_b = """
apiVersion: v1
kind: Pod
metadata:
  name: labeled-pod
  labels:
    feature-flag: "encryption-true"
    description: "This pod has encryption set to true in the docs"
spec:
  containers: []
"""

    # Case C: Real Semantic Config
    case_c = """
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  securityContext:
    encryption: true
  containers: []
"""

    scenarios = [("Case A (Comment)", case_a), ("Case B (Label)", case_b), ("Case C (Real)", case_c)]

    for title, content in scenarios:
        print(f"--- {title} ---")
        
        # 1. Regex Approach (The Trivial Way)
        # Regex looking for word 'encryption' followed vaguely by 'true'
        regex_hit = bool(re.search(r"encryption.*true", content, re.IGNORECASE))
        print(f"  [Regex Matcher]   Found 'encryption...true'?  {regex_hit}")
        
        # 2. Witness Arithmetic Approach (The Structural Way)
        witnesses = extract_witnesses_k8s(content)
        struct_hit = "encryption_at_rest" in witnesses
        print(f"  [Witness Engine]  Found 'encryption_at_rest'? {struct_hit}")
        
        if regex_hit and not struct_hit:
            print("  ➡️  RESULT: Regex FALSE POSITIVE (Failed). Witness Engine Correct.")
        elif regex_hit and struct_hit:
            print("  ➡️  RESULT: Both Correct.")
        else:
             print("  ➡️  RESULT: No Match.")
        print()

if __name__ == "__main__":
    run_proof()
