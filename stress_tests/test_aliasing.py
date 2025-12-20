from witness_ts_codegen_poc.ts_api_extractor_v2 import extract_witnesses_v2 as extract_witnesses_ts_api
from stress_tests.harness import print_test_header

def test_semantic_aliasing():
    print_test_header("ST2 - Semantic Aliasing")
    
    # Program A: Non-destructive
    code_a = "async function handle() { await archive_all(); res.json({status: 'archived'}); }"
    
    # Program B: Destructive (Identical AST shape/flow)
    code_b = "async function handle() { await delete_all(); res.json({status: 'deleted'}); }"
    
    print("Goal: Differentiate 'archive' from 'delete' in identical AST shapes.")
    
    # The current POC extractor is basic, so this test's "Expected Failure"
    # will document the need for 'Semantic Sinks'.
    
    w_a = extract_witnesses_ts_api(code_a)
    w_b = extract_witnesses_ts_api(code_b)
    
    print(f"Program A witnesses: {w_a}")
    print(f"Program B witnesses: {w_b}")
    
    delta = w_a ^ w_b
    if not delta:
        print("🔴 ALIASING DETECTED: System treated Archive and Delete as identical.")
        print("   Resolution: Move to Semantic Sinks / Data-flow analysis.")
    else:
        print("✅ SUCCESS: Systems successfully differentiated by intent.")

if __name__ == "__main__":
    test_semantic_aliasing()
