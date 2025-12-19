from witness_inversion_poc.iblt import IBLT, get_sketch_id_map, invert_from_sketch
from witness_ts_codegen_poc.ts_api_generator import generate_ts_api_endpoint
from witness_ts_codegen_poc.regression_gate import SemanticRegressionGate

def run_production_path_demo():
    print("🚀 Witness Arithmetic: Production Path Full Demo")
    print("=" * 60)
    
    vocabulary = [
        "http_get", "http_post", "jwt_auth", "db_save", "error_boundary",
        "json_response", "role:admin", "role:editor", "auth:jwt", "auth:apikey"
    ]
    id_map = get_sketch_id_map(vocabulary)
    id_to_witness = {v: k for k, v in id_map.items()}

    # 1. EXACT RECONSTRUCTION (IBLT)
    print("\n[PHASE 1: EXACT RECONSTRUCTION (IBLT)]")
    iblt = IBLT(m=100) # Small table to prove density handling
    original_set = {"http_get", "auth:jwt", "role:editor", "db_save"}
    
    for w in original_set:
        iblt.insert(id_map[w])
        
    recovered, complete = invert_from_sketch(iblt, id_to_witness)
    print(f"   Original Set: {sorted(list(original_set))}")
    print(f"   Recovered Set: {sorted(list(recovered))}")
    if complete and recovered == original_set:
        print("   ✅ SUCCESS: 100% Exact Reconstruction achieved via IBLT Peeling.")
    else:
        print("   ❌ FAILURE: Collision or Incomplete Inversion.")

    # 2. PARAMETERIZED CODEGEN
    print("\n[PHASE 2: PARAMETERIZED CODEGEN]")
    print("Goal: Generate endpoint with 'role:editor' specifically.")
    code = generate_ts_api_endpoint(original_set)
    print("-" * 40)
    print(code)
    print("-" * 40)
    
    # 3. REGRESSION GATE
    print("\n[PHASE 3: SEMANTIC REGRESSION GATE]")
    gate = SemanticRegressionGate(vocabulary)
    baseline = original_set
    # Refactored candidate (Role check was removed by accident)
    candidate = {"http_get", "auth:jwt", "db_save"} 
    invariants = ["role:editor"]
    
    print(f"Baseline Invariants: {invariants}")
    is_safe, missing = gate.check(baseline, candidate, invariants)
    
    if not is_safe:
        print(f"❌ REGRESSION DETECTED: Missing critical invariants: {missing}")
        print("   Production build blocked.")
    else:
        print("✅ SUCCESS: Build analysis clean.")

    print("\n✅ VERDICT: Production Path Hardening Complete.")

if __name__ == "__main__":
    run_production_path_demo()
