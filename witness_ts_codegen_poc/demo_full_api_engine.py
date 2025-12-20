from witness_ts_codegen_poc.ts_api_generator import generate_ts_api_endpoint
from witness_ts_codegen_poc.ts_api_extractor import extract_witnesses_ts_api

def run_engine_demo():
    print("🚀 Witness Arithmetic: Full TS API Generation Engine")
    print("=" * 60)
    
    # 1. Define High-Level Intent (The Semantic Sketch)
    # We want a Secure, Validated, Async PUT endpoint with generic types and error handling.
    intent = {
        "http_put",
        "route_params",
        "zod_schema",
        "body_validation",
        "jwt_auth",
        "role_protected",
        "async_handler",
        "error_boundary",
        "json_response",
        "typed_interface",
        "resource_id",
        "db_save"
    }
    
    print("\n[STEP 1: INTENT]")
    print(f"Goal: Generate a production-ready API with: {sorted(list(intent))}")

    # 2. GENERATE (Synthesis)
    print("\n[STEP 2: SYNTHESIS]")
    code = generate_ts_api_endpoint(intent)
    print("-" * 40)
    print(code)
    print("-" * 40)

    # 3. EXTRACT (Verification)
    print("\n[STEP 3: RE-EXTRACTION (Verification)]")
    extracted = extract_witnesses_ts_api(code)
    
    print(f"Extracted Semantics: {sorted(list(extracted))}")
    
    # 4. PARITY CHECK
    missing = intent - extracted
    extra = extracted - intent
    
    if not missing:
        print("\n✅ SUCCESS: Full semantic parity achieved.")
        print("   The code perfectly represents the intended structural constraints.")
    else:
        print(f"\n❌ FAILURE: Missing witnesses: {missing}")
        
    if extra:
        print(f"   Note: Found additional context/bits: {extra}")

if __name__ == "__main__":
    run_engine_demo()
