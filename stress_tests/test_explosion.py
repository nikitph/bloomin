from witness_ts_codegen_poc.ts_api_generator import generate_ts_api_endpoint
from stress_tests.harness import print_test_header

def test_witness_explosion():
    print_test_header("ST1 - Witness Explosion")
    
    # 15+ complex overlapping witnesses
    exploding_intent = {
        "http_post", "http_get", "http_put", "http_delete", # Method explosion
        "zod_schema", "body_validation", "param_validation", # Validation explosion
        "jwt_auth", "role_protected", "api_key_required", # Security explosion
        "db_save", "async_handler", "error_boundary", # Logic explosion
        "json_response", "typed_interface", "pagination_metadata", # Data explosion
        "low_latency", "batch_processing" # Optimization explosion
    }
    
    print(f"Goal: Synthesize from {len(exploding_intent)} concurrent witnesses.")
    
    try:
        code = generate_ts_api_endpoint(exploding_intent)
        print("✅ SUCCESS: Generator produced code without crashing.")
        print("Final Code Body Preview (first 10 lines):")
        print("\n".join(code.split("\n")[:10]))
        
        # Check for traces
        if "[WITNESS_TRACE]" in code:
             num_traces = code.count("- ")
             print(f"Detected {num_traces} verified witness traces.")
        else:
             print("❌ FAILURE: Witness traces missing under load.")
             
    except Exception as e:
        print(f"❌ FAILURE: Generator crashed under load: {str(e)}")

if __name__ == "__main__":
    test_witness_explosion()
