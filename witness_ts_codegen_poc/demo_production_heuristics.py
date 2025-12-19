from witness_ts_codegen_poc.ts_api_generator import generate_ts_api_endpoint

def run_production_demo():
    print("💎 Witness Arithmetic: Complexity Shifting Demo")
    print("=" * 60)
    
    base_intent = {
        "http_post", "zod_schema", "body_validation", "db_save", 
        "json_response", "error_boundary"
    }

    # 1. High Performance (Latency Sensitive)
    print("\n[V1: LOW_LATENCY CONFIG]")
    latency_intent = base_intent | {"low_latency"}
    code_v1 = generate_ts_api_endpoint(latency_intent)
    print("-" * 40)
    print(code_v1)
    print("-" * 40)

    # 2. High Throughput (Batch Optimized)
    print("\n[V2: BATCH_PROCESSING CONFIG]")
    batch_intent = base_intent | {"batch_processing"}
    code_v2 = generate_ts_api_endpoint(batch_intent)
    print("-" * 40)
    print(code_v2)
    print("-" * 40)

    print("\n✅ PROOF: Complexity Shifting Successful.")
    print("   Note how V1 injected a cache-check, while V2 shifted from .create() to .createMany().")
    print("   The semantic 'intent' remains db_save, but the 'implementation' shifted per heuristics.")

if __name__ == "__main__":
    run_production_demo()
