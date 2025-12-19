from witness_ts_codegen_poc.ts_api_extractor_v2 import extract_witnesses_v2
from stress_tests.harness import print_test_header

def test_effect_masking_via_indirection():
    print_test_header("ST9 - Effect Masking via Indirection")
    
    # The DB call is buried deep inside a nested function
    code = """
    async function handler(request_obj, response_obj) {
        async function middlewareA(data) {
            async function internal_persist(payload) {
                return await db.user.create(payload); // Semantic Sink
            }
            return await internal_persist(data);
        }
        await middlewareA(request_obj.body);
        response_obj.send({ok: true});
    }
    """
    
    print("Goal: Detect 'db_save' and 'body_validation' despite indirection and renaming.")
    
    witnesses = extract_witnesses_v2(code)
    print(f"Extracted Witnesses: {sorted(list(witnesses))}")
    
    # In our v2 extractor, we search for sink calls (create) and shadowed usage
    pass_db = "db_save" in witnesses
    pass_body = "body_validation" in witnesses
    
    if pass_db and pass_body:
        print("✅ SUCCESS: Semantic witnesses leaked through indirection layers.")
    else:
        if not pass_db: print("❌ FAILURE: Missing 'db_save' (Lost in indirection)")
        if not pass_body: print("❌ FAILURE: Missing 'body_validation' (Lost in variable renaming)")

if __name__ == "__main__":
    test_effect_masking_via_indirection()
