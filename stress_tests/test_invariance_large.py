from witness_ts_codegen_poc.ts_api_extractor_v2 import extract_witnesses_v2 as extract_witnesses_ts_api
from stress_tests.harness import print_test_header

def test_large_refactor_invariance():
    print_test_header("ST6 - Large-Scale Refactor Invariance")
    
    # Version 1: Standard layout
    code_v1 = """
    import { Router } from 'express';
    import { z } from 'zod';
    const UserSchema = z.object({ name: z.string() });
    const router = Router();
    router.post('/users', async (req, res) => {
        try {
            const val = UserSchema.parse(req.body);
            await db.user.save(val);
            res.json({ success: true });
        } catch (e) {
            res.status(500).json({ error: e.message });
        }
    });
    """
    
    # Version 2: Massive Syntactic Churn (Renaming, Reordering, Extracted functions, Formatting)
    code_v2 = """
    const S = z.object({ name: z.string() }); // Renamed schema
    async function persistentLayer(d) { return await db.user.save(d); } // Inlined logical unit
    
    const r = Router(); // Renamed router
    
    /* 
       Adversarial Noise Comments 
    */
    r.post('/users', async (request_obj, response_obj) => {
        // Validation moved to variable
        const output = S.parse(request_obj.body);
        
        try {
            await persistentLayer(output);
            
            return response_obj.send({
                success: true,
                metadata: "noise"
            });
        } catch (err_context) {
            const msg = err_context.message;
            response_obj.status(500).json({ error: msg });
        }
    });
    """
    
    print("Goal: Prove Sketch(V1) == Sketch(V2) despite massive syntactic churn.")
    
    w1 = extract_witnesses_ts_api(code_v1)
    w2 = extract_witnesses_ts_api(code_v2)
    
    print(f"V1 Witnesses: {sorted(list(w1))}")
    print(f"V2 Witnesses: {sorted(list(w2))}")
    
    if w1 == w2:
        print("✅ SUCCESS: Semantic Sketch is perfectly invariant to refactoring.")
    else:
        diff = w1 ^ w2
        print(f"❌ FAILURE: Sketch drifted. Delta: {diff}")

if __name__ == "__main__":
    test_large_refactor_invariance()
