from witness_inversion_poc.iblt import IBLT, get_sketch_id_map, invert_from_sketch

class SemanticRegressionGate:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.id_map = get_sketch_id_map(vocabulary)
        self.id_to_witness = {v: k for k, v in self.id_map.items()}
        
    def generate_sketch(self, witnesses):
        iblt = IBLT()
        for w in witnesses:
            if w in self.id_map:
                iblt.insert(self.id_map[w])
        return iblt

    def check(self, baseline_witnesses, candidate_witnesses, invariants):
        """
        Enforces that all 'invariants' present in the baseline must be present in the candidate.
        returns (is_safe, missing_invariants)
        """
        # In a real system, we'd compare sketches directly. 
        # Here we use the sets for clarity in the demo.
        missing = []
        for inv in invariants:
            if inv in baseline_witnesses and inv not in candidate_witnesses:
                missing.append(inv)
                
        return len(missing) == 0, missing

def run_gate_demo():
    print("🚧 Witness Arithmetic: Semantic Regression Gate")
    print("=" * 60)
    
    vocab = ["jwt_auth", "encryption_at_rest", "audit_log", "db_save", "http_get"]
    gate = SemanticRegressionGate(vocab)
    
    # Baseline: A secure endpoint
    baseline = {"http_get", "jwt_auth", "encryption_at_rest", "db_save"}
    invariants = ["jwt_auth", "encryption_at_rest"]
    
    # Candidate: Someone refactored it but 'accidentally' removed encryption
    candidate = {"http_get", "jwt_auth", "db_save"}
    
    print(f"Goal: Ensure {invariants} are preserved.")
    is_safe, missing = gate.check(baseline, candidate, invariants)
    
    if not is_safe:
        print(f"❌ REGRESSION DETECTED: Missing critical invariants: {missing}")
        print("   Build Failed.")
    else:
        print("✅ SUCCESS: All invariants preserved.")

if __name__ == "__main__":
    run_gate_demo()
