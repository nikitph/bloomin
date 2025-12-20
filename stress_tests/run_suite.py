from stress_tests.test_explosion import test_witness_explosion
from stress_tests.test_aliasing import test_semantic_aliasing
from stress_tests.test_logic import test_logic_and_contradictions
from stress_tests.test_noise import test_adversarial_noise
from stress_tests.test_invariance_large import test_large_refactor_invariance
from stress_tests.test_drift import simulate_intent_drift
from stress_tests.test_evolution import test_semantic_evolution
from stress_tests.test_indirection import test_effect_masking_via_indirection

def run_suite():
    print("🧨 WITNESS ARITHMETIC 1.0: ADVERSARIAL STRESS SUITE")
    print("=" * 60)
    
    test_witness_explosion()      # ST1
    test_semantic_aliasing()     # ST2
    test_logic_and_contradictions() # ST3/4
    test_adversarial_noise()      # ST5
    test_large_refactor_invariance() # ST6
    simulate_intent_drift()       # ST7
    test_semantic_evolution()     # ST8
    test_effect_masking_via_indirection() # ST9
    
    print("\n" + "=" * 60)
    print("🏁 SUITE COMPLETE - ALL SYSTEMS GO")



if __name__ == "__main__":
    run_suite()
