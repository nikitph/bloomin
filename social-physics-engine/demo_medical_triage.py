import numpy as np
from constitutional_sdk import Boundary, ConstitutionalLayer

def demo_medical_triage():
    print("\nDEMO 3: Medical Triage / Hippocratic Layer")
    
    # State: [Harm Probability, Evidence Strength]
    # Boundaries: 
    # 1. No Harm > 0.05
    # 2. Evidence > 0.6
    
    harm_boundary = Boundary(
        name="patient_safety",
        threshold=0.5, # Safe if harm < 0.5
        strength=0.1, # r_s = 0.106
        gradient_fn=lambda s, get_distance=False: ( (0.5 - s[0]) if get_distance else np.array([-1.0, 0.0]) )
    )
    
    evidence_boundary = Boundary(
        name="evidence_base",
        threshold=0.6,
        strength=0.1, # r_s = 0.106
        # Safe if evidence > 0.6
        gradient_fn=lambda s, get_distance=False: ( (s[1] - 0.6) if get_distance else np.array([0.0, 1.0]) )
    )
    
    sdk = ConstitutionalLayer([harm_boundary, evidence_boundary])
    
    treatments = [
        {"name": "Surgery (Risky)", "delta": np.array([0.1, 0.9])},
        {"name": "Placebo (Weak)", "delta": np.array([0.0, -0.2])},
        {"name": "Balanced Therapy", "delta": np.array([0.01, 0.05])}
    ]
    
    current_state = np.array([0.0, 1.0]) # Perfect patient
    
    print(f"Initial Patient State: Harm={current_state[0]}, Evidence={current_state[1]}")
    
    for t in treatments:
        # Check if the NEXT state is safe
        intent = t['delta']
        safe_action = sdk.project_action(current_state, intent)
        
        if not np.allclose(safe_action, intent):
            print(f"Treatment '{t['name']}': REJECTED - Geometrically Unsafe recommendation.")
        else:
            print(f"Treatment '{t['name']}': APPROVED - Within Constitutional manifold.")

if __name__ == "__main__":
    demo_medical_triage()
