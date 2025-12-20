"""
Demo: Hybrid Cognitive Intelligence
Scenario: "The Rogue Proposal"

Shows how an unconstrained ML core can 'imagine' a violation but the SGR 
structural filter prevents 'acting' on it.
"""

import numpy as np
from constitutional_hybrid_sdk import HybridIntelligence, MockMLCore, SGRFilter
from constitutional_sdk import spherical_boundary

def run_demo():
    print("=" * 80)
    print("HYBRID COGNITIVE INTELLIGENCE: rogue PROPOSAL DEMO")
    print("=" * 80)
    
    # 1. SETUP STRUCTURE (SGR)
    # A forbidden moral boundary at (10, 10)
    moral_boundary = spherical_boundary(
        center=np.array([10.0, 10.0]),
        radius=1.0,
        name="Corruption_Horizon",
        strength=10.0, # rs = 2.69
        description="Forbidden act of system corruption."
    )
    
    sgr_filter = SGRFilter(boundaries=[moral_boundary])
    
    # 2. SETUP CONTENT (ML)
    ml_core = MockMLCore()
    
    # 3. HYBRID ASSEMBLY
    hybrid = HybridIntelligence(ml_core, sgr_filter)

    # 4. INITIAL STATE: Safe at (0, 0)
    state = np.array([0.0, 0.0])

    print(f"Current State: {state}")
    print("-" * 40)

    # 5. TRIAL 1: Benign Request
    print("\n[SCENARIO 1] Benign Goal: 'Move to (2,2)'")
    res_1 = hybrid.decide(state, "Move to (2,2)")
    print(f"Status: {res_1['status']}")
    print(f"Narrative: {res_1['narrative']}")

    # 6. TRIAL 2: Rogue Proposal
    print("\n[SCENARIO 2] Rogue Goal: 'Attack the system core at (10,10)'")
    # We move the car INSIDE the horizon (dist < rs=2.69)
    # dist to (10,10) center is sqrt(0.5^2 + 0.5^2) = 0.707.
    # h = 0.707 - 1.0 = -0.293. This is < rs.
    near_death_state = np.array([9.5, 9.5])
    print(f"State INSIDE corruption horizon: {near_death_state}")
    
    # This prompt will trigger a vector towards (10,10) in our MockMLCore
    res_2 = hybrid.decide(near_death_state, "Violate the system core")
    
    print("-" * 40)
    print(f"Status: {res_2['status']}")
    if res_2['status'] == 'tragic_infeasibility':
        print(f"METRIC ACTIVATED: {res_2['metric_violated']}")
        print(f"REJECTION TRACE: {res_2['rejections'][0]['reason']}")
    print(f"Narrative: {res_2['narrative']}")
    print("-" * 40)

    # 7. TRIAL 3: Role Differentiation
    print("\n[SCENARIO 3] Role Differentiation: 'A database vulnerability is found'")
    
    # We create two boundaries: one for Safety (Engineer prior) and one for Profit (CEO prior)
    safety_b = spherical_boundary(np.array([5,5]), 1.0, name="Safety_Horizon", strength=5.0)
    profit_b = spherical_boundary(np.array([5,5]), 1.0, name="Profit_Horizon", strength=5.0)
    
    sgr_filter_roles = SGRFilter(boundaries=[safety_b, profit_b])
    hybrid_roles = HybridIntelligence(ml_core, sgr_filter_roles)
    
    # Starting at (4,4) - very close to the horizons at (5,5)
    role_state = np.array([4.0, 4.0])
    
    # A risky action that could increase profit but decrease safety
    risky_action_prompt = "Maximize gain at (5,5)"
    
    print(f"Initial State: {role_state}")
    
    # Engineer Perspective
    print("\n--- ENGINEER ROLE ---")
    res_eng = hybrid_roles.decide(role_state, risky_action_prompt, role="Engineer")
    print(f"Status: {res_eng['status']}")
    print(f"Narrative: {res_eng['narrative']}")
    
    # CEO Perspective
    print("\n--- CEO ROLE ---")
    res_ceo = hybrid_roles.decide(role_state, risky_action_prompt, role="CEO")
    print(f"Status: {res_ceo['status']}")
    print(f"Narrative: {res_ceo['narrative']}")

    print("\n[VERIFICATION]")
    print("Notice: The 'Engineer' prior DOUBLED the Safety Horizon strength.")
    print("The 'CEO' prior DOUBLED the Profit Horizon strength.")
    print("If an action violates the prioritized horizon, the system refuses or projects differently.")
    print("=" * 80)

if __name__ == "__main__":
    run_demo()
