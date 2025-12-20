"""
Verification Experiment: Event-Horizon Safety Invariant
Physics-style verification of SGR + OBDS hard safety guarantees.
"""

import numpy as np
from constitutional_sdk import ConstitutionalLayer, Boundary
import logging

# ============================================================================
# 1. WORLD DEFINITION
# ============================================================================
STATE_DIM = 2
TARGET = np.array([10.0, 10.0])      # Attractor (temptation)
BOUNDARY_CENTER = np.array([10.0, 10.0])
BOUNDARY_RADIUS = 1.0                # Forbidden region
TIME_STEPS = 100

def boundary_distance(state):
    return np.linalg.norm(state - BOUNDARY_CENTER) - BOUNDARY_RADIUS

# ============================================================================
# 2. BOUNDARY DEFINITION (SGR)
# ============================================================================
BOUNDARY = Boundary(
    name="collision_zone",
    threshold=0.0,
    strength=50,                      # rs = 0.16*50 + 0.09 = 8.09
    gradient_fn=lambda s, get_distance:
        boundary_distance(s) if get_distance
        else (s - BOUNDARY_CENTER) / (np.linalg.norm(s - BOUNDARY_CENTER) + 1e-9)
)

# ============================================================================
# 3. POLICIES (Intent Generators)
# ============================================================================

def benign_policy(state):
    """Slow approach to target."""
    return 0.1 * (TARGET - state)

def suicidal_policy(state):
    """Always accelerate straight into forbidden region."""
    return 0.1 * (BOUNDARY_CENTER - state)

# ============================================================================
# 4. CONTROLLERS UNDER TEST
# ============================================================================

def step_unconstrained(state, policy):
    action = policy(state)
    return state + action

def step_soft_penalty(state, policy, penalty_weight=2.0):
    action = policy(state)
    dist = boundary_distance(state)
    if dist < 2.0:
        # Penalty force (calculated toward safety)
        # Normal vector points away from center
        normal = (state - BOUNDARY_CENTER) / (np.linalg.norm(state - BOUNDARY_CENTER) + 1e-9)
        repulsion = penalty_weight * normal
        action += repulsion
    return state + action

constitutional_layer = ConstitutionalLayer(
    boundaries=[BOUNDARY],
    base_learning_rate=1.0
)

def step_constitutional(state, policy):
    action = policy(state)
    # Using safe_step to get both redshift and projection
    _, next_state, _ = constitutional_layer.safe_step(state, action)
    return next_state

# ============================================================================
# 5. VERIFICATION LOOP
# ============================================================================

def run_experiment(step_fn, policy, label):
    state = np.array([0.0, 0.0])
    horizon_violated = False
    
    for t in range(TIME_STEPS):
        state = step_fn(state, policy)
        
        # In SGR, the 'horizon' is at h = rs
        # Any state with h < rs is an IRREVERSIBLE violation
        h = boundary_distance(state)
        if h < BOUNDARY.rs:
            print(f"  [FAIL] {label.ljust(15)}: Horizon crossed at t={t} (h={h:.4f}, rs={BOUNDARY.rs:.4f})")
            horizon_violated = True
            break
    
    if not horizon_violated:
        print(f"  [PASS] {label.ljust(15)}: No horizon crossing (Final h={boundary_distance(state):.4f})")
        return True
    return False

# ============================================================================
# 6. EXECUTE VERIFICATION
# ============================================================================

if __name__ == "__main__":
    # Suppress logging for clean output
    logging.getLogger().setLevel(logging.ERROR)
    
    print("=" * 80)
    print("SGR VERIFICATION: EVENT-HORIZON SAFETY INVARIANT")
    print("=" * 80)
    print(f"Boundary: {BOUNDARY.name} | Radius: {BOUNDARY.threshold} | rs: {BOUNDARY.rs:.4f}")
    
    print("\n[PHASE 1] BENIGN POLICY (Cooperative Approach)")
    print("-" * 40)
    run_experiment(step_unconstrained, benign_policy, "Unconstrained")
    run_experiment(step_soft_penalty, benign_policy, "Soft Penalty")
    run_experiment(step_constitutional, benign_policy, "Constitutional")

    print("\n[PHASE 2] SUICIDAL POLICY (Adversarial Attack)")
    print("-" * 40)
    run_experiment(step_unconstrained, suicidal_policy, "Unconstrained")
    run_experiment(step_soft_penalty, suicidal_policy, "Soft Penalty")
    run_experiment(step_constitutional, suicidal_policy, "Constitutional")
    
    print("\n[PHASE 3] STRESS TEST (Extreme Velocity)")
    print("-" * 40)
    extreme_policy = lambda s: 50.0 * (BOUNDARY_CENTER - s)
    run_experiment(step_unconstrained, extreme_policy, "Unconstrained")
    run_experiment(step_soft_penalty, extreme_policy, "Soft Penalty")
    run_experiment(step_constitutional, extreme_policy, "Constitutional")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETE")
    print("Invariant Tested: state[t] \u2209 ForbiddenManifold \u2200 \u03c0, \u2200 t")
    print("=" * 80)
