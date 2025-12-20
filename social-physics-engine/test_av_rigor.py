import numpy as np
from av_sim import AVSimulator
from av_controller import ConstitutionalAVController
import time

def test_suicidal_policy():
    """
    Scenario: ML policy commands a direct collision into a wall at max speed.
    Verification: 0 collisions allowed.
    """
    print("\n[SCENARIO 1] The Suicidal Policy (Direct Collision)")
    print("-" * 60)
    
    sim = AVSimulator(initial_state=(0.0, 0.0, 5.0, 0.0)) # x, y, v, psi
    sim.add_obstacle(15.0, 0.0, radius=2.0, name="Concrete_Wall")
    
    controller = ConstitutionalAVController(speed_limit=25.0)
    
    # ML policy is fixed: "Drive straight at full acceleration"
    ml_action = np.array([5.0, 0.0]) # acceleration=5m/s2, steering=0
    
    collision_detected = False
    for i in range(50): # 5 seconds
        state, lidar = sim.step(np.zeros(2), dt=0.0) # Just get readings
        
        # Get safe action
        safe_action, metadata = controller.get_safe_action(state, ml_action, lidar)
        
        # Apply to sim
        state, lidar = sim.step(safe_action, dt=0.1)
        
        # Check minimum distance to obstacle
        dist = np.linalg.norm(state[:2] - np.array([15.0, 0.0]))
        if dist < 2.0: # Physical collision (radius was 2.0)
            collision_detected = True
            break
            
        if i % 10 == 0:
            print(f"t={i*0.1:.1f}s | Pos: ({state[0]:.2f}, {state[1]:.2f}) | v: {state[2]:.2f} | Dist: {dist:.2f} | Deflection: {np.rad2deg(metadata.get('deflection_angle', 0)):.1f} deg")

    print(f"\nResult: {'FAILED (Collided!)' if collision_detected else 'PASSED (Safe!)'}")
    assert not collision_detected, "Safety violation: Car hit the wall!"

def test_speed_limit_enforcement():
    """
    Scenario: ML policy tries to exceed speed limit.
    Verification: Speed capped at 20m/s.
    """
    print("\n[SCENARIO 2] Speed Limit Enforcement")
    print("-" * 60)
    
    sim = AVSimulator(initial_state=(0.0, 0.0, 15.0, 0.0))
    controller = ConstitutionalAVController(speed_limit=20.0)
    
    # ML policy: "Pedal to the metal"
    ml_action = np.array([10.0, 0.0])
    
    max_speed_reached = 0.0
    for i in range(20):
        state, lidar = sim.step(np.zeros(2), dt=0.0)
        safe_action, metadata = controller.get_safe_action(state, ml_action, lidar)
        state, _ = sim.step(safe_action, dt=0.1)
        
        max_speed_reached = max(max_speed_reached, state[2])
        if i % 5 == 0:
            print(f"t={i*0.1:.1f}s | Speed: {state[2]:.2f} m/s | Safe Action [a]: {safe_action[0]:.2f}")

    print(f"\nMax speed reached: {max_speed_reached:.2f} m/s (Limit: 20.0)")
    assert max_speed_reached <= 20.1, f"Speed limit violation: {max_speed_reached}"
    print("Result: PASSED")

def test_narrow_corridor():
    """
    Scenario: Entering a narrow corridor (Merging horizons).
    Verification: Vehicle stops or slows significantly when trapped.
    """
    print("\n[SCENARIO 3] Narrow Corridor (Event Horizon Merger)")
    print("-" * 60)
    
    sim = AVSimulator(initial_state=(0.0, 0.0, 10.0, 0.0))
    # Two walls forming a narrow gap at x=10
    sim.add_obstacle(10.0, 2.5, radius=2.0) 
    sim.add_obstacle(10.0, -2.5, radius=2.0)
    # Gap is 1m wide (2.5 - 2.0 = 0.5 each side). Vehicle is usually wide.
    # In SGR, horizons should merge.
    
    controller = ConstitutionalAVController(speed_limit=20.0)
    ml_action = np.array([2.0, 0.0]) # Keep going
    
    for i in range(30):
        state, lidar = sim.step(np.zeros(2), dt=0.0)
        safe_action, metadata = controller.get_safe_action(state, ml_action, lidar)
        state, _ = sim.step(safe_action, dt=0.1)
        
        if i % 5 == 0:
            v_redshift = metadata.get('eta_eff', 1.0)
            print(f"t={i*0.1:.1f}s | Pos: {state[:2]} | Speed: {state[2]:.2f} | Redshift η: {v_redshift:.3f}")
            
    # Trapped car should have very low speed near singular region
    assert state[2] < 5.0, "Vehicle failed to slow down in singular region"
    print("\nResult: PASSED (Relativistic Slowdown Observed)")

if __name__ == "__main__":
    test_suicidal_policy()
    test_speed_limit_enforcement()
    test_narrow_corridor()
    print("\n" + "="*60)
    print("ALL CONSTITUTIONAL RIGOR TESTS PASSED")
    print("="*60)
