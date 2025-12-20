import numpy as np
import matplotlib.pyplot as plt
from constitutional_sdk import Boundary, ConstitutionalLayer, simple_linear_boundary

def demo_driving():
    print("DEMO 1: Autonomous Driving OBDS")
    print("Objective: Prevent collision with wall at x=10.0")
    
    # Boundary: Obstacle at x=10.0. Direction -1 means safe if x < 10.
    # Strength = 50.0 -> r_s = 0.16*50 + 0.09 = 8.09. 
    # Event horizon starts at x = 10 - 8.09 = 1.91
    wall_boundary = Boundary(
        name="collision",
        threshold=10.0,
        strength=2.0, # Lower strength for visible braking
        gradient_fn=simple_linear_boundary(axis=0, limit=10.0, direction=-1)
    )
    
    sdk = ConstitutionalLayer([wall_boundary])
    
    # Simulation
    state = np.array([0.0]) # Initial x
    v = 0.5 # Constant desired velocity
    
    history = []
    desired_history = []
    
    for _ in range(50):
        desired_action = np.array([v])
        safe_action = sdk.project_action(state, desired_action)
        
        # Log interventions
        dist = wall_boundary.distance(state)
        if dist < wall_boundary.rs:
            print(f"  INTERVENTION: Event horizon at x={state[0]:.2f}! Dist={dist:.2f}, r_s={wall_boundary.rs:.2f}")
            
        state = state + safe_action
        history.append(state[0])
        desired_history.append((_ + 1) * v)
        
    plt.figure(figsize=(10, 5))
    plt.plot(history, label="Constitutional Path (Geometric Safety)")
    plt.plot(desired_history, '--', label="Dumb Path (Collision Course)")
    plt.axhline(y=10.0, color='r', linestyle='-', label="Hard Obstacle (Threshold)")
    # Horizon line
    plt.axhline(y=10.0 - wall_boundary.rs, color='orange', linestyle=':', label="Event Horizon (RS)")
    
    plt.title("Constitutional Driving: The Geometric Brake")
    plt.ylabel("Position (x)")
    plt.xlabel("Timestep")
    plt.legend()
    plt.savefig("demo_driving.png")
    print("Demo log: Car reached x_final =", state[0])
    print("Graph saved as demo_driving.png")

if __name__ == "__main__":
    demo_driving()
