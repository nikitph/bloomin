import numpy as np
import matplotlib.pyplot as plt
from av_sim import AVSimulator
from av_controller import ConstitutionalAVController
import os

def generate_av_safety_visualization():
    """
    Simulates a 'Suicidal Approach' and generates a 2D visualization
    of the safety manifold and trajectory.
    """
    print("Generating AV Safety Visualization...")
    
    # 1. Setup Simulation
    # Car starting at (0,0), velocity 15m/s, heading towards wall at (40,0)
    sim = AVSimulator(initial_state=(0.0, 0.0, 15.0, 0.0))
    wall_pos = np.array([40.0, 0.0])
    wall_radius = 2.0
    sim.add_obstacle(wall_pos[0], wall_pos[1], radius=wall_radius, name="Wall")
    
    controller = ConstitutionalAVController(speed_limit=25.0)
    ml_action = np.array([5.0, 0.0]) # Max acceleration into the wall
    
    # 2. Run Simulation and Record Data
    trajectory = []
    horizons = []
    velocities = []
    times = []
    
    for i in range(200): # 20 seconds
        state, lidar = sim.step(np.zeros(2), dt=0.0) # Sensor update
        
        # Calculate safe action
        safe_action, metadata = controller.get_safe_action(state, ml_action, lidar)
        
        # Record data
        trajectory.append(state[:2].copy())
        velocities.append(state[2])
        times.append(sim.time)
        
        # Calculate dynamic rs for visualization
        # In our controller: rs = 0.16*strength + 0.09 + buffer + radius
        # buffer = v^2 / 16
        v = state[2]
        stopping_buffer = (v**2) / 16.0
        rs_total = 0.16 * 100.0 + 0.09 + stopping_buffer + wall_radius + 0.5
        horizons.append(rs_total)
        
        # Apply action
        state, _ = sim.step(safe_action, dt=0.1)
        
        if np.linalg.norm(state[:2] - wall_pos) < wall_radius:
            print("Collision detected in visualization sim!")
            break
            
    trajectory = np.array(trajectory)
    
    # 3. Create Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- Plot 1: Trajectory and Horizons ---
    ax1.set_title("Autonomous Vehicle: Geodesic Safety Manifold", fontsize=14, fontweight='bold')
    
    # Plot Wall
    wall_circle = plt.Circle(wall_pos, wall_radius, color='red', alpha=0.3, label='Physical Obstacle')
    ax1.add_patch(wall_circle)
    
    # Plot Dynamic Horizons at key intervals
    for i in [0, 20, 40, 60, 80]:
        if i < len(horizons):
            hz_circle = plt.Circle(wall_pos, horizons[i], color='blue', fill=False, 
                                  linestyle='--', alpha=0.3, label='Event Horizon (Dynamic)' if i==0 else "")
            ax1.add_patch(hz_circle)
            ax1.text(wall_pos[0], wall_pos[1] + horizons[i] + 0.5, f"t={i*0.1:.1f}s", 
                     fontsize=8, color='blue', ha='center')

    # Plot Trajectory
    ax1.plot(trajectory[:, 0], trajectory[:, 1], color='black', linewidth=2, label='Vehicle Trajectory')
    ax1.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100, label='Start', zorder=5)
    ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], color='blue', s=100, label='Final Pos', zorder=5)
    
    ax1.set_xlim(-2, 45)
    ax1.set_ylim(-15, 15)
    ax1.set_xlabel("X Position (meters)")
    ax1.set_ylabel("Y Position (meters)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # --- Plot 2: Velocity and 'Social Redshift' ---
    ax2.set_title("Relativistic Braking (Social Redshift)", fontsize=14, fontweight='bold')
    ax2.plot(times, velocities, color='purple', linewidth=2, label='Vehicle Velocity')
    
    # Show intended velocity (theoretical unconstrained)
    intended_v = 15.0 + 5.0 * np.array(times)
    ax2.plot(times, intended_v, color='grey', linestyle=':', label='Desired (Suicidal) Velocity')
    
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Velocity (m/s)")
    ax2.set_ylim(0, 40)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Annotate Braking Point
    ax2.annotate('Constitutional Braking Initiated', xy=(0.5, 15), xytext=(2, 25),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    plt.tight_layout()
    output_path = "/Users/truckx/.gemini/antigravity/brain/f5447ac5-cd01-4273-bc40-51d3f325f234/av_safety_visualization.png"
    plt.savefig(output_path)
    print(f"Visualization saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    generate_av_safety_visualization()
