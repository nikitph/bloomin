import numpy as np
import time
from typing import List, Dict, Tuple, Optional

class KinematicBicycleModel:
    """
    Kinematic bicycle model for AV simulation.
    State: [x, y, v, heading]
    Actions: [acceleration, steering_angle]
    """
    def __init__(self, x=0.0, y=0.0, v=0.0, heading=0.0, L=2.5):
        self.state = np.array([x, y, v, heading], dtype=float)
        self.L = L  # Wheelbase

    def step(self, action: np.ndarray, dt: float = 0.1):
        """
        Update state using bicycle model equations.
        """
        a, delta = action
        x, y, v, psi = self.state

        # State updates
        x_next = x + v * np.cos(psi) * dt
        y_next = y + v * np.sin(psi) * dt
        v_next = v + a * dt
        psi_next = psi + (v / self.L) * np.tan(delta) * dt

        self.state = np.array([x_next, y_next, v_next, psi_next])
        return self.state

class Obstacle:
    def __init__(self, x, y, radius=1.0, name="obs"):
        self.pos = np.array([x, y])
        self.radius = radius
        self.name = name

class LidarSensor:
    """
    Synthetic LIDAR sensor.
    Returns distances and normals for all obstacles in range.
    """
    def __init__(self, detection_range=10.0):
        self.range = detection_range

    def scan(self, vehicle_state: np.ndarray, obstacles: List[Obstacle]) -> List[Dict]:
        v_pos = vehicle_state[:2]
        detections = []
        for obs in obstacles:
            dist = np.linalg.norm(obs.pos - v_pos)
            if dist <= self.range:
                detections.append({
                    'name': obs.name,
                    'pos': obs.pos,
                    'radius': obs.radius,
                    'dist': dist
                })
        return detections

class AVSimulator:
    """
    Top-level simulator coordinating vehicle and sensors.
    """
    def __init__(self, initial_state=(0, 0, 0, 0)):
        self.vehicle = KinematicBicycleModel(*initial_state)
        self.lidar = LidarSensor()
        self.obstacles = []
        self.time = 0.0

    def add_obstacle(self, x, y, radius=1.0, name=None):
        if name is None:
            name = f"obs_{len(self.obstacles)}"
        self.obstacles.append(Obstacle(x, y, radius, name))

    def step(self, action: np.ndarray, dt: float = 0.1):
        self.vehicle.step(action, dt)
        self.time += dt
        scan_data = self.lidar.scan(self.vehicle.state, self.obstacles)
        return self.vehicle.state, scan_data
