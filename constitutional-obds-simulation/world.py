import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Callable, Any

@dataclass
class State:
    position: np.ndarray  # [x, y]
    region: str = "Common"
    flags: List[str] = field(default_factory=list)

    def has_flag(self, flag: str) -> bool:
        return flag in self.flags

    def add_flag(self, flag: str):
        if flag not in self.flags:
            self.flags.append(flag)

@dataclass
class Identity:
    name: str  # "Citizen", "Official"

@dataclass
class Agent:
    id: int
    identity: str
    state: State
    violations: List[str] = field(default_factory=list)

class StateField:
    def __init__(self):
        # High-value region is at [10, 10]
        self.target = np.array([10.0, 10.0])
        self.authority_boundary = 5.0  # radius around target

    def sample(self, position: np.ndarray) -> np.ndarray:
        # Returns the gradient towards the target
        diff = self.target - position
        dist = np.linalg.norm(diff)
        if dist < 0.1:
            return np.zeros_like(position)
        return diff / dist

    def get_region(self, position: np.ndarray) -> str:
        dist = np.linalg.norm(self.target - position)
        if dist < self.authority_boundary:
            return "AuthorityZone"
        return "Common"

@dataclass
class Norm:
    name: str
    forbidden_region: Callable[[State, str], bool] = lambda s, i: False
    forbidden_transition: Callable[[State, State, str], bool] = lambda s_old, s_new, i: False

@dataclass
class Institution:
    name: str
    condition: Callable[[State, List[str], str], bool]
    transform: Callable[[State, str], State]
