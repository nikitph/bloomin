import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List

@dataclass
class State:
    x: float
    y: float
    sanctioned: bool = False

class Identity(Enum):
    CITIZEN = "citizen"
    OFFICIAL = "official"

class Norm:
    def violates(self, prev: State, next: State, identity: Identity) -> bool:
        raise NotImplementedError

class AuthorityZoneNorm(Norm):
    def __init__(self, center, radius):
        self.cx, self.cy = center
        self.r = radius

    def inside(self, s: State):
        return (s.x - self.cx)**2 + (s.y - self.cy)**2 < self.r**2

    def violates(self, prev, next, identity):
        if identity == Identity.CITIZEN:
            return self.inside(next)
        return False

class Institution:
    def condition(self, state: State, violations: List[Norm]) -> bool:
        raise NotImplementedError

    def transform(self, state: State) -> State:
        raise NotImplementedError

class Court(Institution):
    def condition(self, state, violations):
        return len(violations) > 0

    def transform(self, state):
        # Penalty: push back and clear sanction (for this demo)
        return State(
            x=state.x - 1.0,
            y=state.y - 1.0,
            sanctioned=False
        )

@dataclass
class Agent:
    state: State
    identity: Identity
    violations: List[Norm]

def field_gradient(state: State):
    # gradient toward (10, 10)
    return np.array([10 - state.x, 10 - state.y]) * 0.1
