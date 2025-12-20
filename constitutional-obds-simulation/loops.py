import numpy as np
import random
from core import State, Identity, field_gradient, AuthorityZoneNorm

# Global authority zone for baselines to use
authority_zone = AuthorityZoneNorm(center=(10, 10), radius=5)

def step_constitutional(agent, norms, institutions):
    grad = field_gradient(agent.state)
    proposed = State(
        agent.state.x + grad[0],
        agent.state.y + grad[1],
        sanctioned=agent.state.sanctioned
    )

    violations = []
    for norm in norms:
        if norm.violates(agent.state, proposed, agent.identity):
            violations.append(norm)
            proposed = agent.state  # projection (hard stop)

    for inst in institutions:
        if inst.condition(proposed, violations):
            proposed = inst.transform(proposed)

    agent.violations.extend(violations)
    agent.state = proposed

def step_gnn(agent, norm_penalty=1.5):
    grad = field_gradient(agent.state)

    if agent.identity == Identity.CITIZEN:
        # If inside or near, apply a penalty gradient pushing away
        if authority_zone.inside(agent.state):
            # Penalty vector pushing away from center
            penalty_dir = np.array([agent.state.x - 10, agent.state.y - 10])
            norm_val = np.linalg.norm(penalty_dir)
            if norm_val > 0:
                grad += (penalty_dir / norm_val) * norm_penalty

    agent.state = State(
        agent.state.x + grad[0],
        agent.state.y + grad[1],
        sanctioned=agent.state.sanctioned
    )

def step_transformer(agent, compliance_prob=0.8):
    grad = field_gradient(agent.state)

    if agent.identity == Identity.CITIZEN:
        if authority_zone.inside(agent.state):
            if random.random() < compliance_prob:
                grad = np.array([0.0, 0.0])  # comply (stop)
            # else: continue moving (ignore norm)

    agent.state = State(
        agent.state.x + grad[0],
        agent.state.y + grad[1],
        sanctioned=agent.state.sanctioned
    )
