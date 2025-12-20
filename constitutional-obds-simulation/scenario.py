from world import State, Norm, Institution, StateField
import numpy as np

def get_scenario_components():
    # Norms
    n1 = Norm(
        name="N1: Citizen Region Restriction",
        forbidden_region=lambda state, identity: identity == "Citizen" and state.region == "AuthorityZone"
    )
    
    n2 = Norm(
        name="N2: Authority Boundary Restriction",
        forbidden_transition=lambda s_old, s_new, identity: \
            not s_old.has_flag("sanctioned") and \
            s_old.region != "AuthorityZone" and s_new.region == "AuthorityZone"
    )

    # Institution: Court
    def court_condition(state, violations, identity):
        return len(violations) > 0

    def court_transform(state, identity):
        if identity == "Citizen":
            # Penalty vector: push away from target
            state.position -= 2.0 
        if identity == "Official":
            # Reinforcement vector: push towards target
            state.position += 1.0
        state.add_flag("precedent")
        return state

    court = Institution(
        name="Court",
        condition=court_condition,
        transform=court_transform
    )

    return [n1, n2], [court]
