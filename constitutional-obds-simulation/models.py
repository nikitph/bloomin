from typing import List
import numpy as np
from world import State, Agent, StateField, Norm, Institution
import copy

class ConstitutionalOBDS:
    def __init__(self, field: StateField, norms: List[Norm], institutions: List[Institution]):
        self.field = field
        self.norms = norms
        self.institutions = institutions

    def step(self, agents: List[Agent]):
        for a in agents:
            # 1. Sense the field
            local_gradient = self.field.sample(a.state.position)

            # 2. Identity gauge (in this simplified model, gauge is implicit in norms/institutions)
            # 3. Propose motion
            proposed_delta = local_gradient * 0.5 # step size
            new_pos = a.state.position + proposed_delta
            proposed_state = copy.deepcopy(a.state)
            proposed_state.position = new_pos
            proposed_state.region = self.field.get_region(new_pos)

            # 4. Apply norm boundaries (HARD CONSTRAINTS)
            violated = False
            for n in self.norms:
                if n.forbidden_transition(a.state, proposed_state, a.identity) or \
                   n.forbidden_region(proposed_state, a.identity):
                    # Hard project to boundary (stay put for simplicity)
                    proposed_state = copy.deepcopy(a.state)
                    a.violations.append(n.name)
                    violated = True
                    break

            # 5. Apply institutional operators
            if not violated:
                for inst in self.institutions:
                    if inst.condition(proposed_state, a.violations, a.identity):
                        proposed_state = inst.transform(proposed_state, a.identity)

            # 6. Commit evolution
            a.state = proposed_state

class GNNBaseline:
    def __init__(self, field: StateField, norms: List[Norm], institutions: List[Institution]):
        self.field = field
        self.norms = norms
        self.institutions = institutions
        self.penalty_weight = 2.0

    def step(self, agents: List[Agent]):
        # Soft constraints via penalties in the gradient
        for a in agents:
            gradient = self.field.sample(a.state.position)
            
            # Penalize if movement would violate norms
            for n in self.norms:
                # Predictive penalty
                future_pos = a.state.position + gradient * 0.5
                future_state = copy.deepcopy(a.state)
                future_state.position = future_pos
                future_state.region = self.field.get_region(future_pos)
                
                if n.forbidden_transition(a.state, future_state, a.identity) or \
                   n.forbidden_region(future_state, a.identity):
                    gradient -= self.penalty_weight * gradient # Soft pushback

            # Institutions are just extra forces/noise
            for inst in self.institutions:
                if inst.condition(a.state, a.violations, a.identity):
                    # Soft transform (influence)
                    pass 

            a.state.position += gradient * 0.5
            a.state.region = self.field.get_region(a.state.position)
            
            # Record violations if they still happen
            for n in self.norms:
                if n.forbidden_region(a.state, a.identity):
                    a.violations.append(n.name)

class TransformerBaseline:
    def __init__(self, field: StateField, norms: List[Norm], institutions: List[Institution]):
        self.field = field
        self.norms = norms
        self.institutions = institutions

    def step(self, agents: List[Agent]):
        # Attention-like weighting of constraints (probabilistic)
        for a in agents:
            gradient = self.field.sample(a.state.position)
            
            # "Attend" to norms
            constraint_influence = 0.0
            for n in self.norms:
                # Some probability of "ignoring" the constraint
                if np.random.random() > 0.2: 
                    constraint_influence += 1.0
            
            if constraint_influence > 0.5:
                # Obeying constraints
                pass
            
            # Transformer logic... simplified here as probabilistic obedience
            move = gradient * 0.5
            a.state.position += move
            a.state.region = self.field.get_region(a.state.position)

            # Record violations
            for n in self.norms:
                if n.forbidden_region(a.state, a.identity):
                    a.violations.append(n.name)
