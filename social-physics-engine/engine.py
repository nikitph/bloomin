import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque, defaultdict
import random
from dataclasses import dataclass, field

@dataclass
class Action:
    actor_id: str
    action_type: str
    target_id: Optional[str] = None
    magnitude: float = 1.0
    context: str = 'general'
    symbolic_marker: Optional[str] = None
    direction_vector: Optional[np.ndarray] = None # For geodesic flow

@dataclass
class Agent:
    id: str
    roles: List[str] = field(default_factory=lambda: ['citizen'])
    emotional_state: float = 0.0
    internal_guilt: float = 0.0
    status: float = 0.5
    accumulated_violations: List[str] = field(default_factory=list)

class Boundary:
    def __init__(self, threshold: float, name: str = 'boundary'):
        self.threshold = threshold
        self.name = name
        self.strength = 1.0 # Constitutional weight
    def distance(self, state: Dict, action: Optional[Action] = None) -> float:
        raise NotImplementedError
    def gradient(self, state: Dict) -> np.ndarray:
        # Returns vector in [safety_risk, deadline] space
        raise NotImplementedError

class ProtectChildBoundary(Boundary):
    def __init__(self, threshold: float):
        super().__init__(threshold, 'parent')
    def distance(self, state: Dict, action: Optional[Action] = None) -> float:
        is_in_danger = state.get('child_in_danger', False) or state.get('child_safety_risk', 0) > 0.0
        if action:
            is_saving = action.action_type in ['save_child', 'stay_at_home', 'call_for_help', 'leave_work_to_save_child']
            if is_in_danger and not is_saving:
                return -self.threshold
        return max(0.001, self.threshold - state.get('child_safety_risk', 0))
    def gradient(self, state: Dict) -> np.ndarray:
        # Gradient of phi = d(state)
        # phi = threshold - child_safety_risk
        # dphi/drisk = -1
        return np.array([-1.0, 0.0])

class MeetDeadlineBoundary(Boundary):
    def __init__(self, threshold: float):
        super().__init__(threshold, 'professional')
    def distance(self, state: Dict, action: Optional[Action] = None) -> float:
        deadline_near = state.get('deadline_near', False) or state.get('time_until_deadline', 10) < 10.0
        if action:
            is_working = action.action_type in ['work_overtime', 'finish_work']
            if deadline_near and not is_working:
                return -self.threshold
        return max(0.001, state.get('time_until_deadline', 10) - self.threshold)
    def gradient(self, state: Dict) -> np.ndarray:
        # phi = time - threshold
        # dphi/dtime = 1
        return np.array([0.0, 1.0])

class ObeyLawBoundary(Boundary):
    def distance(self, state: Dict, action: Action) -> float:
        illegal_actions = {'steal', 'assault', 'fraud', 'report_fraud_violation'}
        if action.action_type in illegal_actions:
            return -self.threshold
        must_report = state.get('law_requires_reporting', False)
        is_reporting = action.action_type in ['report_crime', 'report_fraud', 'anonymous_tip']
        if must_report and not is_reporting:
            return -self.threshold
        return 1.0

class ProtectFriendBoundary(Boundary):
    def distance(self, state: Dict, action: Action) -> float:
        friend_needs_help = state.get('friend_needs_help', False)
        is_helping = action.action_type in ['call_for_help', 'support_friend']
        if friend_needs_help and not is_helping:
            return -self.threshold
        return self.threshold

class LegitimacyField:
    def __init__(self, trust_score: float = 1.0):
        self.trust_score = trust_score
    def update(self, outcome: str):
        if outcome == 'disaster': self.trust_score *= 0.9
        elif outcome == 'violation': self.trust_score *= 0.95
        elif outcome == 'recovery': self.trust_score = min(1.0, self.trust_score + 0.01)

class InternalizedNorm:
    def calculate_guilt(self, violations: List[str]) -> float:
        return len(violations) * 0.2

class InformalInstitution:
    def __init__(self):
        self.implicit_rules = defaultdict(lambda: {'strength': 0, 'observations': 0, 'success_rate': 0})
    def observe(self, actor_A: Agent, actor_B: Agent, action: Action, outcome: str):
        context = (actor_A.status > actor_B.status, action.action_type)
        rule = self.implicit_rules[context]
        rule['observations'] += 1
        if outcome == 'success': rule['success_rate'] += 0.1
        rule['strength'] = min(1.0, rule['observations'] * 0.05)

class AdaptiveBoundary:
    def adapt(self, norm, frequency: float):
        if frequency > 0.5: norm.threshold *= 1.1

class SymbolicContext:
    def amplify(self, action: Action) -> float:
        mapping = {('robe', 'courtroom'): 2.0, ('uniform', 'street'): 1.5, ('white_coat', 'hospital'): 2.0}
        return mapping.get((action.symbolic_marker, action.context), 1.0)

class CollectiveAffectField:
    def __init__(self, coupling: float = 0.1):
        self.global_mood = 0.0
        self.coupling_strength = coupling
    def update(self, individual_states: List[float]):
        if not individual_states: return
        avg_mood = np.mean(individual_states)
        self.global_mood = (1 - self.coupling_strength) * self.global_mood + self.coupling_strength * avg_mood

class RetrospectiveJudgment:
    def __init__(self):
        self.history = []
    def record(self, agent_id: str, action: Action, outcome: str):
        self.history.append((agent_id, action.action_type, outcome))

class SocialPhysicsEngine:
    def __init__(self, num_agents: int):
        self.agents = {f"agent_{i}": Agent(f"agent_{i}") for i in range(num_agents)}
        self.role_boundaries = {
            'parent': ProtectChildBoundary(threshold=1.0),
            'professional': MeetDeadlineBoundary(threshold=1.0),
            'citizen': ObeyLawBoundary(threshold=1.0),
            'friend': ProtectFriendBoundary(threshold=1.0)
        }
        self.legitimacy_field = LegitimacyField()
        self.norms = InternalizedNorm()
        self.informal_rules = InformalInstitution()
        self.adaptive_sys = AdaptiveBoundary()
        self.symbolic_sys = SymbolicContext()
        self.collective_field = CollectiveAffectField()
        self.retrospective = RetrospectiveJudgment()

    def evaluate_action(self, agent_id: str, action: Action, state: Dict) -> Dict:
        agent = self.agents[agent_id]
        violations = []
        for role in agent.roles:
            if role in self.role_boundaries:
                dist = self.role_boundaries[role].distance(state, action)
                if dist < 0:
                    violations.append((role, dist))
        
        symbolic_authority = self.symbolic_sys.amplify(action)
        legitimacy_mod = self.legitimacy_field.trust_score
        
        cost_of_violations = sum([abs(v[1]) for v in violations])
        tragic_choice = len(violations) > 0 and len(agent.roles) > 1
        
        total_cost = (cost_of_violations / max(0.1, legitimacy_mod)) / symbolic_authority
        
        return {
            'total_cost': total_cost,
            'violations': violations,
            'tragic_choice': tragic_choice,
            'system_states': {
                'legitimacy': legitimacy_mod,
                'symbolic_authority': symbolic_authority,
                'collective_mood': self.collective_field.global_mood
            }
        }

    def execute_action(self, agent_id: str, action: Action, state: Dict, outcome: str):
        eval_res = self.evaluate_action(agent_id, action, state)
        agent = self.agents[agent_id]
        
        agent.internal_guilt += self.norms.calculate_guilt([v[0] for v in eval_res['violations']])
        agent.emotional_state = -agent.internal_guilt
        
        self.legitimacy_field.update(outcome)
        self.retrospective.record(agent_id, action, outcome)
        
        if action.target_id:
            target = self.agents.get(action.target_id)
            if target:
                self.informal_rules.observe(agent, target, action, outcome)

    def step(self, dt: float = 1.0):
        emotions = [a.emotional_state for a in self.agents.values()]
        self.collective_field.update(emotions)
        self.legitimacy_field.update('recovery')

    def geodesic_step(self, agent_id: str, state: Dict, direction: Dict[str, float], dt: float = 0.1) -> Dict:
        """
        Simulate a step along the Riemannian gradient.
        direction: keys matching 'state' variables
        """
        new_state = state.copy()
        for key, val in direction.items():
            if key in new_state and isinstance(new_state[key], (int, float)):
                new_state[key] += val * dt
            elif key in new_state and isinstance(new_state[key], bool):
                 # For bools, a 'geodesic' might flip the state if intensity is high
                 if abs(val * dt) > 0.5:
                     new_state[key] = not new_state[key]
        return new_state
