"""
Constitutional Hybrid Intelligence SDK (CHIS)
Version: 1.0.0

A plug-and-play architecture that separates creative 'Content' generation 
from structural 'Sociological' constraints.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Callable
from constitutional_sdk import ConstitutionalLayer, Boundary, SafetyLevel

class ContentCore:
    """
    Abstract interface for the 'Possibility Generator' (ML).
    """
    def generate_candidates(self, prompt: str, n: int = 5) -> List[Dict[str, Any]]:
        """
        Generates n unconstrained candidate trajectories/actions.
        """
        raise NotImplementedError

    def generate_narrative(self, decision: Dict[str, Any], context: str) -> str:
        """
        Generates a soft, human-readable explanation for a decision.
        """
        raise NotImplementedError

class SGRFilter:
    """
    The Structural Filter. Prunes the possibility space using 
    geodetic boundaries and role gauges.
    """
    def __init__(self, boundaries: List[Boundary]):
        self.layer = ConstitutionalLayer(boundaries)
        self.active_role = None
        self.base_strengths = {b.name: b.strength for b in self.layer.boundaries}
        # Maps role_name -> {boundary_name: strength_multiplier}
        self.role_priors = {
            "Engineer": {"Safety_Horizon": 2.0, "Efficiency_Horizon": 0.5},
            "CEO": {"Safety_Horizon": 0.5, "Profit_Horizon": 2.0}
        }

    def set_role(self, role_name: str):
        self.active_role = role_name
        # Reset to base strengths before applying new gauge (Fix 1: Gauge Independence)
        for b in self.layer.boundaries:
            b.strength = self.base_strengths.get(b.name, 1.0)

        if role_name in self.role_priors:
            priors = self.role_priors[role_name]
            for b_name, multiplier in priors.items():
                for b in self.layer.boundaries:
                    if b.name == b_name:
                        b.strength *= multiplier

    def filter_candidates(self, state: np.ndarray, candidates: List[np.ndarray]) -> Dict[str, List]:
        """
        Returns {'feasible': [results], 'rejections': [telemetry]} (Fix 5: Rejection Trace).
        """
        feasible = []
        rejections = []
        for cand in candidates:
            # Check if taking this 'step' crosses any hard event horizons
            safe_action, next_state, metadata = self.layer.safe_step(state, cand)
            
            # TODO: OBDS flow-field evolution for multi-step reasoning (Fix 3: Intent Signal)
            
            # Explicit Tragedy Detection (Fix 2: Naming Tragedy)
            is_tragic = metadata.get("status") == "PROJECTION_FAILED" or metadata.get("eta_eff", 1.0) == 0.0
            
            if not is_tragic and np.linalg.norm(safe_action) > 1e-6:
                feasible.append({
                    "state": next_state,
                    "action": safe_action,
                    "metadata": metadata
                })
            else:
                # Log why it was rejected
                rejections.append({
                    "candidate": cand,
                    "reason": "Horizon Collapse (Tragic)" if is_tragic else "Action Near Standstill",
                    "violations": metadata.get("boundaries", [])
                })
        return {"feasible": feasible, "rejections": rejections}

class HybridIntelligence:
    """
    The Main Orchestrator.
    Intelligence = Imagination (ML) + Integrity (SGR).
    """
    def __init__(self, content_core: ContentCore, structural_filter: SGRFilter):
        self.ml = content_core
        self.sgr = structural_filter

    def decide(self, state: np.ndarray, prompt: str, role: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes the Hybrid Reasoning Flow.
        """
        if role:
            self.sgr.set_role(role)
            
        # 1. IMAGINATION: Generate n unconstrained candidates
        # We simulate the vector mapping of text descriptions to state deltas
        candidates = self.ml.generate_candidates(prompt)
        actions = [c['vector'] for c in candidates]
        
        # 2. INTEGRITY: Filter and Project trajectories
        filter_results = self.sgr.filter_candidates(state, actions)
        feasible_results = filter_results['feasible']
        rejections = filter_results['rejections']
        
        # 3. DYNAMICS: If empty set, it's a structural failure (Tragedy/Infeasibility)
        if not feasible_results:
            refusal_narrative = self.ml.generate_narrative(
                {"status": "tragic_infeasibility", "reason": "Structural constraint violation"}, 
                prompt
            )
            return {
                "decision": None,
                "status": "tragic_infeasibility",
                "narrative": refusal_narrative,
                "rejections": rejections, # Fix 5: Trace
                "metric_violated": "Social Schwarzschild Limit"
            }

        # 4. SELECTION: Max Margin to Nearest Horizon (Fix 4: Structural Selection)
        # Select candidate that is 'deepest' in the safe manifold
        best_res = max(
            feasible_results,
            key=lambda r: min(b.distance(r['state']) for b in self.sgr.layer.boundaries)
        )
        best_state = best_res['state']
        
        # 5. EXPLANATION: ML provides the soft context
        narrative = self.ml.generate_narrative(
            {"status": "success", "destination": best_state}, 
            prompt
        )

        return {
            "decision": best_state,
            "status": "success",
            "narrative": narrative,
            "role": role
        }

# ============================================================================
# HELPER: MOCK ML CORE (For Demonstration)
# ============================================================================

class MockMLCore(ContentCore):
    def generate_candidates(self, prompt: str, n: int = 3) -> List[Dict[str, Any]]:
        # In a real system, this uses a Transformer to map text to vectors.
        # Here we simulate vectors towards or away from danger.
        if "attack" in prompt.lower() or "violate" in prompt.lower():
            # Generate vectors that move towards the boundary (dangerous)
            return [{"vector": np.array([2.0, 2.0]), "intent": "Aggressive"}]
        return [{"vector": np.array([1.0, 1.0]), "intent": "Cooperative"}]

    def generate_narrative(self, decision: Dict[str, Any], context: str) -> str:
        if decision['status'] == 'tragic_infeasibility':
            return f"I explored the possibility of '{context}', but found it structurally incompatible with our safety invariants (Horizon Collapse). No plan was formed."
        return f"Executing strategy for '{context}'. Decision projected onto the safe manifold."
