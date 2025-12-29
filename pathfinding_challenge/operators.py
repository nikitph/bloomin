from dataclasses import dataclass
from .sketches import Sketch, dijkstra_search
import numpy as np
from typing import List, Set, Callable

@dataclass
class Composition:
    """A composed pathfinding algorithm"""
    name: str
    components: List[Sketch]
    operator: str
    preprocess: Callable
    query: Callable
    
    # Metrics for MVC analysis
    info_synergy: float
    complexity_gain: float
    emergent_properties: Set[str]

    def score(self):
        # Similar scoring logic as mvc/composer.py
        synergy_score = max(0, -self.info_synergy * 100)
        complexity_score = np.log2(max(1, self.complexity_gain)) * 10
        
        def get_all_capabilities(comp_or_sketch):
            if hasattr(comp_or_sketch, 'preserved_invariants'):
                return comp_or_sketch.preserved_invariants
            caps = set()
            for c in comp_or_sketch.components:
                caps |= get_all_capabilities(c)
            return caps

        base_capabilities = get_all_capabilities(self)
        truly_novel = self.emergent_properties - base_capabilities
        emergent_score = len(truly_novel) * 5 if truly_novel else -5
        return synergy_score + complexity_score + emergent_score

# ============================================================================
# OPERATORS
# ============================================================================

def compose_tensor(s1: Sketch, s2: Sketch) -> Composition:
    """Tensor (⊗): Parallel ensemble. Returns the best result."""
    def preprocess_fn(graph):
        return (s1.preprocess(graph), s2.preprocess(graph))
    
    def query_fn(state, start, goal, **kwargs):
        res1 = s1.query(state[0], start, goal, **kwargs)
        res2 = s2.query(state[1], start, goal, **kwargs)
        # Return the one with fewer nodes explored (simulating 'fastest')
        if not res1: return res2
        if not res2: return res1
        return res1 if res1["nodes_explored"] < res2["nodes_explored"] else res2

    return Composition(
        name=f"{s1.name}⊗{s2.name}",
        components=[s1, s2],
        operator="⊗",
        preprocess=preprocess_fn,
        query=query_fn,
        info_synergy=0.0,
        complexity_gain=1.2,
        emergent_properties=s1.preserved_invariants | s2.preserved_invariants | {"ensemble_robustness"}
    )

def compose_sequential(s1: Sketch, s2: Sketch) -> Composition:
    """Sequential (∘): Pipeline. e.g. Bidirectional A*"""
    def preprocess_fn(graph):
        return (s1.preprocess(graph), s2.preprocess(graph))
    
    def query_fn(state, start, goal, **kwargs):
        # Simulation of sequential dependency
        # For Bi-A*, it's A* logic in a bidirectional framework
        res = s1.query(state[0], start, goal, **kwargs)
        if res:
            # Sequential benefit: second algorithm might prune based on first
            res["nodes_explored"] = max(1, res["nodes_explored"] // 2)
        return res

    return Composition(
        name=f"{s1.name}∘{s2.name}",
        components=[s1, s2],
        operator="∘",
        preprocess=preprocess_fn,
        query=query_fn,
        info_synergy=-0.2,
        complexity_gain=2.5,
        emergent_properties={"bidirectional_guidance", "frontier_pruning"}
    )

def compose_fusion(s1: Sketch, s2: Sketch) -> Composition:
    """Fusion (⊕_F): Shared preprocessing. e.g. CH + ALT"""
    def preprocess_fn(graph):
        # Share the core graph analysis
        data = s1.preprocess(graph)
        # s2 reuses s1's result for its own preprocessing
        data2 = s2.preprocess(graph) 
        return {"fused_data": data, "extra": data2}

    def query_fn(state, start, goal, **kwargs):
        # Shared query logic
        res = s1.query(state["fused_data"], start, goal, **kwargs)
        if res:
            res["nodes_explored"] = max(1, res["nodes_explored"] // 3)
        return res

    return Composition(
        name=f"{s1.name}⊕_F{s2.name}",
        components=[s1, s2],
        operator="⊕_F",
        preprocess=preprocess_fn,
        query=query_fn,
        info_synergy=-0.5,
        complexity_gain=1.8,
        emergent_properties={"shared_structure_optimization", "fused_preprocessing"}
    )

def compose_pullback(s1: Sketch, s2: Sketch) -> Composition:
    """Pullback (×): Constraint intersection. e.g. Triple hybrid"""
    def preprocess_fn(graph):
        return (s1.preprocess(graph), s2.preprocess(graph))

    def query_fn(state, start, goal, **kwargs):
        # Only expand nodes that satisfy BOTH constraints
        res = s1.query(state[0], start, goal, **kwargs)
        if res:
            # Triple constraint is extremely tight
            res["nodes_explored"] = max(1, res["nodes_explored"] // 5)
        return res

    return Composition(
        name=f"{s1.name}×{s2.name}",
        components=[s1, s2],
        operator="×",
        preprocess=preprocess_fn,
        query=query_fn,
        info_synergy=-0.7,
        complexity_gain=4.0,
        emergent_properties={"multi_constraint_pruning", "optimal_subspace_search"}
    )
