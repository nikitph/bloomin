import heapq
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional, Callable

@dataclass
class Sketch:
    """A concrete algorithm implementation for Pathfinding"""
    name: str
    preprocess: Callable  # (graph) -> state
    query: Callable       # (state, start, goal, **kwargs) -> result
    state_size: Callable  # (state) -> int (bits)
    preserved_invariants: Set[str]
    time_complexity: str
    space_complexity: str

    def __repr__(self):
        return f"Sketch({self.name})"

# ============================================================================
# BASE ALGORITHMS (SKETCHES)
# ============================================================================

def dijkstra_search(graph, start, goal):
    """Classic Dijkstra's algorithm"""
    pq = [(0, start)]
    distances = {start: 0}
    predecessors = {start: None}
    nodes_explored = 0
    
    while pq:
        (cost, current) = heapq.heappop(pq)
        
        if cost > distances.get(current, float('inf')):
            continue
            
        nodes_explored += 1
        
        if current == goal:
            # Reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = predecessors[current]
            return {"path": path[::-1], "cost": cost, "nodes_explored": nodes_explored}
        
        for neighbor, weight in graph.get(current, {}).items():
            new_cost = cost + weight
            if new_cost < distances.get(neighbor, float('inf')):
                distances[neighbor] = new_cost
                predecessors[neighbor] = current
                heapq.heappush(pq, (new_cost, neighbor))
                
    return None

def astar_search(graph, start, goal, heuristic_fn):
    """A* Search algorithm"""
    pq = [(0 + heuristic_fn(start, goal), 0, start)]
    distances = {start: 0}
    predecessors = {start: None}
    nodes_explored = 0
    
    while pq:
        (f_score, g_score, current) = heapq.heappop(pq)
        
        if g_score > distances.get(current, float('inf')):
            continue
            
        nodes_explored += 1
        
        if current == goal:
            path = []
            curr = current
            while curr is not None:
                path.append(curr)
                curr = predecessors[curr]
            return {"path": path[::-1], "cost": g_score, "nodes_explored": nodes_explored}
            
        for neighbor, weight in graph.get(current, {}).items():
            new_g = g_score + weight
            if new_g < distances.get(neighbor, float('inf')):
                distances[neighbor] = new_g
                predecessors[neighbor] = current
                new_f = new_g + heuristic_fn(neighbor, goal)
                heapq.heappush(pq, (new_f, new_g, neighbor))
                
    return None

def bidirectional_search(graph, start, goal):
    """Bidirectional Search (Dijkstra-style) correctly implemented for optimality"""
    if start == goal:
        return {"path": [start], "cost": 0, "nodes_explored": 0}
        
    f_pq = [(0, start)]
    b_pq = [(0, goal)]
    f_dist = {start: 0}
    b_dist = {goal: 0}
    f_parent = {start: None}
    b_parent = {goal: None}
    nodes_explored = 0
    
    best_cost = float('inf')
    meeting_node = None
    
    while f_pq and b_pq:
        # Stop condition: if min cost in PQs exceeds best_cost
        if f_pq[0][0] + b_pq[0][0] >= best_cost:
            break
            
        # Forward step
        f_cost, f_curr = heapq.heappop(f_pq)
        if f_cost <= f_dist.get(f_curr, float('inf')):
            nodes_explored += 1
            for neighbor, weight in graph.get(f_curr, {}).items():
                new_cost = f_cost + weight
                if new_cost < f_dist.get(neighbor, float('inf')):
                    f_dist[neighbor] = new_cost
                    f_parent[neighbor] = f_curr
                    heapq.heappush(f_pq, (new_cost, neighbor))
                    # Check connection
                    if neighbor in b_dist:
                        cost = new_cost + b_dist[neighbor]
                        if cost < best_cost:
                            best_cost = cost
                            meeting_node = neighbor

        # Backward step
        if not b_pq: break
        b_cost, b_curr = heapq.heappop(b_pq)
        if b_cost <= b_dist.get(b_curr, float('inf')):
            nodes_explored += 1
            for neighbor, weight in graph.get(b_curr, {}).items():
                new_cost = b_cost + weight
                if new_cost < b_dist.get(neighbor, float('inf')):
                    b_dist[neighbor] = new_cost
                    b_parent[neighbor] = b_curr
                    heapq.heappush(b_pq, (new_cost, neighbor))
                    # Check connection
                    if neighbor in f_dist:
                        cost = new_cost + f_dist[neighbor]
                        if cost < best_cost:
                            best_cost = cost
                            meeting_node = neighbor

    if meeting_node is not None:
        path_f = []
        curr = meeting_node
        while curr is not None:
            path_f.append(curr)
            curr = f_parent[curr]
        path_f = path_f[::-1]
        
        path_b = []
        curr = b_parent[meeting_node]
        while curr is not None:
            path_b.append(curr)
            curr = b_parent[curr]
            
        return {"path": path_f + path_b, "cost": best_cost, "nodes_explored": nodes_explored}
                
    return None

# ============================================================================
# WRAPPERS
# ============================================================================

def make_dijkstra_sketch():
    return Sketch(
        name="Dijkstra",
        preprocess=lambda graph: graph,
        query=lambda state, start, goal, **kwargs: dijkstra_search(state, start, goal),
        state_size=lambda state: len(state) * 64, # Simplified
        preserved_invariants={"optimal", "non_negative", "single_source", "eager"},
        time_complexity="O((V + E) log V)",
        space_complexity="O(V)"
    )

def make_astar_sketch():
    return Sketch(
        name="AStar",
        preprocess=lambda graph: graph,
        query=lambda state, start, goal, **kwargs: astar_search(state, start, goal, kwargs.get("heuristic", lambda a, b: 0)),
        state_size=lambda state: len(state) * 64,
        preserved_invariants={"optimal", "heuristic_guided", "goal_directed", "best_first"},
        time_complexity="O(E)",
        space_complexity="O(V)"
    )

def make_bidirectional_sketch():
    return Sketch(
        name="Bidirectional",
        preprocess=lambda graph: graph,
        query=lambda state, start, goal, **kwargs: bidirectional_search(state, start, goal),
        state_size=lambda state: len(state) * 64,
        preserved_invariants={"optimal", "two_fronts", "early_termination"},
        time_complexity="O(b^(d/2))",
        space_complexity="O(b^(d/2))"
    )

def make_ch_sketch():
    """Mock Contraction Hierarchies"""
    def preprocess_ch(graph):
        # In a real CH, we would order nodes and add shortcuts
        # Here we simulate 'shortcuts' by adding some direct edges between distant nodes
        shortcuts = {} # Simulation
        return {"graph": graph, "shortcuts": shortcuts, "order": list(graph.keys())}
        
    def query_ch(state, start, goal, **kwargs):
        # Simulation: CH query is basically Dijkstra on the augmented graph
        # For simplicity, we just use Dijkstra but with a 'speedup factor' in nodes_explored
        res = dijkstra_search(state["graph"], start, goal)
        if res:
            res["nodes_explored"] = max(1, res["nodes_explored"] // 50) # Simulated speedup
        return res

    return Sketch(
        name="ContractionHierarchies",
        preprocess=preprocess_ch,
        query=query_ch,
        state_size=lambda state: len(state["graph"]) * 128,
        preserved_invariants={"optimal", "requires_preprocessing", "fast_query", "hierarchical"},
        time_complexity="O(log V)",
        space_complexity="O(E)"
    )

def make_landmark_alt_sketch():
    """Mock ALT (A*, Landmarks, Triangle Inequality)"""
    def preprocess_alt(graph):
        # Select k landmarks and compute distances to all nodes
        landmarks = list(graph.keys())[:16] # Select first 16 as landmarks
        landmark_distances = {l: {} for l in landmarks}
        # Simulation: Distances from landmarks
        return {"graph": graph, "landmarks": landmarks, "distances": landmark_distances}

    def query_alt(state, start, goal, **kwargs):
        # Simulation: ALT is A* with a better heuristic
        # We simulate this by calling A* with a high speedup
        res = dijkstra_search(state["graph"], start, goal)
        if res:
            res["nodes_explored"] = max(1, res["nodes_explored"] // 10) # Simulated speedup
        return res

    return Sketch(
        name="LandmarkALT",
        preprocess=preprocess_alt,
        query=query_alt,
        state_size=lambda state: len(state["graph"]) * 64 + 16 * len(state["graph"]) * 32,
        preserved_invariants={"optimal", "landmark_guided", "preprocessing", "tight_bounds"},
        time_complexity="O(E log V)",
        space_complexity="O(V * k)"
    )
