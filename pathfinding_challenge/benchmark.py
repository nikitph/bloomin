import time
import random
import numpy as np
from typing import Dict, List, Tuple
from pathfinding_challenge.discoveries import run_discovery

import math

def generate_mock_road_network(nodes_count: int = 1000):
    """Generate a grid-like graph with (x, y) coordinates"""
    graph = {i: {} for i in range(nodes_count)}
    coords = {}
    side = int(math.sqrt(nodes_count))
    
    # 1. Assign coordinates
    for i in range(side):
        for j in range(side):
            u = i * side + j
            if u < nodes_count:
                coords[u] = (float(i), float(j))
    
    # 2. Connect neighbors in grid
    for u in range(nodes_count):
        if u not in coords: continue
        xi, yi = coords[u]
        # Check 4 neighbors
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            # Find node at coord (xi+dx, yi+dy)
            nx, ny = xi + dx, yi + dy
            if 0 <= nx < side and 0 <= ny < side:
                v = int(nx * side + ny)
                if v < nodes_count and u != v:
                    # Weight is Euclidean distance (always 1 here for grid)
                    # Add some noise to make it realistic
                    w = 1.0 + random.random() * 0.5
                    graph[u][v] = w
                    graph[v][u] = w # Ensure symmetry
    
    # 3. Add highways
    for _ in range(nodes_count // 10):
        u = random.randint(0, nodes_count - 1)
        v = random.randint(0, nodes_count - 1)
        if u != v and u in coords and v in coords:
            dist = math.sqrt((coords[u][0] - coords[v][0])**2 + (coords[u][1] - coords[v][1])**2)
            # Highways are faster (half weight)
            graph[u][v] = dist * 0.5
            graph[v][u] = dist * 0.5 # Ensure symmetry
            
    return graph, coords

def benchmark_algorithm(algo, graph, coords, queries):
    """Benchmark with full metrics and optimality check"""
    print(f"Benchmarking {algo.name}...")
    
    start_pre = time.perf_counter()
    state = algo.preprocess(graph)
    pre_time = time.perf_counter() - start_pre
    
    latencies = []
    nodes_explored_list = []
    costs = []
    path_lengths = []
    
    def euclidean_heuristic(a, b):
        if a not in coords or b not in coords: return 0
        # Use 0.5 multiplier to ensure admissibility (highways are dist * 0.5)
        return 0.5 * math.sqrt((coords[a][0] - coords[b][0])**2 + (coords[a][1] - coords[b][1])**2)
        
    for start, goal in queries:
        start_q = time.perf_counter()
        res = algo.query(state, start, goal, heuristic=euclidean_heuristic)
        end_q = time.perf_counter()
        
        if res:
            latencies.append((end_q - start_q) * 1000)
            nodes_explored_list.append(res["nodes_explored"])
            costs.append(res["cost"])
            path_lengths.append(len(res["path"]))
            
    avg_latency = np.mean(latencies) if latencies else 0
    avg_nodes = np.mean(nodes_explored_list) if nodes_explored_list else 0
    avg_cost = np.mean(costs) if costs else 0
    avg_path_len = np.mean(path_lengths) if path_lengths else 0
    
    # Simple memory estim (mock)
    space_mb = 1.0
    if "CH" in algo.name: space_mb = 12.0
    elif "ALT" in algo.name: space_mb = 8.0
    elif "Heat" in algo.name or "Varadhan" in algo.name: space_mb = 20.0
    elif "⊗" in algo.name or "⊕" in algo.name or "×" in algo.name: space_mb = 15.0

    return {
        "name": algo.name,
        "pre_time_s": pre_time,
        "avg_latency_ms": avg_latency,
        "avg_nodes": avg_nodes,
        "avg_path_len": avg_path_len,
        "avg_cost": avg_cost,
        "space_mb": space_mb
    }

def main():
    import random
    random.seed(42)
    np.random.seed(42)

    print("Generating geometric graph...")
    graph, coords = generate_mock_road_network(10000) # Smaller for heat diffusion speed
    
    queries = []
    nodes = list(graph.keys())
    for _ in range(20): # Fewer queries for speed
        queries.append((random.choice(nodes), random.choice(nodes)))
        
    compositions, bases = run_discovery()
    all_algos = bases + compositions
    
    results = []
    for algo in all_algos:
        res = benchmark_algorithm(algo, graph, coords, queries)
        results.append(res)
        
    # Verify optimality (against Dijkstra)
    dijkstra_res = next(r for r in results if r['name'] == 'Dijkstra')
    print("\n" + "="*110)
    print(f"{'Algorithm':<30} | {'Query(ms)':<10} | {'Nodes':<10} | {'PathLen':<10} | {'Cost':<10} | {'Pre(s)':<10} | {'MB':<5}")
    print("-" * 110)
    for res in sorted(results, key=lambda x: x['avg_latency_ms']):
        opt_mark = "✓" if abs(res['avg_cost'] - dijkstra_res['avg_cost']) < 1e-6 else "✗"
        print(f"{res['name']:<30} | {res['avg_latency_ms']:<10.2f} | {int(res['avg_nodes']):<10} | {int(res['avg_path_len']):<10} | {res['avg_cost']:<10.1f} {opt_mark} | {res['pre_time_s']:<10.2f} | {res['space_mb']:<5.1f}")
    print("="*110)
    print("✓ = Optimal relative to Dijkstra")

if __name__ == "__main__":
    main()
