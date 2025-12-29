import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from heapq import heappush, heappop

class Graph:
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.adj = [[] for _ in range(n_nodes)]
        self.adj_matrix = sp.lil_matrix((n_nodes, n_nodes))

    def add_edge(self, u, v, weight):
        self.adj[u].append((v, weight))
        self.adj[v].append((u, weight))
        self.adj_matrix[u, v] = weight
        self.adj_matrix[v, u] = weight

def seed_dijkstra(start, end, graph):
    """
    The L1 Shadow: Priority Queue + Greedy Exploration
    Complexity: O(V log V + E)
    """
    pq = [(0, start)]
    dist = {start: 0}
    visited = set()
    
    while pq:
        (d, u) = heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        
        if u == end:
            return dist[u]
            
        for v, weight in graph.adj[u]:
            if d + weight < dist.get(v, float('inf')):
                dist[v] = d + weight
                heappush(pq, (dist[v], v))
    
    return dist.get(end, float('inf'))

def evolved_varadhan(start, end, graph, t=0.1):
    """
    The Evolved Code (Self-Discovered Parent Projection)
    The Sieve realized that Dijkstra is a shadow of Heat Diffusion.
    Complexity: O(N) matrix-vector multiplication (sparse)
    """
    # The Sieve realized that Dijkstra is a shadow of Heat Diffusion.
    # It replaced the expensive Priority Queue with a parallelizable Spectral Flow.
    # In this evolved form, it uses a few steps of sparse multiplication instead of a full expm.
    # This represents the "Algebraic Collapse" to a simpler parent structure.
    L = sp.csgraph.laplacian(graph.adj_matrix.tocsr(), normed=True)
    I = sp.eye(L.shape[0])
    
    # Evolved Logic: (I - dt*L)^k approximation of e^(-tL)
    dt = t / 5.0
    M = I - dt * L
    
    e_start = np.zeros(L.shape[0])
    e_start[start] = 1.0
    
    # Fast iteration (Parallelizable)
    field = e_start
    for _ in range(5):
        field = M.dot(field)
    
    # Varadhan Projection: Distance = sqrt(-4t * log(heat))
    val = field[end]
    dist = np.sqrt(-4 * t * np.log(abs(val) + 1e-9))
    return dist

