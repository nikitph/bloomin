from collections import deque
import random

class BFSReachability:
    """Standard BFS reachability baseline."""
    def __init__(self, n, edges):
        self.n = n
        self.adj = [[] for _ in range(n)]
        for u, v in edges:
            self.adj[u].append(v)

    def reachable(self, start_node, target_node):
        if start_node == target_node:
            return True
        visited = [False] * self.n
        queue = deque([start_node])
        visited[start_node] = True
        
        while queue:
            u = queue.popleft()
            for v in self.adj[u]:
                if v == target_node:
                    return True
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return False

class LandmarkReachability:
    """Landmark-based reachability baseline."""
    def __init__(self, n, edges, num_landmarks=10):
        self.n = n
        self.num_landmarks = min(num_landmarks, n)
        self.landmarks = random.sample(range(n), self.num_landmarks)
        
        self.adj = [[] for _ in range(n)]
        self.rev_adj = [[] for _ in range(n)]
        for u, v in edges:
            self.adj[u].append(v)
            self.rev_adj[v].append(u)
            
        # Precompute reachability to/from landmarks
        self.can_reach_landmark = [[False] * self.num_landmarks for _ in range(n)]
        self.landmark_can_reach = [[False] * n for _ in range(self.num_landmarks)]
        
        for i, l in enumerate(self.landmarks):
            # To l
            self._bfs(l, self.rev_adj, lambda u, val: self._set_can_reach(u, i, val))
            # From l
            self._bfs(l, self.adj, lambda v, val: self._set_landmark_can_reach(i, v, val))

    def _set_can_reach(self, u, landmark_idx, val):
        self.can_reach_landmark[u][landmark_idx] = val

    def _set_landmark_can_reach(self, landmark_idx, v, val):
        self.landmark_can_reach[landmark_idx][v] = val

    def _bfs(self, start_node, adj, callback):
        visited = [False] * self.n
        queue = deque([start_node])
        visited[start_node] = True
        callback(start_node, True)
        
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    callback(v, True)
                    queue.append(v)

    def reachable(self, u, v):
        """Approximate reachability via landmarks: u -> L -> v."""
        if u == v: return True
        for i in range(self.num_landmarks):
            if self.can_reach_landmark[u][i] and self.landmark_can_reach[i][v]:
                return True
        return False
