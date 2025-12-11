"""
Causal Graph Implementation

Provides a causal graph structure for causal inference with:
- Directed acyclic graph (DAG) representation
- Backdoor criterion checking
- Front-door criterion checking
- d-separation testing
"""

import numpy as np
from typing import List, Set, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class CausalNode:
    """A node in the causal graph."""
    name: str
    is_observed: bool = True
    is_latent: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, CausalNode):
            return self.name == other.name
        return False

    def __repr__(self):
        return f"CausalNode({self.name})"


@dataclass
class CausalEdge:
    """A directed edge in the causal graph."""
    source: str  # Parent node name
    target: str  # Child node name
    is_causal: bool = True  # True for causal edges, False for associational
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        arrow = "→" if self.is_causal else "↔"
        return f"{self.source} {arrow} {self.target}"


class CausalGraph:
    """
    A directed acyclic graph (DAG) for causal inference.

    Supports:
    - Adding nodes and edges
    - Finding parents, ancestors, descendants
    - Backdoor path detection
    - Front-door criterion verification
    - d-separation testing
    """

    def __init__(
        self,
        nodes: Optional[List[str]] = None,
        edges: Optional[List[Tuple[str, str]]] = None
    ):
        """
        Initialize causal graph.

        Args:
            nodes: List of node names
            edges: List of (parent, child) tuples
        """
        self._nodes: Dict[str, CausalNode] = {}
        self._edges: List[CausalEdge] = []
        self._children: Dict[str, Set[str]] = defaultdict(set)
        self._parents: Dict[str, Set[str]] = defaultdict(set)

        # Add nodes
        if nodes:
            for node in nodes:
                self.add_node(node)

        # Add edges
        if edges:
            for source, target in edges:
                self.add_edge(source, target)

    def add_node(
        self,
        name: str,
        is_observed: bool = True,
        is_latent: bool = False,
        **metadata
    ) -> CausalNode:
        """Add a node to the graph."""
        if name not in self._nodes:
            node = CausalNode(
                name=name,
                is_observed=is_observed,
                is_latent=is_latent,
                metadata=metadata
            )
            self._nodes[name] = node
        return self._nodes[name]

    def add_edge(
        self,
        source: str,
        target: str,
        is_causal: bool = True,
        **metadata
    ) -> CausalEdge:
        """Add a directed edge from source to target."""
        # Ensure nodes exist
        if source not in self._nodes:
            self.add_node(source)
        if target not in self._nodes:
            self.add_node(target)

        edge = CausalEdge(
            source=source,
            target=target,
            is_causal=is_causal,
            metadata=metadata
        )
        self._edges.append(edge)
        self._children[source].add(target)
        self._parents[target].add(source)

        return edge

    def nodes(self) -> List[str]:
        """Get all node names."""
        return list(self._nodes.keys())

    def edges(self) -> List[Tuple[str, str]]:
        """Get all edges as (source, target) tuples."""
        return [(e.source, e.target) for e in self._edges]

    def parents(self, node: str) -> Set[str]:
        """Get immediate parents of a node."""
        return self._parents.get(node, set()).copy()

    def children(self, node: str) -> Set[str]:
        """Get immediate children of a node."""
        return self._children.get(node, set()).copy()

    def ancestors(self, node: str) -> Set[str]:
        """Get all ancestors of a node (recursive parents)."""
        ancestors = set()
        to_visit = list(self.parents(node))

        while to_visit:
            current = to_visit.pop()
            if current not in ancestors:
                ancestors.add(current)
                to_visit.extend(self.parents(current))

        return ancestors

    def descendants(self, node: str) -> Set[str]:
        """Get all descendants of a node (recursive children)."""
        descendants = set()
        to_visit = list(self.children(node))

        while to_visit:
            current = to_visit.pop()
            if current not in descendants:
                descendants.add(current)
                to_visit.extend(self.children(current))

        return descendants

    def is_ancestor(self, node: str, potential_ancestor: str) -> bool:
        """Check if potential_ancestor is an ancestor of node."""
        return potential_ancestor in self.ancestors(node)

    def is_descendant(self, node: str, potential_descendant: str) -> bool:
        """Check if potential_descendant is a descendant of node."""
        return potential_descendant in self.descendants(node)

    def backdoor_paths(self, X: str, Y: str) -> List[List[str]]:
        """
        Find all backdoor paths from X to Y.

        A backdoor path is a path that:
        1. Starts with an arrow into X (← )
        2. Ends at Y
        3. Does not go through any descendant of X

        Args:
            X: Treatment variable
            Y: Outcome variable

        Returns:
            List of paths (each path is a list of node names)
        """
        paths = []
        descendants_X = self.descendants(X)

        def find_paths(current: str, target: str, path: List[str], visited: Set[str]):
            if current == target:
                paths.append(path.copy())
                return

            if current in visited:
                return

            visited.add(current)

            # Explore all connections (both directions for undirected search)
            # Parents (going backward)
            for parent in self.parents(current):
                if parent not in visited and parent not in descendants_X:
                    find_paths(parent, target, path + [parent], visited.copy())

            # Children (going forward)
            for child in self.children(current):
                if child not in visited and child != X and child not in descendants_X:
                    find_paths(child, target, path + [child], visited.copy())

        # Start from parents of X (backdoor means entering through the back)
        for parent in self.parents(X):
            if parent not in descendants_X:
                find_paths(parent, Y, [X, parent], {X})

        return paths

    def verify_backdoor(
        self,
        Z: Set[str],
        X: str,
        Y: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if Z satisfies the backdoor criterion for (X, Y).

        The backdoor criterion is satisfied if:
        1. Z does not contain any descendant of X
        2. Z blocks all backdoor paths from X to Y

        Args:
            Z: Set of conditioning variables
            X: Treatment variable
            Y: Outcome variable

        Returns:
            (is_valid, details): Whether criterion is satisfied and diagnostic info
        """
        Z = set(Z)
        details = {
            'criterion': 'backdoor',
            'treatment': X,
            'outcome': Y,
            'adjustment_set': list(Z),
            'blocked_paths': [],
            'unblocked_paths': [],
            'issues': []
        }

        # Check condition 1: Z contains no descendants of X
        descendants_X = self.descendants(X)
        descendant_violations = Z & descendants_X

        if descendant_violations:
            details['issues'].append(
                f"Adjustment set contains descendants of {X}: {descendant_violations}"
            )
            return False, details

        # Check condition 2: Z blocks all backdoor paths
        backdoor_paths = self.backdoor_paths(X, Y)
        details['all_backdoor_paths'] = [list(p) for p in backdoor_paths]

        for path in backdoor_paths:
            if self._is_path_blocked(path, Z):
                details['blocked_paths'].append(path)
            else:
                details['unblocked_paths'].append(path)

        if details['unblocked_paths']:
            details['issues'].append(
                f"Unblocked backdoor paths exist: {len(details['unblocked_paths'])}"
            )
            return False, details

        return True, details

    def verify_frontdoor(
        self,
        M: str,
        X: str,
        Y: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify if M satisfies the front-door criterion for (X, Y).

        The front-door criterion requires:
        1. M intercepts all directed paths from X to Y
        2. There are no unblocked backdoor paths from X to M
        3. All backdoor paths from M to Y are blocked by X

        Args:
            M: Mediator variable
            X: Treatment variable
            Y: Outcome variable

        Returns:
            (is_valid, details): Whether criterion is satisfied and diagnostic info
        """
        details = {
            'criterion': 'frontdoor',
            'treatment': X,
            'mediator': M,
            'outcome': Y,
            'issues': []
        }

        # Condition 1: M intercepts all directed paths from X to Y
        directed_paths = self._find_directed_paths(X, Y)
        paths_through_M = [p for p in directed_paths if M in p]
        paths_not_through_M = [p for p in directed_paths if M not in p]

        details['directed_paths'] = directed_paths
        details['paths_through_M'] = paths_through_M
        details['paths_not_through_M'] = paths_not_through_M

        if paths_not_through_M:
            details['issues'].append(
                f"Direct paths from {X} to {Y} not through {M}: {paths_not_through_M}"
            )
            return False, details

        # Condition 2: No unblocked backdoor paths from X to M
        backdoor_X_M = self.backdoor_paths(X, M)
        unblocked_X_M = [p for p in backdoor_X_M if not self._is_path_blocked(p, set())]
        details['backdoor_X_M'] = backdoor_X_M
        details['unblocked_backdoor_X_M'] = unblocked_X_M

        if unblocked_X_M:
            details['issues'].append(
                f"Unblocked backdoor paths from {X} to {M}: {unblocked_X_M}"
            )
            return False, details

        # Condition 3: All backdoor paths from M to Y are blocked by X
        backdoor_M_Y = self.backdoor_paths(M, Y)
        unblocked_M_Y = [p for p in backdoor_M_Y if not self._is_path_blocked(p, {X})]
        details['backdoor_M_Y'] = backdoor_M_Y
        details['unblocked_backdoor_M_Y'] = unblocked_M_Y

        if unblocked_M_Y:
            details['issues'].append(
                f"Backdoor paths from {M} to {Y} not blocked by {X}: {unblocked_M_Y}"
            )
            return False, details

        return True, details

    def _find_directed_paths(self, start: str, end: str) -> List[List[str]]:
        """Find all directed paths from start to end."""
        paths = []

        def dfs(current: str, path: List[str], visited: Set[str]):
            if current == end:
                paths.append(path.copy())
                return

            if current in visited:
                return

            visited.add(current)

            for child in self.children(current):
                dfs(child, path + [child], visited.copy())

        dfs(start, [start], set())
        return paths

    def _is_path_blocked(self, path: List[str], Z: Set[str]) -> bool:
        """
        Check if a path is blocked by conditioning set Z.

        Uses d-separation rules:
        - Chain (A → B → C): Blocked if B ∈ Z
        - Fork (A ← B → C): Blocked if B ∈ Z
        - Collider (A → B ← C): Blocked if B ∉ Z and no descendant of B is in Z
        """
        if len(path) < 3:
            return False

        for i in range(len(path) - 2):
            A, B, C = path[i], path[i + 1], path[i + 2]

            # Determine the type of triple
            A_to_B = B in self.children(A)
            B_to_C = C in self.children(B)
            A_from_B = A in self.children(B)
            B_from_C = B in self.children(C)

            # Collider: A → B ← C
            is_collider = A_to_B and B_from_C

            if is_collider:
                # Collider is blocked unless B or descendant of B is in Z
                B_or_desc_in_Z = B in Z or bool(self.descendants(B) & Z)
                if not B_or_desc_in_Z:
                    return True  # Path is blocked at this collider
            else:
                # Chain or Fork: blocked if B ∈ Z
                if B in Z:
                    return True

        return False

    def d_separated(
        self,
        X: Set[str],
        Y: Set[str],
        Z: Set[str]
    ) -> bool:
        """
        Test if X and Y are d-separated given Z.

        Args:
            X: First set of variables
            Y: Second set of variables
            Z: Conditioning set

        Returns:
            True if X and Y are d-separated given Z
        """
        # For each pair (x, y), check if all paths are blocked
        for x in X:
            for y in Y:
                paths = self._find_all_paths(x, y)
                for path in paths:
                    if not self._is_path_blocked(path, Z):
                        return False
        return True

    def _find_all_paths(self, start: str, end: str, max_length: int = 20) -> List[List[str]]:
        """Find all paths (directed or undirected) between two nodes."""
        paths = []

        def dfs(current: str, path: List[str], visited: Set[str]):
            if len(path) > max_length:
                return

            if current == end:
                paths.append(path.copy())
                return

            if current in visited:
                return

            visited.add(current)

            # All neighbors (parents and children)
            neighbors = self.parents(current) | self.children(current)
            for neighbor in neighbors:
                dfs(neighbor, path + [neighbor], visited.copy())

        dfs(start, [start], set())
        return paths

    def find_valid_adjustment_set(
        self,
        X: str,
        Y: str,
        available_vars: Optional[Set[str]] = None
    ) -> Optional[Set[str]]:
        """
        Find a valid adjustment set for the backdoor criterion.

        Args:
            X: Treatment variable
            Y: Outcome variable
            available_vars: Variables available for conditioning

        Returns:
            A valid adjustment set, or None if none exists
        """
        if available_vars is None:
            available_vars = set(self.nodes()) - {X, Y}

        # Remove descendants of X
        descendants_X = self.descendants(X)
        valid_candidates = available_vars - descendants_X

        # Try to find minimal adjustment set
        # Start with empty set and add variables as needed
        backdoor = self.backdoor_paths(X, Y)

        if not backdoor:
            return set()  # No backdoor paths, empty set works

        # Greedy approach: add variables that block most paths
        Z = set()
        unblocked = backdoor.copy()

        while unblocked:
            best_var = None
            best_blocked = 0

            for var in valid_candidates - Z:
                test_Z = Z | {var}
                blocked_count = sum(1 for p in unblocked if self._is_path_blocked(p, test_Z))

                if blocked_count > best_blocked:
                    best_blocked = blocked_count
                    best_var = var

            if best_var is None:
                return None  # Cannot block all paths

            Z.add(best_var)
            unblocked = [p for p in unblocked if not self._is_path_blocked(p, Z)]

        return Z

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'nodes': [
                {
                    'name': n.name,
                    'is_observed': n.is_observed,
                    'is_latent': n.is_latent,
                    'metadata': n.metadata
                }
                for n in self._nodes.values()
            ],
            'edges': [
                {
                    'source': e.source,
                    'target': e.target,
                    'is_causal': e.is_causal,
                    'metadata': e.metadata
                }
                for e in self._edges
            ]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CausalGraph':
        """Create from dictionary."""
        graph = cls()

        for node_data in data.get('nodes', []):
            graph.add_node(
                node_data['name'],
                is_observed=node_data.get('is_observed', True),
                is_latent=node_data.get('is_latent', False),
                **node_data.get('metadata', {})
            )

        for edge_data in data.get('edges', []):
            graph.add_edge(
                edge_data['source'],
                edge_data['target'],
                is_causal=edge_data.get('is_causal', True),
                **edge_data.get('metadata', {})
            )

        return graph

    def __repr__(self) -> str:
        return f"CausalGraph(nodes={len(self._nodes)}, edges={len(self._edges)})"


# Convenience functions for creating common graph structures

def create_confounded_graph(X: str, Y: str, U: str = "U") -> CausalGraph:
    """
    Create a graph with confounding: X ← U → Y, X → Y

    Args:
        X: Treatment variable
        Y: Outcome variable
        U: Latent confounder

    Returns:
        CausalGraph with confounding structure
    """
    graph = CausalGraph()
    graph.add_node(X)
    graph.add_node(Y)
    graph.add_node(U, is_observed=False, is_latent=True)

    graph.add_edge(U, X)
    graph.add_edge(U, Y)
    graph.add_edge(X, Y)

    return graph


def create_mediated_graph(X: str, M: str, Y: str) -> CausalGraph:
    """
    Create a graph with mediation: X → M → Y

    Args:
        X: Treatment variable
        M: Mediator variable
        Y: Outcome variable

    Returns:
        CausalGraph with mediation structure
    """
    graph = CausalGraph()
    graph.add_node(X)
    graph.add_node(M)
    graph.add_node(Y)

    graph.add_edge(X, M)
    graph.add_edge(M, Y)

    return graph


def create_frontdoor_graph(X: str, M: str, Y: str, U: str = "U") -> CausalGraph:
    """
    Create a front-door graph: X → M → Y, U → X, U → Y

    Args:
        X: Treatment variable
        M: Mediator variable
        Y: Outcome variable
        U: Latent confounder

    Returns:
        CausalGraph with front-door structure
    """
    graph = CausalGraph()
    graph.add_node(X)
    graph.add_node(M)
    graph.add_node(Y)
    graph.add_node(U, is_observed=False, is_latent=True)

    graph.add_edge(X, M)
    graph.add_edge(M, Y)
    graph.add_edge(U, X)
    graph.add_edge(U, Y)

    return graph
