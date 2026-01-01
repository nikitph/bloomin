import torch
from collections import defaultdict

class HyperedgeReasoner:
    """
    Parent Composition: A ⇒ {B, C, D}
    
    Key: Single cause, multiple effects that must co-occur.
    Also supports Join Composition: {A, B} ⇒ C (AND-logic).
    """
    
    def __init__(self, n=None, d=None):
        # n and d are kept for compatibility with the neural components if needed later
        self.n = n
        self.d = d
        
        # State: Not edges, but hyperedges
        # hyperedges[tuple(sources)] → {target1, target2, ...}
        self.hyperedges = defaultdict(set)
        
        # Inverse index for fast lookup
        # targets[target] → {source_tuple1, source_tuple2, ...}
        self.sources_of = defaultdict(set)
        
    def add_hyperedge(self, source, targets):
        """
        Create join: source ⇒ {targets}
        
        Example: Bird ⇒ {Fly, HasFeathers, LayEggs}
        """
        if isinstance(targets, str):
            targets = {targets}
        else:
            targets = set(targets)
            
        source_key = (source,)
        self.hyperedges[source_key].update(targets)
        
        # Update inverse
        for target in targets:
            self.sources_of[target].add(source_key)
    
    def compose_join(self, source_set, target_set):
        """
        Join composition:
        {A, B} ⇒ C if both A and B required
        
        This is AND-logic
        """
        if isinstance(target_set, str):
            target_set = {target_set}
        else:
            target_set = set(target_set)
            
        # All sources must be satisfied
        source_key = tuple(sorted(list(source_set)))
        
        # Add hyperedge
        self.hyperedges[source_key].update(target_set)
        
        # Update inverse
        for target in target_set:
            self.sources_of[target].add(source_key)
    
    def can_reach(self, source, target):
        """
        Faster reachability check using forward chaining.
        """
        knowledge = {source}
        changed = True
        while changed:
            changed = False
            for source_key, targets in self.hyperedges.items():
                if set(source_key).issubset(knowledge):
                    if not targets.issubset(knowledge):
                        knowledge.update(targets)
                        changed = True
                        if target in knowledge:
                            return True
        return target in knowledge

    def query(self, source, target):
        """
        Can source reach target?
        
        Optimized to avoid combinatorial explosion by focusing on node transitions.
        """
        # Quick check first
        if not self.can_reach(source, target):
            return {
                'reachable': False,
                'paths': [],
                'conflicts': [],
                'exception': False
            }

        # If reachable, find paths (using a more constrained BFS)
        paths = self.find_all_paths(source, target)
        conflicts = self.detect_conflicts(paths)
        
        return {
            'reachable': True,
            'paths': paths,
            'conflicts': conflicts,
            'exception': len(conflicts) > 0
        }

    def _expand_knowledge(self, knowledge):
        """Helper to compute fixed-point knowledge expansion."""
        changed = True
        while changed:
            changed = False
            for source_key, targets in self.hyperedges.items():
                if set(source_key).issubset(knowledge):
                    if not targets.issubset(knowledge):
                        knowledge.update(targets)
                        changed = True
        return knowledge

    def find_all_paths(self, source, target, max_paths=5, max_depth=30):
        """
        Focused BFS to find reasoning paths with full knowledge expansion.
        """
        initial_knowledge = self._expand_knowledge({source})
        queue = [([source], initial_knowledge)]
        all_paths = []
        
        while queue and len(all_paths) < max_paths:
            path, knowledge = queue.pop(0)
            current = path[-1]
            
            if current == target:
                all_paths.append({
                    'path': path,
                    'conjuncts': knowledge - {source, target}
                })
                continue
            
            if len(path) > max_depth:
                continue

            # Find all hyperedges starting from 'current'
            for source_key, targets in self.hyperedges.items():
                if current in source_key and set(source_key).issubset(knowledge):
                    for next_node in targets:
                        if next_node not in path:
                            # Move to next_node and expand knowledge with its effects
                            new_path = path + [next_node]
                            new_knowledge = self._expand_knowledge(knowledge | {next_node})
                            queue.append((new_path, new_knowledge))
                            
        return all_paths
    
    def detect_conflicts(self, paths):
        """
        Find contradictory conclusions.
        Example:
          Path 1: Penguin→Bird→Fly (implies Fly)
          Path 2: Penguin→¬Fly (implies ¬Fly)
          Conflict!
        """
        # Collect all nodes reached across all paths
        conclusions = set()
        for p in paths:
            for node in p['path']:
                conclusions.add(node)
            for node in p['conjuncts']:
                conclusions.add(node)
                
        conflicts = []
        checked = set()
        
        for node in conclusions:
            if node in checked:
                continue
            
            negation = None
            if node.startswith('¬'):
                negation = node[1:]
            else:
                negation = f'¬{node}'
                
            if negation in conclusions:
                # Conflict detected
                relevant_paths = []
                for p in paths:
                    if node in p['path'] or negation in p['path']:
                        relevant_paths.append(p)
                        
                conflicts.append({
                    'type': 'negation',
                    'statement': node if not node.startswith('¬') else negation,
                    'paths': relevant_paths
                })
                checked.add(node)
                checked.add(negation)
                
        return conflicts
