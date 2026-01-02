import networkx as nx
from rads.discretization import HyperedgeReasoner

class SymbolicHypergraph(HyperedgeReasoner):
    """
    Reasoning engine over extracted hypergraphs.
    Supports pathfinding, conflict detection, and knowledge aggregation.
    """
    def __init__(self, n=0):
        super().__init__(n)
        # We can map ID -> Name if provided, but raw logic uses IDs or string labels.
        # Let's support string labels for ease of use in Exp 3.1
        self.edges = [] # List of (source, target) tuples for simple traversal
        # Hyperedges are 1-to-many: source -> {targets}
        
    def add_hyperedge(self, source, targets, weight=1.0):
        # Store as base class struct
        super().add_hyperedge(source, targets, weight)
        # Also build efficient graph for traversal
        # Assuming Targets are reachable from Source
        ensure_list = list(targets) if not isinstance(targets, (list, set)) else targets
        for t in ensure_list:
            self.edges.append((source, t))
            
    def incorporate_graph(self, other_graph):
        """
        Merge knowledge from another HyperedgeReasoner or SymbolicHypergraph.
        """
        for he in other_graph.hyperedges:
            self.add_hyperedge(he['source'], he['targets'], he.get('weight', 1.0))

    def query(self, start_node, target_node):
        """
        Reason about relationship between start and target.
        Returns dict with reachability, paths, and conflict info.
        """
        # Build NX graph for pathfinding
        G = nx.DiGraph()
        G.add_edges_from(self.edges)
        
        result = {
            'reachable': False,
            'exception': False,
            'paths': [],
            'conflicts': []
        }
        
        if not G.has_node(start_node):
            return result
            
        # 1. Check positive reachability
        try:
            if G.has_node(target_node) and nx.has_path(G, start_node, target_node):
                result['reachable'] = True
                # Get one path efficiently or all?
                # all_simple_paths can be slow for large graphs, but for these tests it's fine.
                # Remove cutoff to allow long chain generalization (Exp 3.2 L=20)
                try:
                    # Just get first path for efficiency if we don't need all
                    path_gen = nx.all_simple_paths(G, start_node, target_node)
                    result['paths'] = [next(path_gen)] # Just store one proof
                    # result['paths'] = list(nx.all_simple_paths(G, start_node, target_node))
                except StopIteration:
                    pass # Should be covered by has_path but just in case
        except nx.NodeNotFound:
            pass
            
        # 2. Check for negation/conflict
        # Assumption: Negation is represented as "not_Target" or "!Target" or "¬Target"
        # We check if we can reach ANY node that represents negation of target
        # Heuristic: Check for "not_" + target or "!" + target
        # Valid negation prefixes
        neg_prefixes = ["not_", "!", "¬"]
        
        neg_target = None
        for p in neg_prefixes:
            candidate = f"{p}{target_node}"
            if G.has_node(candidate):
                neg_target = candidate
                break
                
            # Also check if target itself is a negation, e.g. target="!Fly", negated="Fly"
            if target_node.startswith(p):
                clean = target_node[len(p):]
                if G.has_node(clean):
                    neg_target = clean
                    break
                    
        if neg_target:
            if nx.has_path(G, start_node, neg_target):
                result['exception'] = True
                result['conflicts'].append(neg_target)
                # Store conflict paths too
                try:
                    neg_gen = nx.all_simple_paths(G, start_node, neg_target)
                    result['paths'].append(next(neg_gen))
                except StopIteration:
                    pass
                    
        return result
    
    def test_accuracy(self, chains):
        """
        Test generalization accuracy on a list of chains.
        Each chain is A->B->...->Z.
        Query: (A, Z) should be reachable.
        """
        correct = 0
        total = 0
        
        # Build graph once from accumulated edges (self is the KB)
        G = nx.DiGraph()
        G.add_edges_from(self.edges)
        
        for chain in chains:
            start = chain[0]
            end = chain[-1]
            total += 1
            
            # Check reachability
            if G.has_node(start) and G.has_node(end):
                if nx.has_path(G, start, end):
                    correct += 1
                    
        return correct / total if total > 0 else 0.0

    def query_nlp(self, text_query):
        """
        Simple NLP query parser.
        Supports: "Is <X> in <Y>?"
        """
        import re
        
        # Regex for "Is X in Y?"
        match = re.search(r"Is (.+) in (.+)\?", text_query, re.IGNORECASE)
        if match:
            subject = match.group(1).lower().strip()
            container = match.group(2).lower().strip()
            
            # Map names to nodes?
            # Assuming nodes are stored as strings or can match case-insensitive
            # My edges are stored as whatever was added (strings).
            
            # Find best match nodes
            all_nodes = set()
            for u, v in self.edges:
                all_nodes.add(u)
                all_nodes.add(v)
                
            print(f"DEBUG: NLP Searching for '{subject}' and '{container}' in nodes: {all_nodes}")
            
            def find_node(name):
                # Clean query name (remove 'the', 'a')
                base_name = name.replace("the ", "").replace("a ", "").strip()
                
                # Exact match
                if base_name in all_nodes: return base_name
                
                # Case insensitive
                for n in all_nodes:
                    if str(n).lower() == base_name: return n
                    
                # Substring (bidirectional)
                for n in all_nodes:
                    n_str = str(n).lower()
                    if base_name in n_str or n_str in base_name: return n
                return None
                
            s_node = find_node(subject)
            o_node = find_node(container)
            
            if s_node and o_node:
                res = self.query(s_node, o_node)
                if res['reachable']:
                    return {'answer': 'YES', 'confidence': 1.0, 'path': res['paths']}
                else:
                    return {'answer': 'NO', 'confidence': 0.8}
            else:
                return {'answer': 'UNKNOWN', 'error': 'Entities not found'}
        
        return {'answer': 'UNKNOWN', 'error': 'Query not understood'}
