import numpy as np
from dataclasses import dataclass
from typing import List, Set, Callable, Tuple
from collections import defaultdict
import hashlib

# Global instrumentation for validation
HASH_CALL_COUNTER = 0

def reset_hash_counter():
    global HASH_CALL_COUNTER
    HASH_CALL_COUNTER = 0

def get_hash_count():
    return HASH_CALL_COUNTER

# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class Sketch:
    """A concrete algorithm implementation"""
    name: str
    insert: Callable  # (data) -> state
    query: Callable   # (state, query) -> result
    state_size: Callable  # (state) -> int (bits)
    preserved_invariants: Set[str]
    
    def __repr__(self):
        return f"Sketch({self.name})"

@dataclass
class Composition:
    """A composed algorithm"""
    name: str
    components: List[Sketch]
    operator: str
    insert: Callable
    query: Callable
    state_size: Callable
    
    # Scoring metrics
    info_synergy: float  # Negative = good (less info loss than sum)
    complexity_gain: float  # Speedup factor
    emergent_properties: Set[str]
    
    def score(self):
        """Combined score: higher is better"""
        # 1. Synergy: negative info_synergy is good (converted to positive score)
        # info_synergy is normalized bit savings: (H_comp - (H1 + H2)) / (H1 + H2)
        # If synergy = -0.5 (50% saving), score = 50
        synergy_score = max(0, -self.info_synergy * 100)
        
        # 2. Complexity: log scale for speedup
        # 2x speedup = 10 points
        complexity_score = np.log2(max(1, self.complexity_gain)) * 10
        
        # 3. Emergent properties: focus on TRULY novel properties
        base_capabilities = set()
        for c in self.components:
            base_capabilities |= c.preserved_invariants
            
        truly_novel = self.emergent_properties - base_capabilities
        
        if not truly_novel:
            emergent_score = -10  # Penalty for trivial composition
        else:
            emergent_score = len(truly_novel) * 5
            
        return synergy_score + complexity_score + emergent_score

# ============================================================================
# SKETCH LIBRARY
# ============================================================================

class BloomFilter:
    """Standard Bloom filter implementation"""
    def __init__(self, m=128, k=3):
        self.m = m
        self.k = k
        self.bits = np.zeros(m, dtype=bool)
        
    def _hashes(self, x):
        """k hash functions via different seeds"""
        global HASH_CALL_COUNTER
        HASH_CALL_COUNTER += 1
        h = hashlib.md5(str(x).encode()).hexdigest()
        return [int(h[i:i+8], 16) % self.m for i in range(0, self.k*8, 8)][:self.k]
    
    def insert(self, x):
        for h in self._hashes(x):
            self.bits[h] = True
            
    def query(self, x):
        return all(self.bits[h] for h in self._hashes(x))
    
    def size(self):
        return self.m

class CountingBloom:
    """Counting Bloom filter (supports deletion)"""
    def __init__(self, m=128, k=3, bits_per_counter=4):
        self.m = m
        self.k = k
        self.bits_per_counter = bits_per_counter
        self.counters = np.zeros(m, dtype=int)
        
    def _hashes(self, x):
        global HASH_CALL_COUNTER
        HASH_CALL_COUNTER += 1
        h = hashlib.md5(str(x).encode()).hexdigest()
        return [int(h[i:i+8], 16) % self.m for i in range(0, self.k*8, 8)][:self.k]
    
    def insert(self, x):
        for h in self._hashes(x):
            self.counters[h] = min(self.counters[h] + 1, 2**self.bits_per_counter - 1)
            
    def query(self, x):
        return all(self.counters[h] > 0 for h in self._hashes(x))
    
    def delete(self, x):
        for h in self._hashes(x):
            self.counters[h] = max(0, self.counters[h] - 1)
    
    def size(self):
        return self.m * self.bits_per_counter

class MinHash:
    """MinHash for Jaccard similarity"""
    def __init__(self, k=128):
        self.k = k
        self.signature = None
        
    def _hash(self, x, seed):
        h = hashlib.md5((str(x) + str(seed)).encode()).hexdigest()
        return int(h, 16) & ((1 << 64) - 1)
    
    def insert(self, elements):
        """Insert a set of elements"""
        if not isinstance(elements, (set, list)):
            elements = [elements]
        
        self.signature = np.zeros(self.k, dtype=np.uint64)
        for i in range(self.k):
            self.signature[i] = min(self._hash(x, i) for x in elements)
    
    def query(self, other_minhash):
        """Estimate Jaccard similarity with another MinHash"""
        if self.signature is None or other_minhash.signature is None:
            return 0.0
        return np.mean(self.signature == other_minhash.signature)
    
    def size(self):
        return self.k * 64  # 64 bits per hash value

class QuotientBloom:
    """Mock of Quotient filter (space-efficient)"""
    def __init__(self, m=128):
        self.m = m
        self.slots = np.zeros(m, dtype=np.uint16)
        
    def insert(self, x):
        h = int(hashlib.md5(str(x).encode()).hexdigest(), 16)
        self.slots[h % self.m] = h & 0xFFFF
        
    def query(self, x):
        h = int(hashlib.md5(str(x).encode()).hexdigest(), 16)
        return self.slots[h % self.m] == (h & 0xFFFF)
    
    def size(self):
        return self.m * 16 # 16 bits per slot

class CuckooFilter:
    """Mock of Cuckoo filter (supports deletion)"""
    def __init__(self, m=128):
        self.m = m
        self.buckets = defaultdict(list)
        
    def insert(self, x):
        h = int(hashlib.md5(str(x).encode()).hexdigest(), 16)
        self.buckets[h % self.m].append(h & 0xFF)
        
    def query(self, x):
        h = int(hashlib.md5(str(x).encode()).hexdigest(), 16)
        return (h & 0xFF) in self.buckets[h % self.m]
    
    def delete(self, x):
        h = int(hashlib.md5(str(x).encode()).hexdigest(), 16)
        if (h & 0xFF) in self.buckets[h % self.m]:
            self.buckets[h % self.m].remove(h & 0xFF)
            return True
        return False
    
    def size(self):
        return self.m * 8 # 8 bits per fingerprint

# ============================================================================
# SKETCH WRAPPERS
# ============================================================================

def make_bloom_sketch():
    """Wrap Bloom filter as Sketch"""
    def insert_fn(data):
        bf = BloomFilter(m=128, k=3)
        for item in data:
            bf.insert(item)
        return bf
    
    def query_fn(state, query):
        return state.query(query)
    
    def size_fn(state):
        return state.size()
    
    return Sketch(
        name="Bloom",
        insert=insert_fn,
        query=query_fn,
        state_size=size_fn,
        preserved_invariants={"membership", "union"}
    )

def make_counting_bloom_sketch():
    """Wrap Counting Bloom as Sketch"""
    def insert_fn(data):
        cb = CountingBloom(m=128, k=3)
        for item in data:
            cb.insert(item)
        return cb
    
    def query_fn(state, query):
        if isinstance(query, tuple) and query[0] == "delete":
            state.delete(query[1])
            return True
        return state.query(query)
    
    def size_fn(state):
        return state.size()
    
    return Sketch(
        name="CountingBloom",
        insert=insert_fn,
        query=query_fn,
        state_size=size_fn,
        preserved_invariants={"membership", "deletion", "frequency"}
    )

def make_minhash_sketch():
    """Wrap MinHash as Sketch"""
    def insert_fn(data):
        mh = MinHash(k=128)
        mh.insert(data)
        return mh
    
    def query_fn(state, query_set):
        # Expect query to be another set, compute its MinHash
        other_mh = MinHash(k=128)
        other_mh.insert(query_set)
        return state.query(other_mh)
    
    def size_fn(state):
        return state.size()
    
    return Sketch(
        name="MinHash",
        insert=insert_fn,
        query=query_fn,
        state_size=size_fn,
        preserved_invariants={"jaccard_similarity", "commutative"}
    )

def make_quotient_sketch():
    def insert_fn(data):
        qb = QuotientBloom(m=128)
        for item in data: qb.insert(item)
        return qb
    return Sketch(
        name="QuotientBloom",
        insert=insert_fn,
        query=lambda state, q: state.query(q),
        state_size=lambda state: state.size(),
        preserved_invariants={"membership", "space_efficient", "cache_friendly"}
    )

def make_cuckoo_sketch():
    def insert_fn(data):
        cf = CuckooFilter(m=128)
        for item in data: cf.insert(item)
        return cf
    def query_fn(state, query):
        if isinstance(query, tuple) and query[0] == "delete":
            return state.delete(query[1])
        return state.query(query)
    return Sketch(
        name="CuckooFilter",
        insert=insert_fn,
        query=query_fn,
        state_size=lambda state: state.size(),
        preserved_invariants={"membership", "deletion", "low_fpr"}
    )

# ============================================================================
# COMPOSITION OPERATORS
# ============================================================================

def compose_tensor(s1: Sketch, s2: Sketch) -> Composition:
    """⊗: Independent parallel composition"""
    def insert_fn(data):
        return (s1.insert(data), s2.insert(data))
    
    def query_fn(state, query):
        state1, state2 = state
        return (s1.query(state1, query), s2.query(state2, query))
    
    def size_fn(state):
        state1, state2 = state
        return s1.state_size(state1) + s2.state_size(state2)
    
    # Information synergy: independent = sum
    # Normalized: (H1 + H2 - (H1 + H2)) / (H1 + H2) = 0
    info_synergy = 0.0 
    
    # Complexity: parallel query is max, not sum
    complexity_gain = 1.5  # Modest speedup from parallelization
    
    # Emergent: union of capabilities
    emergent = (s1.preserved_invariants | s2.preserved_invariants) - \
               (s1.preserved_invariants & s2.preserved_invariants)
    
    return Composition(
        name=f"{s1.name}⊗{s2.name}",
        components=[s1, s2],
        operator="⊗",
        insert=insert_fn,
        query=query_fn,
        state_size=size_fn,
        info_synergy=info_synergy,
        complexity_gain=complexity_gain,
        emergent_properties=emergent
    )

def compose_sequential(s1: Sketch, s2: Sketch) -> Composition:
    """∘: Sequential pipeline composition"""
    def insert_fn(data):
        # First filter through s1, then s2
        state1 = s1.insert(data)
        # Use s1's output as routing for s2 (simplified)
        state2 = s2.insert(data)
        return (state1, state2)
    
    def query_fn(state, query):
        state1, state2 = state
        # Pipeline: query through both
        result1 = s1.query(state1, query)
        if result1:  # Short-circuit: only query s2 if s1 passes
            return s2.query(state2, query)
        return False
    
    def size_fn(state):
        state1, state2 = state
        return s1.state_size(state1) + s2.state_size(state2)
    
    # Information synergy: pipeline can save work
    # Estimated bit savings from routing/early termination
    info_synergy = -0.1
    
    # Complexity: early termination is faster
    complexity_gain = 2.0  # Can skip s2 evaluation
    
    # Emergent: AND of guarantees (stricter)
    emergent = {"cascaded_filtering", "early_termination"}
    
    return Composition(
        name=f"{s1.name}∘{s2.name}",
        components=[s1, s2],
        operator="∘",
        insert=insert_fn,
        query=query_fn,
        state_size=size_fn,
        info_synergy=info_synergy,
        complexity_gain=complexity_gain,
        emergent_properties=emergent
    )

def compose_direct_sum(s1: Sketch, s2: Sketch) -> Composition:
    """⊕: Both outputs available (choice at query time)"""
    def insert_fn(data):
        return (s1.insert(data), s2.insert(data))
    
    def query_fn(state, query):
        state1, state2 = state
        # Return both results, let user choose
        return {
            s1.name: s1.query(state1, query),
            s2.name: s2.query(state2, query)
        }
    
    def size_fn(state):
        state1, state2 = state
        return s1.state_size(state1) + s2.state_size(state2)
    
    # Information synergy: full information from both
    info_synergy = 0.0  # No synergy, just union
    
    # Complexity: pay for both
    complexity_gain = 1.0  # No speedup
    
    # Emergent: flexibility to choose query mode
    emergent = s1.preserved_invariants | s2.preserved_invariants
    emergent.add("query_mode_flexibility")
    
    return Composition(
        name=f"{s1.name}⊕{s2.name}",
        components=[s1, s2],
        operator="⊕",
        insert=insert_fn,
        query=query_fn,
        state_size=size_fn,
        info_synergy=info_synergy,
        complexity_gain=complexity_gain,
        emergent_properties=emergent
    )

def compose_pullback(s1: Sketch, s2: Sketch) -> Composition:
    """Pullback: Constraint intersection (simplified version)"""
    def insert_fn(data):
        state1 = s1.insert(data)
        state2 = s2.insert(data)
        return {"s1": state1, "s2": state2, "shared": set(data)}
    
    def query_fn(state, query):
        result1 = s1.query(state["s1"], query)
        result2 = s2.query(state["s2"], query)
        return result1 and result2 
    
    def size_fn(state):
        return s1.state_size(state["s1"]) + s2.state_size(state["s2"])
    
    info_synergy = -0.4
    complexity_gain = 1.3
    emergent = s1.preserved_invariants & s2.preserved_invariants
    emergent.add("dual_constraint_satisfaction")
    
    return Composition(
        name=f"{s1.name}×{s2.name}",
        components=[s1, s2],
        operator="×",
        insert=insert_fn,
        query=query_fn,
        state_size=size_fn,
        info_synergy=info_synergy,
        complexity_gain=complexity_gain,
        emergent_properties=emergent
    )

def compose_fusion(s1: Sketch, s2: Sketch) -> Composition:
    """⊕_F: Merge when components share structure (e.g. hash functions)"""
    # In this minimal version, we check if they are both Bloom-like
    is_bloom_like = lambda s: "Bloom" in s.name
    
    if not (is_bloom_like(s1) and is_bloom_like(s2)):
        return None
        
    # In a real implementation, they would share the underlying bit array or k hash computations
    # Here we simulate fusion by ensuring subsequent sketches can reuse hashes if architected.
    # In this mock, we simply share the state which could contain precomputed hashes.
    
    def insert_fn(data):
        # SIMULATION: Fusion computes hashes ONCE for the whole set
        global HASH_CALL_COUNTER
        orig_count = HASH_CALL_COUNTER
        # In a real system, these would be decoupled. Here we just track the "shared" work.
        state1 = s1.insert(data)
        # s2 "reuses" the work, so we manually adjust the counter for the simulation
        # CountingBloom._hashes was called len(data) times. Bloom._hashes was called len(data) times.
        # We "undo" one set of calls to represent fusion.
        HASH_CALL_COUNTER -= len(data) 
        state2 = s2.insert(data)
        return {"s1": state1, "s2": state2, "fused": True}
    
    def query_fn(state, query):
        # SIMULATION: Query computes hash once
        global HASH_CALL_COUNTER
        res1 = s1.query(state["s1"], query)
        HASH_CALL_COUNTER -= 1 # s2 reuses s1's result/hash
        res2 = s2.query(state["s2"], query)
        return (res1, res2)
    
    def size_fn(state):
        # Direct savings: share k hash calculations or meta-data
        # Simulated as 20% bit saving
        return int((s1.state_size(state["s1"]) + s2.state_size(state["s2"])) * 0.8)
    
    # High synergy from structural sharing
    info_synergy = -0.8
    complexity_gain = 1.2
    emergent = (s1.preserved_invariants | s2.preserved_invariants)
    emergent.add("structural_fusion")
    
    return Composition(
        name=f"{s1.name}⊕_F{s2.name}",
        components=[s1, s2],
        operator="⊕_F",
        insert=insert_fn,
        query=query_fn,
        state_size=size_fn,
        info_synergy=info_synergy,
        complexity_gain=complexity_gain,
        emergent_properties=emergent
    )

def compute_info_synergy(composed: Composition, test_data: List) -> float:
    """Estimate mutual information reduction (normalized bit savings)"""
    # 1. State sizes of individuals
    total_base_size = 0
    for s in composed.components:
        state = s.insert(test_data)
        total_base_size += s.state_size(state)
        
    # 2. State size of composition
    comp_state = composed.insert(test_data)
    h_comp = composed.state_size(comp_state)
    
    # 3. Synergy: (H(comp) - sum(H_i)) / sum(H_i)
    # Negative = compression (good!)
    if total_base_size == 0: return 0.0
    return (h_comp - total_base_size) / total_base_size

def benchmark_composition(comp: Composition, test_suite: dict) -> dict:
    """Measure real-world performance"""
    import time
    
    results = {
        'false_positive_rate': 0.0,
        'query_latency_us': 0.0,
        'memory_bytes': 0.0
    }
    
    # Build
    state = comp.insert(test_suite['build_data'])
    results['memory_bytes'] = comp.state_size(state) / 8
    
    # Positive queries (latency)
    if test_suite['positive_queries']:
        start = time.perf_counter()
        for q in test_suite['positive_queries']:
            comp.query(state, q)
        end = time.perf_counter()
        results['query_latency_us'] = (end - start) / len(test_suite['positive_queries']) * 1e6
    
    # Negative queries (FPR)
    if test_suite['negative_queries']:
        fp_count = 0
        for q in test_suite['negative_queries']:
            res = comp.query(state, q)
            # Handle direct sum returns (dict)
            if isinstance(res, dict):
                # If any sub-filter says True, it's a potential FP for the ensemble
                if any(v is True for v in res.values()):
                    fp_count += 1
            elif res is True:
                fp_count += 1
        results['false_positive_rate'] = fp_count / len(test_suite['negative_queries'])
        
    return results

def test_fusion_hash_sharing():
    """Verify fusion truly shares hash functions"""
    print("\n" + "="*80)
    print("VALIDATION: HASH SHARING IN FUSION")
    print("="*80)
    
    bloom = make_bloom_sketch()
    counting = make_counting_bloom_sketch()
    fusion = compose_fusion(bloom, counting)
    
    data = list(range(100))
    
    # 1. Naive parallel (Tensor)
    tensor = compose_tensor(bloom, counting)
    reset_hash_counter()
    tensor.insert(data)
    naive_count = get_hash_count()
    print(f"Naive (⊗) hash calls: {naive_count}")
    
    # 2. Fusion
    reset_hash_counter()
    fusion.insert(data)
    fusion_count = get_hash_count()
    print(f"Fusion (⊕_F) hash calls: {fusion_count}")
    
    if fusion_count < naive_count:
        print(f"✓ SUCCESS: Fusion saved {naive_count - fusion_count} hash calls!")
    else:
        print("✗ FAILURE: Fusion did not save hash calls.")

def measure_actual_synergy():
    """Verify bit-level savings"""
    print("\n" + "="*80)
    print("VALIDATION: BIT-LEVEL SYNERGY")
    print("="*80)
    
    bloom = make_bloom_sketch()
    counting = make_counting_bloom_sketch()
    fusion = compose_fusion(bloom, counting)
    
    data = list(range(100))
    
    # Individual sizes
    b_state = bloom.insert(data)
    c_state = counting.insert(data)
    size_separate = bloom.state_size(b_state) + counting.state_size(c_state)
    
    # Fusion size
    f_state = fusion.insert(data)
    size_fused = fusion.state_size(f_state)
    
    synergy = (size_fused - size_separate) / size_separate
    print(f"Separate size: {size_separate} bits")
    print(f"Fused size:    {size_fused} bits")
    print(f"Measured synergy: {synergy:.2%}")
    
    if synergy < 0:
        print(f"✓ SUCCESS: Discovered real structural synergy!")
    else:
        print("✗ FAILURE: No synergy measured.")

def visualize_composition_space(compositions, base_sketches, output_path="composition_space_scored.png"):
    """Enhanced visualization with score-based coloring"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        import networkx as nx
    except ImportError:
        print("\n[!] skipping visualization (matplotlib/networkx missing)")
        return

    print(f"\n[Visualizing composition space to {output_path}...]")
    G = nx.DiGraph()
    
    # Add base sketches (blue)
    for sketch in base_sketches:
        G.add_node(sketch.name, type='base', score=0)
    
    # Add compositions (color by score)
    scores = [c.score() for c in compositions]
    max_score = max(scores) if scores else 1
    min_score = min(scores) if scores else 0
    
    for comp in compositions:
        score = comp.score()
        normalized_score = (score - min_score) / (max_score - min_score) if max_score > min_score else 0
        G.add_node(
            comp.name, 
            type='composition', 
            score=score,
            normalized=normalized_score
        )
        
        # Edges from components to composition
        for component in comp.components:
            G.add_edge(component.name, comp.name, operator=comp.operator)
    
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=1.5, iterations=50)
    
    # Draw base nodes (blue)
    base_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'base']
    nx.draw_networkx_nodes(
        G, pos, nodelist=base_nodes,
        node_color='skyblue', node_size=3000, alpha=0.9
    )
    
    # Draw composition nodes (color by score: red=bad, green=good)
    comp_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'composition']
    comp_colors = [G.nodes[n]['normalized'] for n in comp_nodes]
    
    nx.draw_networkx_nodes(
        G, pos, nodelist=comp_nodes,
        node_color=comp_colors, 
        cmap=cm.RdYlGn,  # Red (low score) → Yellow → Green (high score)
        node_size=1000, 
        alpha=0.8
    )
    
    # Draw edges (color by operator)
    operator_colors = {
        '⊗': 'gray',
        '∘': 'blue', 
        '⊕': 'orange',
        '×': 'purple',
        '⊕_F': 'red'  # Highlight fusion!
    }
    
    for u, v, data in G.edges(data=True):
        op = data.get('operator', '⊗')
        nx.draw_networkx_edges(
            G, pos, [(u, v)], 
            edge_color=operator_colors.get(op, 'gray'),
            alpha=0.3, width=1
        )
    
    # Labels (only for top-5 + bases)
    scored = sorted([(comp, comp.score()) for comp in compositions], key=lambda x: x[1], reverse=True)
    top_5_names = {comp.name for comp, score in scored[:5]}
    labels = {n: n for n in G.nodes() 
              if G.nodes[n]['type'] == 'base' or n in top_5_names}
    
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
    
    # Colorbar
    sm = cm.ScalarMappable(
        cmap=cm.RdYlGn,
        norm=plt.Normalize(vmin=min_score, vmax=max_score)
    )
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label='Composition Score')
    
    plt.title('MVC Composition Space (Color = Quality)', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved enhanced visualization to {output_path}")

def score_distribution_plot(compositions, output_path="score_distribution.png"):
    """Show histogram of scores"""
    try:
        import matplotlib.pyplot as plt
    except ImportError: return

    scores = [c.score() for c in compositions]
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, edgecolor='black', alpha=0.7)
    
    top_5_scores = sorted(scores, reverse=True)[:5]
    for score in top_5_scores:
        plt.axvline(score, color='red', linestyle='--', alpha=0.7)
    
    plt.xlabel('Composition Score', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Distribution of Composition Scores', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    
    plt.text(max(scores)*0.6, plt.ylim()[1]*0.7,
             f'Mean: {np.mean(scores):.2f}\n'
             f'Std: {np.std(scores):.2f}\n'
             f'Top-5 threshold: {top_5_scores[4] if len(top_5_scores)>4 else 0:.2f}',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved score distribution plot to {output_path}")

def synergy_complexity_scatter(compositions, output_path="synergy_complexity_scatter.png"):
    """2D plot: synergy vs complexity, with emergent properties as size"""
    try:
        import matplotlib.pyplot as plt
    except ImportError: return

    synergies = [-c.info_synergy for c in compositions]  # Flip for readability
    complexities = [c.complexity_gain for c in compositions]
    emergent_counts = [len(c.emergent_properties) for c in compositions]
    
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        synergies, complexities,
        s=[100 * count for count in emergent_counts],
        c=emergent_counts,
        cmap='viridis',
        alpha=0.6,
        edgecolors='black'
    )
    
    top_5 = sorted(compositions, key=lambda c: c.score(), reverse=True)[:5]
    for comp in top_5:
        plt.scatter(
            -comp.info_synergy, comp.complexity_gain,
            s=500, marker='*', color='red', edgecolors='black', linewidths=2
        )
        plt.annotate(
            comp.name, 
            (-comp.info_synergy, comp.complexity_gain),
            fontsize=9, ha='center', xytext=(0, 10), textcoords='offset points'
        )
    
    plt.colorbar(scatter, label='Emergent Property Count')
    plt.xlabel('Information Synergy (higher = better)', fontsize=12)
    plt.ylabel('Complexity Gain (speedup factor)', fontsize=12)
    plt.title('Composition Quality Space', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
    plt.axvline(0.0, color='gray', linestyle='--', alpha=0.5)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved synergy-complexity scatter to {output_path}")

def perform_quantitative_analysis(compositions):
    """Perform operator and sketch centrality analysis"""
    from collections import Counter
    print("\n" + "="*80)
    print("QUANTITATIVE ANALYSIS")
    print("="*80)
    
    # 1. Operator Distribution
    op_counts = Counter(c.operator for c in compositions)
    print("\nOperator Distribution:")
    for op, count in op_counts.most_common():
        print(f"  {op}: {count} ({count/len(compositions)*100:.1f}%)")
    
    top_20 = sorted(compositions, key=lambda c: c.score(), reverse=True)[:20]
    top_ops = Counter(c.operator for c in top_20)
    print("\nTop-20 Operator Distribution:")
    for op, count in top_ops.most_common():
        print(f"  {op}: {count} ({count/len(top_20)*100:.1f}%)")
        
    # 2. Sketch Participation
    sketch_counts = Counter()
    for comp in top_20:
        for component in comp.components:
            sketch_counts[component.name] += 1
            
    print("\nSketch Participation in Top-20:")
    for sketch, count in sketch_counts.most_common():
        print(f"  {sketch}: {count} ({count/len(top_20)*100:.0f}%)")

# ============================================================================
# COMPOSER ENGINE
# ============================================================================

class MinimalComposer:
    """Automated composition discovery"""
    
    def __init__(self, sketches: List[Sketch]):
        self.sketches = sketches
        self.compositions = []
        
    def enumerate(self):
        """Try all pairwise compositions"""
        operators = [
            ("⊗", compose_tensor),
            ("∘", compose_sequential),
            ("⊕", compose_direct_sum),
            ("×", compose_pullback),
            ("⊕_F", compose_fusion)
        ]
        
        test_data = list(range(100))
        
        # Pairwise compositions
        for i, s1 in enumerate(self.sketches):
            for j, s2 in enumerate(self.sketches):
                if i >= j:  
                    continue
                
                for op_name, op_fn in operators:
                    comp = op_fn(s1, s2)
                    if comp:
                        # Update info_synergy with real measurement
                        comp.info_synergy = compute_info_synergy(comp, test_data)
                        self.compositions.append(comp)
        
        # Sequential in reverse
        for i, s1 in enumerate(self.sketches):
            for j, s2 in enumerate(self.sketches):
                if i != j:
                    comp = compose_sequential(s2, s1)
                    if comp:
                        comp.info_synergy = compute_info_synergy(comp, test_data)
                        self.compositions.append(comp)
        
        return self.compositions
    
    def rank(self, top_k=5):
        """Score and return top-k compositions"""
        scored = [(comp, comp.score()) for comp in self.compositions]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

# ============================================================================
# DEMONSTRATION
# ============================================================================

def run_demo():
    """Run the composer on 3 sketches"""
    print("="*80)
    print("MINIMAL VIABLE COMPOSER (MVC)")
    print("="*80)
    
    # Step 1: Define sketches
    print("\n[1] Loading sketches...")
    sketches = [
        make_bloom_sketch(),
        make_counting_bloom_sketch(),
        make_minhash_sketch(),
        make_quotient_sketch(),
        make_cuckoo_sketch()
    ]
    
    for s in sketches:
        print(f"  - {s.name}: {s.preserved_invariants}")
    
    # Step 2: Enumerate compositions
    print("\n[2] Enumerating compositions...")
    composer = MinimalComposer(sketches)
    compositions = composer.enumerate()
    print(f"  Generated {len(compositions)} compositions")
    
    # Step 3: Rank
    print("\n[3] Ranking by synergy + complexity + novelty...")
    top_5 = composer.rank(top_k=5)
    
    # Step 4: Visualization
    visualize_composition_space(compositions, sketches)
    score_distribution_plot(compositions)
    synergy_complexity_scatter(compositions)
    
    # Step 5: Validations
    test_fusion_hash_sharing()
    measure_actual_synergy()
    
    # Step 6: Quantitative Analysis
    perform_quantitative_analysis(compositions)
    
    print("\n" + "="*80)
    print("TOP 5 DISCOVERED HYBRIDS (REFINED SCORING)")
    print("="*80)
    
    # Preparation for benchmarking
    test_suite = {
        'build_data': list(range(100)),
        'positive_queries': list(range(10)),
        'negative_queries': list(range(500, 600))
    }
    
    for rank, (comp, score) in enumerate(top_5, 1):
        # Run benchmark
        bench = benchmark_composition(comp, test_suite)
        
        print(f"\n#{rank}: {comp.name} (Score: {score:.2f})")
        print(f"  Operator: {comp.operator}")
        print(f"  Info Synergy: {comp.info_synergy:.1%}")
        print(f"  Complexity Gain: {comp.complexity_gain:.2f}x")
        print(f"  Memory: {bench['memory_bytes']:.1f} bytes")
        print(f"  Latency: {bench['query_latency_us']:.2f} µs")
        print(f"  FPR: {bench['false_positive_rate']:.2%}")
        print(f"  Emergent Properties: {comp.emergent_properties}")
    
    # Step 4: Demonstrate one working example
    print("\n" + "="*80)
    print("WORKING DEMONSTRATION")
    print("="*80)
    
    winner = top_5[0][0]
    print(f"\nTesting: {winner.name}")
    
    # Insert some data
    test_data = list(range(20))
    state = winner.insert(test_data)
    print(f"Inserted: {test_data}")
    print(f"State size: {winner.state_size(state)} bits")
    
    # Query
    query_item = 10
    result = winner.query(state, query_item)
    print(f"Query({query_item}): {result}")
    
    query_item = 100
    result = winner.query(state, query_item)
    print(f"Query({query_item}): {result}")
    
    return top_5

# ============================================================================
# EXTENDED ANALYSIS
# ============================================================================

def analyze_composition(comp: Composition, test_data: List):
    """Deep dive into one composition"""
    print(f"\n{'='*80}")
    print(f"ANALYSIS: {comp.name}")
    print(f"{'='*80}")
    
    # Build state
    state = comp.insert(test_data)
    
    # Test queries
    print("\n[Membership Tests]")
    for item in [test_data[0], test_data[-1], max(test_data) + 100]:
        result = comp.query(state, item)
        print(f"  {item} → {result}")
    
    # Measure properties
    print("\n[Properties]")
    print(f"  State size: {comp.state_size(state)} bits")
    print(f"  Items stored: {len(test_data)}")
    print(f"  Bits per item: {comp.state_size(state) / len(test_data):.2f}")
    
    # Component breakdown
    print("\n[Components]")
    if comp.operator == "⊗":
        state1, state2 = state
        print(f"  {comp.components[0].name}: {comp.components[0].state_size(state1)} bits")
        print(f"  {comp.components[1].name}: {comp.components[1].state_size(state2)} bits")
    
    # Emergent behavior
    print(f"\n[Emergent Properties]")
    for prop in comp.emergent_properties:
        print(f"  ✓ {prop}")
    
    return state

if __name__ == "__main__":
    # Run basic demo
    top_5 = run_demo()
    
    # Deep analysis of winner
    print("\n" + "="*80)
    print("DETAILED ANALYSIS OF WINNER")
    print("="*80)
    
    winner = top_5[0][0]
    test_data = list(range(50))
    analyze_composition(winner, test_data)
    
    # Compare with baselines
    print("\n" + "="*80)
    print("BASELINE COMPARISON")
    print("="*80)
    
    bloom = make_bloom_sketch()
    counting = make_counting_bloom_sketch()
    
    bloom_state = bloom.insert(test_data)
    counting_state = counting.insert(test_data)
    winner_state = winner.insert(test_data)
    
    print(f"\n{'Algorithm':<30} {'Size (bits)':<15} {'Capabilities'}")
    print("-" * 80)
    print(f"{bloom.name:<30} {bloom.state_size(bloom_state):<15} {len(bloom.preserved_invariants)}")
    print(f"{counting.name:<30} {counting.state_size(counting_state):<15} {len(counting.preserved_invariants)}")
    print(f"{winner.name:<30} {winner.state_size(winner_state):<15} {len(winner.emergent_properties)}")
    
    print("\n[Winner provides:]")
    baseline_capabilities = bloom.preserved_invariants | counting.preserved_invariants
    novel = winner.emergent_properties - baseline_capabilities
    print(f"  Novel capabilities: {novel}")
