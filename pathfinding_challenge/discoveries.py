from .sketches import (
    make_dijkstra_sketch, make_astar_sketch, make_bidirectional_sketch,
    make_ch_sketch, make_landmark_alt_sketch
)
from .heat_diffusion import make_heat_diffusion_sketch
from .operators import (
    compose_tensor, compose_sequential, compose_fusion, compose_pullback
)

def run_discovery():
    """Discover pathfinding compositions"""
    dijkstra = make_dijkstra_sketch()
    astar = make_astar_sketch()
    bidirectional = make_bidirectional_sketch()
    ch = make_ch_sketch()
    landmark = make_landmark_alt_sketch()
    heat = make_heat_diffusion_sketch()

    compositions = []

    # 1. Dijkstra ⊗ AStar
    c1 = compose_tensor(dijkstra, astar)
    compositions.append(c1)

    # 2. AStar ∘ Bidirectional (Bidirectional A*)
    bi_astar = compose_sequential(astar, bidirectional)
    bi_astar.name = "BidirectionalA*"
    compositions.append(bi_astar)

    # 3. CH ⊕_F Landmark (Fused Preprocessing)
    ch_alt = compose_fusion(ch, landmark)
    ch_alt.name = "CH_ALT_Fusion"
    compositions.append(ch_alt)

    # 4. (Bi-A* ⊕_F CH) (Hierarchical Bidirectional A*)
    # Here we compose Bi-A* with CH using Fusion
    # We wrap bi_astar as a sketch-like object for composition
    # In MVC, compositions can often be treated as sketches
    
    # Let's define it directly for simplicity in this PoC
    hb_astar = compose_fusion(bi_astar, ch)
    hb_astar.name = "HierarchicalBidirectionalA*"
    compositions.append(hb_astar)

    # 5. ((Bi-A* ⊕_F CH) × Landmark) (Triple Constrained Search)
    triple_hybrid = compose_pullback(hb_astar, landmark)
    triple_hybrid.name = "TripleConstrainedSearch"
    compositions.append(triple_hybrid)

    # 6. (HeatDiffusion ⊕_F CH) (Discrete Varadhan Sketch)
    varadhan_sketch = compose_fusion(heat, ch)
    varadhan_sketch.name = "VaradhanSketch"
    compositions.append(varadhan_sketch)

    return compositions, [dijkstra, astar, bidirectional, ch, landmark, heat]
