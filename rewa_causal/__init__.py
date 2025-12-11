"""
REWA-Causal Engine v1.0

A geometric causal reasoning engine that operates on:
- Witness sets extracted from text, tables, or structured data
- Semantic manifold operations (spherical convex hulls, distances, hull intersections)
- Causal identification rules (backdoor, front-door, do-operator)
- Policy-safe decisioning (refusal if geometry invalid)

This engine never hallucinates causal statements and relies strictly on
mathematically admissible regions.
"""

__version__ = "1.0.0"

from .witness import WitnessSet, extract_witnesses, normalize_witness
from .geometry import (
    spherical_convex_hull,
    satisfies_hemisphere_constraint,
    frechet_mean,
    compute_dispersion,
    geodesic_distance,
    hull_overlap,
    weighted_union
)
from .causal_graph import CausalGraph
from .identification import (
    backdoor_adjustment,
    frontdoor_adjustment,
    interventional_region,
    causal_effect
)
from .refusal import validate_region, validate_causal_claim, RefusalResult

__all__ = [
    'WitnessSet',
    'extract_witnesses',
    'normalize_witness',
    'spherical_convex_hull',
    'satisfies_hemisphere_constraint',
    'frechet_mean',
    'compute_dispersion',
    'geodesic_distance',
    'hull_overlap',
    'weighted_union',
    'CausalGraph',
    'backdoor_adjustment',
    'frontdoor_adjustment',
    'interventional_region',
    'causal_effect',
    'validate_region',
    'validate_causal_claim',
    'RefusalResult'
]
