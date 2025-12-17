"""
Spectral-CRDT Witness Fields (SCWF)

A Thermodynamically Stable Model of Distributed Semantics

This package implements a genuinely novel computational primitive that combines:
- Information-theoretic bounds (witness sets)
- Algebraic semantics (commutative idempotent monoid)
- Distributed systems (CRDTs for coordination-free merging)
- Spectral analysis (semantic FFT)
- Safety-driven refusal (negative evidence dominance)

Key components:
- WitnessAlgebra: The (W, ⊕, ★) algebraic structure
- SpectralTransform: T: W ↔ Ŵ (semantic FFT)
- SpectralCRDT: Conflict-free merging in spectral domain
- WitnessFlow: Thermodynamic evolution equations
- HallucinationDetector: Proves hallucination is impossible

This is not "AI alignment". This is SEMANTIC PHYSICS.

References:
- Theorem 1: Bellman-REWA Equivalence (DP is thermodynamic equilibrium)
- Theorem 2: Spectral Diagonalization (semantic FFT)
- Theorem 3: Negative Evidence Dominance (refusal is faster than approval)
"""

from witness_algebra import (
    Witness,
    WitnessField,
    WitnessPolarity,
    WitnessAlgebra,
    join_fields,
    convolve_fields,
)

from spectral_transform import (
    SpectralWitnessField,
    SpectralWitnessTransform,
    to_spectral,
    from_spectral,
    spectral_compose,
    spectral_join,
)

from crdt_merge import (
    CRDTSpectralField,
    SpectralCRDT,
    MergeStrategy,
    VectorClock,
    DistributedWitnessNetwork,
    merge_spectral_fields,
)

from witness_flow import (
    WitnessFlowEquation,
    FlowType,
    FlowParameters,
    WitnessFlowSimulator,
    BellmanWitnessFlow,
    evolve_field,
    find_semantic_equilibrium,
)

from hallucination_impossibility import (
    QueryResult,
    QueryMetrics,
    NegativeEvidenceDominance,
    HallucinationDetector,
    ConsistencyProof,
    run_hallucination_impossibility_demo,
)

__version__ = "0.1.0"
__author__ = "SCWF Research"
__doc_title__ = "Spectral-CRDT Witness Fields: A Thermodynamically Stable Model of Distributed Semantics"
