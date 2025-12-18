"""
Unified Regulator Engine
========================

A thermodynamically regulated computation engine where inference
is relaxation into stable attractors.

Core Principle:
    Computation is controlled energy flow on a manifold, regulated by dissipation.

Usage:
    from unified_regulator_engine import UnifiedRegulatorEngine, Mode

    engine = UnifiedRegulatorEngine()
    engine.build_index(vectors)
    result = engine.retrieve(query, k=10)
"""

from .ure_core import (
    # Core types
    Mode,
    RegulatorParams,
    RegulatorState,
    RegulatorResult,

    # Graph construction
    build_knn_graph,
    build_grid_graph,
    build_epsilon_graph,

    # Operators
    op_wave,
    op_telegrapher,
    op_fokker_planck,
    op_schrodinger_imag,
    op_cahn_hilliard,
    op_fisher_kpp,
    op_poisson,

    # Main regulator
    regulator,
    initialize_state,
    initialize_from_query,

    # High-level API
    UnifiedRegulatorEngine,

    # Convenience functions
    quick_retrieve,
    quick_cluster,
    quick_decide,
)

__all__ = [
    # Types
    "Mode",
    "RegulatorParams",
    "RegulatorState",
    "RegulatorResult",

    # Graph
    "build_knn_graph",
    "build_grid_graph",
    "build_epsilon_graph",

    # Operators
    "op_wave",
    "op_telegrapher",
    "op_fokker_planck",
    "op_schrodinger_imag",
    "op_cahn_hilliard",
    "op_fisher_kpp",
    "op_poisson",

    # Core
    "regulator",
    "initialize_state",
    "initialize_from_query",

    # API
    "UnifiedRegulatorEngine",

    # Convenience
    "quick_retrieve",
    "quick_cluster",
    "quick_decide",
]

__version__ = "1.0.0"
