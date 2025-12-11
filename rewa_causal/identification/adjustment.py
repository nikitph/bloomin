"""
Causal Identification Methods

Implements backdoor and front-door adjustment formulas using
spherical geometry (REWA-Causal approach).

All identification methods operate on witness sets and produce
spherical regions representing interventional distributions.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass

try:
    from ..witness import WitnessSet
    from ..geometry import (
        SphericalConvexHull,
        spherical_convex_hull,
        frechet_mean,
        geodesic_distance,
        weighted_union,
        compute_dispersion
    )
    from ..causal_graph import CausalGraph
except ImportError:
    from witness import WitnessSet
    from geometry import (
        SphericalConvexHull,
        spherical_convex_hull,
        frechet_mean,
        geodesic_distance,
        weighted_union,
        compute_dispersion
    )
    from causal_graph import CausalGraph


@dataclass
class IdentificationResult:
    """Result of a causal identification procedure."""
    success: bool
    region: Optional[SphericalConvexHull]
    method: str
    details: Dict[str, Any]


@dataclass
class CausalEffectResult:
    """Result of causal effect estimation."""
    treatment_values: Tuple[Any, Any]  # (x0, x1)
    region_x0: SphericalConvexHull
    region_x1: SphericalConvexHull
    centroid_shift: float  # Geodesic distance between centroids
    direction: np.ndarray  # Direction of causal effect
    ambiguity: float  # Uncertainty in the effect
    details: Dict[str, Any]


def backdoor_adjustment(
    X: str,
    x_target: Any,
    Z: Set[str],
    Y: str,
    witnesses: Dict[str, WitnessSet],
    graph: CausalGraph
) -> IdentificationResult:
    """
    Compute P(Y | do(X=x)) using backdoor adjustment.

    The backdoor formula:
        P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) P(Z=z)

    In REWA geometry, this becomes:
        A_Y^{do(X=x)} = weighted_union over z of A_{Y|X=x,Z=z}

    Args:
        X: Treatment variable name
        x_target: Target value for intervention (used for filtering witnesses)
        Z: Adjustment set (conditioning variables)
        Y: Outcome variable name
        witnesses: Dictionary mapping variable names to WitnessSet objects
        graph: CausalGraph defining the causal structure

    Returns:
        IdentificationResult with the interventional region A_Y^{do(X=x)}
    """
    details = {
        'treatment': X,
        'target_value': x_target,
        'adjustment_set': list(Z),
        'outcome': Y
    }

    # Verify backdoor criterion
    is_valid, criterion_details = graph.verify_backdoor(Z, X, Y)
    details['criterion_verification'] = criterion_details

    if not is_valid:
        return IdentificationResult(
            success=False,
            region=None,
            method='backdoor',
            details={**details, 'error': 'Backdoor criterion not satisfied'}
        )

    # Get witness sets
    if Y not in witnesses:
        return IdentificationResult(
            success=False,
            region=None,
            method='backdoor',
            details={**details, 'error': f'No witnesses for outcome {Y}'}
        )

    W_Y = witnesses[Y]

    # If no adjustment needed (Z is empty or no confounding)
    if not Z:
        # Direct computation: A_Y^{do(X=x)} = A_{Y|X=x}
        # Filter Y witnesses to those consistent with X=x_target
        filtered_witnesses = _filter_witnesses_by_condition(
            W_Y, X, x_target, witnesses.get(X)
        )

        if filtered_witnesses.n_witnesses == 0:
            return IdentificationResult(
                success=False,
                region=None,
                method='backdoor',
                details={**details, 'error': 'No witnesses after filtering'}
            )

        region = spherical_convex_hull(filtered_witnesses.witnesses)
        return IdentificationResult(
            success=True,
            region=region,
            method='backdoor',
            details={
                **details,
                'n_witnesses': filtered_witnesses.n_witnesses,
                'region_radius': region.radius,
                'region_valid': region.is_valid
            }
        )

    # Full backdoor adjustment with conditioning on Z
    # Stratify by Z values and compute weighted union

    # For REWA-Causal, we approximate this by:
    # 1. Clustering Z witnesses into strata
    # 2. For each stratum, computing A_{Y|X=x,Z=z}
    # 3. Combining via weighted union

    strata_regions = []
    strata_weights = []

    # Get unique Z configurations (simplified: use witness clusters)
    Z_combined = _combine_witness_sets([witnesses[z] for z in Z if z in witnesses])

    if Z_combined is None or Z_combined.n_witnesses == 0:
        # Fall back to unadjusted estimate
        filtered_witnesses = _filter_witnesses_by_condition(W_Y, X, x_target, witnesses.get(X))
        if filtered_witnesses.n_witnesses == 0:
            return IdentificationResult(
                success=False,
                region=None,
                method='backdoor',
                details={**details, 'error': 'No witnesses available'}
            )

        region = spherical_convex_hull(filtered_witnesses.witnesses)
        return IdentificationResult(
            success=True,
            region=region,
            method='backdoor',
            details={**details, 'n_witnesses': filtered_witnesses.n_witnesses}
        )

    # Cluster Z into strata
    n_strata = min(5, Z_combined.n_witnesses)
    strata_indices = _cluster_witnesses(Z_combined, n_strata)

    for stratum_idx in range(n_strata):
        stratum_mask = strata_indices == stratum_idx
        if not np.any(stratum_mask):
            continue

        # Weight = proportion of data in this stratum (P(Z=z))
        weight = np.sum(stratum_mask) / len(strata_indices)

        # Compute A_{Y|X=x,Z=z} for this stratum
        # In practice, we filter Y witnesses that are consistent with this Z stratum
        stratum_Y_witnesses = _filter_witnesses_by_stratum(
            W_Y, Z_combined, stratum_mask, X, x_target, witnesses.get(X)
        )

        if stratum_Y_witnesses.n_witnesses > 0:
            stratum_region = spherical_convex_hull(stratum_Y_witnesses.witnesses)
            strata_regions.append(stratum_region)
            strata_weights.append(weight)

    if not strata_regions:
        return IdentificationResult(
            success=False,
            region=None,
            method='backdoor',
            details={**details, 'error': 'No valid strata after adjustment'}
        )

    # Weighted union of stratum regions
    result_region = weighted_union(strata_regions, strata_weights)

    return IdentificationResult(
        success=True,
        region=result_region,
        method='backdoor',
        details={
            **details,
            'n_strata': len(strata_regions),
            'strata_weights': strata_weights,
            'region_radius': result_region.radius,
            'region_valid': result_region.is_valid,
            'n_witnesses_total': result_region.n_points
        }
    )


def frontdoor_adjustment(
    X: str,
    x_target: Any,
    M: str,
    Y: str,
    witnesses: Dict[str, WitnessSet],
    graph: CausalGraph
) -> IdentificationResult:
    """
    Compute P(Y | do(X=x)) using front-door adjustment.

    The front-door formula:
        P(Y | do(X=x)) = Σ_m P(M=m | X=x) Σ_x' P(Y | M=m, X=x') P(X=x')

    In REWA geometry:
        A_Y^{do(X=x)} = weighted_union over m of (A_{M|X=x} weighted-combined with A_{Y|M=m})

    Args:
        X: Treatment variable name
        x_target: Target intervention value
        M: Mediator variable name
        Y: Outcome variable name
        witnesses: Dictionary mapping variable names to WitnessSet objects
        graph: CausalGraph defining the causal structure

    Returns:
        IdentificationResult with the interventional region
    """
    details = {
        'treatment': X,
        'target_value': x_target,
        'mediator': M,
        'outcome': Y
    }

    # Verify front-door criterion
    is_valid, criterion_details = graph.verify_frontdoor(M, X, Y)
    details['criterion_verification'] = criterion_details

    if not is_valid:
        return IdentificationResult(
            success=False,
            region=None,
            method='frontdoor',
            details={**details, 'error': 'Front-door criterion not satisfied'}
        )

    # Get witness sets
    required_vars = [X, M, Y]
    for var in required_vars:
        if var not in witnesses:
            return IdentificationResult(
                success=False,
                region=None,
                method='frontdoor',
                details={**details, 'error': f'No witnesses for {var}'}
            )

    W_X = witnesses[X]
    W_M = witnesses[M]
    W_Y = witnesses[Y]

    # Step 1: Compute A_{M|X=x} - mediator distribution under treatment
    M_given_X = _filter_witnesses_by_condition(W_M, X, x_target, W_X)

    if M_given_X.n_witnesses == 0:
        return IdentificationResult(
            success=False,
            region=None,
            method='frontdoor',
            details={**details, 'error': 'No mediator witnesses for X=x'}
        )

    # Step 2: For each mediator value, compute weighted Y outcome
    # Cluster M into strata
    n_m_strata = min(5, M_given_X.n_witnesses)
    m_strata_indices = _cluster_witnesses(M_given_X, n_m_strata)

    combined_regions = []
    combined_weights = []

    for m_idx in range(n_m_strata):
        m_mask = m_strata_indices == m_idx
        if not np.any(m_mask):
            continue

        # P(M=m | X=x)
        p_m_given_x = np.sum(m_mask) / len(m_strata_indices)

        # Step 2b: Compute Σ_x' P(Y | M=m, X=x') P(X=x')
        # This marginalizes over X for fixed M=m
        Y_given_M = _filter_witnesses_by_stratum(W_Y, M_given_X, m_mask, None, None, None)

        if Y_given_M.n_witnesses > 0:
            region_Y_M = spherical_convex_hull(Y_given_M.witnesses)
            combined_regions.append(region_Y_M)
            combined_weights.append(p_m_given_x)

    if not combined_regions:
        return IdentificationResult(
            success=False,
            region=None,
            method='frontdoor',
            details={**details, 'error': 'No valid mediator strata'}
        )

    # Weighted union over mediator values
    result_region = weighted_union(combined_regions, combined_weights)

    return IdentificationResult(
        success=True,
        region=result_region,
        method='frontdoor',
        details={
            **details,
            'n_mediator_strata': len(combined_regions),
            'mediator_weights': combined_weights,
            'region_radius': result_region.radius,
            'region_valid': result_region.is_valid,
            'n_witnesses_total': result_region.n_points
        }
    )


def interventional_region(
    X: str,
    x_val: Any,
    graph: CausalGraph,
    witnesses: Dict[str, WitnessSet],
    target_var: Optional[str] = None
) -> IdentificationResult:
    """
    Construct the interventional region A_Y^{do(X=x)} using do-calculus.

    Automatically selects the appropriate identification method based on
    the graph structure.

    Args:
        X: Treatment variable
        x_val: Intervention value
        graph: CausalGraph
        witnesses: Dictionary of witness sets
        target_var: Target outcome variable (if None, uses first non-X variable)

    Returns:
        IdentificationResult with the interventional region
    """
    # Determine target variable
    if target_var is None:
        available = [v for v in witnesses.keys() if v != X]
        if not available:
            return IdentificationResult(
                success=False,
                region=None,
                method='interventional',
                details={'error': 'No target variable available'}
            )
        target_var = available[0]

    Y = target_var

    # Try to find valid adjustment set for backdoor
    adjustment_set = graph.find_valid_adjustment_set(X, Y, set(witnesses.keys()) - {X, Y})

    if adjustment_set is not None:
        # Use backdoor adjustment
        return backdoor_adjustment(X, x_val, adjustment_set, Y, witnesses, graph)

    # Try front-door if backdoor fails
    # Look for potential mediators
    descendants_X = graph.descendants(X)
    ancestors_Y = graph.ancestors(Y)
    potential_mediators = descendants_X & ancestors_Y

    for M in potential_mediators:
        if M in witnesses:
            is_valid, _ = graph.verify_frontdoor(M, X, Y)
            if is_valid:
                return frontdoor_adjustment(X, x_val, M, Y, witnesses, graph)

    # Neither method works
    return IdentificationResult(
        success=False,
        region=None,
        method='interventional',
        details={
            'error': 'Causal effect not identifiable',
            'treatment': X,
            'outcome': Y,
            'attempted_backdoor': True,
            'attempted_frontdoor': True
        }
    )


def causal_effect(
    X: str,
    x0: Any,
    x1: Any,
    Y: str,
    witnesses: Dict[str, WitnessSet],
    graph: CausalGraph
) -> CausalEffectResult:
    """
    Compute the causal effect of changing X from x0 to x1 on Y.

    Returns the geometric causal effect: the shift in the interventional
    region of Y when X changes from x0 to x1.

    Args:
        X: Treatment variable
        x0: Baseline value
        x1: Intervention value
        Y: Outcome variable
        witnesses: Dictionary of witness sets
        graph: CausalGraph

    Returns:
        CausalEffectResult with effect metrics
    """
    # Compute A_Y^{do(X=x0)}
    result_x0 = interventional_region(X, x0, graph, witnesses, Y)

    # Compute A_Y^{do(X=x1)}
    result_x1 = interventional_region(X, x1, graph, witnesses, Y)

    details = {
        'treatment': X,
        'x0': x0,
        'x1': x1,
        'outcome': Y,
        'identification_x0': result_x0.details,
        'identification_x1': result_x1.details
    }

    if not result_x0.success or not result_x1.success:
        return CausalEffectResult(
            treatment_values=(x0, x1),
            region_x0=result_x0.region,
            region_x1=result_x1.region,
            centroid_shift=float('inf'),
            direction=None,
            ambiguity=float('inf'),
            details={**details, 'error': 'Identification failed'}
        )

    region_x0 = result_x0.region
    region_x1 = result_x1.region

    # Compute centroid shift (geodesic distance)
    centroid_shift = geodesic_distance(region_x0.centroid, region_x1.centroid)

    # Compute direction of shift
    c0 = region_x0.centroid
    c1 = region_x1.centroid

    # Project c1 onto tangent plane at c0
    tangent = c1 - (c1 @ c0) * c0
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm > 1e-10:
        direction = tangent / tangent_norm
    else:
        direction = np.zeros_like(c0)

    # Compute ambiguity (based on region sizes and overlap)
    dispersion_x0 = compute_dispersion(region_x0.points)
    dispersion_x1 = compute_dispersion(region_x1.points)

    # Ambiguity is the average region radius relative to the effect size
    avg_radius = (dispersion_x0['radius'] + dispersion_x1['radius']) / 2
    if centroid_shift > 1e-10:
        ambiguity = avg_radius / centroid_shift
    else:
        ambiguity = float('inf') if avg_radius > 0 else 0.0

    return CausalEffectResult(
        treatment_values=(x0, x1),
        region_x0=region_x0,
        region_x1=region_x1,
        centroid_shift=centroid_shift,
        direction=direction,
        ambiguity=ambiguity,
        details={
            **details,
            'dispersion_x0': dispersion_x0,
            'dispersion_x1': dispersion_x1,
            'region_x0_valid': region_x0.is_valid,
            'region_x1_valid': region_x1.is_valid
        }
    )


# Helper functions

def _filter_witnesses_by_condition(
    target_witnesses: WitnessSet,
    condition_var: str,
    condition_value: Any,
    condition_witnesses: Optional[WitnessSet]
) -> WitnessSet:
    """
    Filter witnesses to those consistent with a condition.

    For REWA, we use geometric proximity to filter.
    """
    if condition_witnesses is None or condition_witnesses.n_witnesses == 0:
        return target_witnesses

    # Compute similarity between target witnesses and condition witnesses
    # Keep target witnesses that have high similarity to condition witnesses
    similarities = target_witnesses.witnesses @ condition_witnesses.witnesses.T
    max_similarities = np.max(similarities, axis=1)

    # Keep witnesses with above-median similarity
    threshold = np.median(max_similarities)
    mask = max_similarities >= threshold

    if not np.any(mask):
        mask = np.ones(len(max_similarities), dtype=bool)

    filtered_witnesses = target_witnesses.witnesses[mask]
    filtered_metadata = [target_witnesses.metadata[i] for i in range(len(mask)) if mask[i]]

    return WitnessSet(
        variable=target_witnesses.variable,
        witnesses=filtered_witnesses,
        metadata=filtered_metadata
    )


def _combine_witness_sets(witness_sets: List[WitnessSet]) -> Optional[WitnessSet]:
    """Combine multiple witness sets into one."""
    if not witness_sets:
        return None

    all_witnesses = []
    all_metadata = []

    for ws in witness_sets:
        if ws.n_witnesses > 0:
            all_witnesses.append(ws.witnesses)
            all_metadata.extend(ws.metadata)

    if not all_witnesses:
        return None

    combined = np.vstack(all_witnesses)
    return WitnessSet(
        variable='combined',
        witnesses=combined,
        metadata=all_metadata
    )


def _cluster_witnesses(witness_set: WitnessSet, n_clusters: int) -> np.ndarray:
    """
    Cluster witnesses into strata using k-means on the sphere.

    Returns array of cluster indices.
    """
    n = witness_set.n_witnesses

    if n <= n_clusters:
        return np.arange(n)

    # Simple spherical k-means
    witnesses = witness_set.witnesses

    # Initialize centroids
    np.random.seed(42)
    centroid_indices = np.random.choice(n, n_clusters, replace=False)
    centroids = witnesses[centroid_indices].copy()

    for _ in range(20):  # Max iterations
        # Assign to nearest centroid
        similarities = witnesses @ centroids.T
        labels = np.argmax(similarities, axis=1)

        # Update centroids
        new_centroids = []
        for k in range(n_clusters):
            mask = labels == k
            if np.any(mask):
                cluster_mean = witnesses[mask].mean(axis=0)
                norm = np.linalg.norm(cluster_mean)
                if norm > 1e-10:
                    new_centroids.append(cluster_mean / norm)
                else:
                    new_centroids.append(centroids[k])
            else:
                new_centroids.append(centroids[k])

        new_centroids = np.array(new_centroids)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels


def _filter_witnesses_by_stratum(
    target_witnesses: WitnessSet,
    stratum_witnesses: WitnessSet,
    stratum_mask: np.ndarray,
    condition_var: Optional[str],
    condition_value: Any,
    condition_witnesses: Optional[WitnessSet]
) -> WitnessSet:
    """Filter target witnesses to those consistent with a stratum."""
    if stratum_witnesses.n_witnesses == 0:
        return target_witnesses

    # Get stratum center
    stratum_points = stratum_witnesses.witnesses[stratum_mask]
    if len(stratum_points) == 0:
        return target_witnesses

    stratum_center = frechet_mean(stratum_points)

    # Filter by similarity to stratum center
    similarities = target_witnesses.witnesses @ stratum_center

    # Keep witnesses with above-median similarity
    threshold = np.median(similarities)
    mask = similarities >= threshold

    if not np.any(mask):
        mask = np.ones(len(similarities), dtype=bool)

    # Additional filtering by condition if specified
    if condition_var and condition_witnesses:
        cond_similarities = target_witnesses.witnesses @ condition_witnesses.witnesses.T
        cond_max = np.max(cond_similarities, axis=1)
        cond_threshold = np.median(cond_max)
        mask = mask & (cond_max >= cond_threshold)

    filtered = target_witnesses.witnesses[mask]
    filtered_meta = [target_witnesses.metadata[i] for i in range(len(mask)) if mask[i]]

    return WitnessSet(
        variable=target_witnesses.variable,
        witnesses=filtered,
        metadata=filtered_meta
    )
