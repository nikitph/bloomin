"""
Refusal & Safety Validation

Validates causal claims using REWA geometry rules.
Refuses to output causal claims when geometry is invalid.

Refusal conditions:
1. Hemisphere constraint violated
2. Region has too much ambiguity (radius > threshold)
3. Interventional regions significantly overlap
4. Identification conditions fail
5. Sample size too low for stable hull
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

try:
    from ..geometry import (
        SphericalConvexHull,
        satisfies_hemisphere_constraint,
        compute_dispersion,
        hull_overlap,
        geodesic_distance
    )
except ImportError:
    from geometry import (
        SphericalConvexHull,
        satisfies_hemisphere_constraint,
        compute_dispersion,
        hull_overlap,
        geodesic_distance
    )


class RefusalReason(Enum):
    """Enumeration of refusal reasons."""
    HEMISPHERE_VIOLATION = "hemisphere_violation"
    HIGH_AMBIGUITY = "high_ambiguity"
    REGION_OVERLAP = "region_overlap"
    IDENTIFICATION_FAILURE = "identification_failure"
    INSUFFICIENT_WITNESSES = "insufficient_witnesses"
    INVALID_GEOMETRY = "invalid_geometry"
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    UNSTABLE_HULL = "unstable_hull"


@dataclass
class ValidationConfig:
    """Configuration for validation thresholds."""
    # Hemisphere constraint threshold (minimum pairwise similarity)
    hemisphere_threshold: float = -0.3

    # Maximum allowed region radius (ambiguity threshold)
    max_ambiguity: float = 1.2  # ~70 degrees

    # Maximum allowed overlap between interventional regions
    max_overlap_fraction: float = 0.7

    # Minimum number of witnesses for stable hull
    min_witnesses: int = 3

    # Minimum effect size (geodesic distance) for causal claims
    min_effect_size: float = 0.1  # ~6 degrees

    # Maximum dispersion relative to effect size
    max_relative_dispersion: float = 2.0


@dataclass
class RefusalResult:
    """Result of validation check."""
    status: str  # "ok" or "refuse"
    reason: Optional[RefusalReason] = None
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status,
            'reason': self.reason.value if self.reason else None,
            'details': self.details,
            'recommendations': self.recommendations
        }


def validate_region(
    region: SphericalConvexHull,
    config: Optional[ValidationConfig] = None
) -> RefusalResult:
    """
    Validate a single spherical region for causal inference.

    Checks:
    1. Hemisphere constraint
    2. Region size (ambiguity)
    3. Number of witnesses
    4. Hull stability

    Args:
        region: SphericalConvexHull to validate
        config: ValidationConfig with thresholds

    Returns:
        RefusalResult indicating validity
    """
    config = config or ValidationConfig()
    details = {}
    recommendations = []

    # Check 1: Sufficient witnesses
    if region.n_points < config.min_witnesses:
        details['n_witnesses'] = region.n_points
        details['min_required'] = config.min_witnesses
        recommendations.append(f"Provide at least {config.min_witnesses} witnesses")
        return RefusalResult(
            status="refuse",
            reason=RefusalReason.INSUFFICIENT_WITNESSES,
            details=details,
            recommendations=recommendations
        )

    # Check 2: Hemisphere constraint
    is_valid, violations = satisfies_hemisphere_constraint(
        region.points,
        threshold=config.hemisphere_threshold
    )

    if not is_valid:
        details['violation_pairs'] = violations
        details['threshold'] = config.hemisphere_threshold
        recommendations.append("Evidence contains contradictions - review witness set")
        recommendations.append("Consider removing contradictory witnesses")
        return RefusalResult(
            status="refuse",
            reason=RefusalReason.HEMISPHERE_VIOLATION,
            details=details,
            recommendations=recommendations
        )

    # Check 3: Region ambiguity (radius)
    dispersion = compute_dispersion(region.points, region.centroid)
    details['dispersion'] = dispersion

    if dispersion['radius'] > config.max_ambiguity:
        details['radius'] = dispersion['radius']
        details['max_allowed'] = config.max_ambiguity
        recommendations.append("Region too ambiguous - gather more specific evidence")
        recommendations.append("Consider conditioning on additional variables")
        return RefusalResult(
            status="refuse",
            reason=RefusalReason.HIGH_AMBIGUITY,
            details=details,
            recommendations=recommendations
        )

    # Check 4: Hull stability (variance in witness positions)
    if dispersion['variance'] > 0.5:  # High variance indicates unstable hull
        details['variance'] = dispersion['variance']
        recommendations.append("Witness positions highly variable - hull may be unstable")

    # All checks passed
    return RefusalResult(
        status="ok",
        details={
            'n_witnesses': region.n_points,
            'radius': dispersion['radius'],
            'variance': dispersion['variance'],
            'is_hemisphere_valid': True
        }
    )


def validate_causal_claim(
    region_x0: SphericalConvexHull,
    region_x1: SphericalConvexHull,
    config: Optional[ValidationConfig] = None
) -> RefusalResult:
    """
    Validate a causal claim comparing two interventional regions.

    Checks:
    1. Both regions are valid
    2. Regions don't overlap excessively
    3. Effect size is distinguishable from noise
    4. Relative dispersion is acceptable

    Args:
        region_x0: Region under treatment X=x0
        region_x1: Region under treatment X=x1
        config: ValidationConfig with thresholds

    Returns:
        RefusalResult indicating validity of causal claim
    """
    config = config or ValidationConfig()
    details = {}
    recommendations = []

    # Validate individual regions
    result_x0 = validate_region(region_x0, config)
    result_x1 = validate_region(region_x1, config)

    details['region_x0_validation'] = result_x0.to_dict()
    details['region_x1_validation'] = result_x1.to_dict()

    if not result_x0.is_valid:
        details['failed_region'] = 'x0'
        recommendations.extend(result_x0.recommendations)
        return RefusalResult(
            status="refuse",
            reason=result_x0.reason,
            details=details,
            recommendations=recommendations
        )

    if not result_x1.is_valid:
        details['failed_region'] = 'x1'
        recommendations.extend(result_x1.recommendations)
        return RefusalResult(
            status="refuse",
            reason=result_x1.reason,
            details=details,
            recommendations=recommendations
        )

    # Check effect size
    centroid_distance = geodesic_distance(region_x0.centroid, region_x1.centroid)
    details['centroid_distance'] = centroid_distance

    if centroid_distance < config.min_effect_size:
        details['min_effect_size'] = config.min_effect_size
        recommendations.append("Effect size too small to distinguish from noise")
        recommendations.append("Consider if causal effect is practically significant")
        return RefusalResult(
            status="refuse",
            reason=RefusalReason.INVALID_GEOMETRY,
            details=details,
            recommendations=recommendations
        )

    # Check region overlap
    overlap = hull_overlap(region_x0, region_x1)
    details['overlap'] = overlap

    if overlap['overlap_fraction'] > config.max_overlap_fraction:
        details['max_overlap'] = config.max_overlap_fraction
        recommendations.append("Interventional regions overlap significantly")
        recommendations.append("Effect may not be reliably distinguishable")
        return RefusalResult(
            status="refuse",
            reason=RefusalReason.REGION_OVERLAP,
            details=details,
            recommendations=recommendations
        )

    # Check relative dispersion
    avg_radius = (region_x0.radius + region_x1.radius) / 2
    relative_dispersion = avg_radius / centroid_distance if centroid_distance > 0 else float('inf')
    details['relative_dispersion'] = relative_dispersion

    if relative_dispersion > config.max_relative_dispersion:
        details['max_relative_dispersion'] = config.max_relative_dispersion
        recommendations.append("Region uncertainty too large relative to effect size")
        recommendations.append("Gather more evidence to reduce uncertainty")
        return RefusalResult(
            status="refuse",
            reason=RefusalReason.HIGH_AMBIGUITY,
            details=details,
            recommendations=recommendations
        )

    # All checks passed
    return RefusalResult(
        status="ok",
        details={
            'centroid_distance': centroid_distance,
            'overlap_fraction': overlap['overlap_fraction'],
            'relative_dispersion': relative_dispersion,
            'region_x0_radius': region_x0.radius,
            'region_x1_radius': region_x1.radius,
            'effect_direction': 'distinguishable'
        }
    )


def validate_identification(
    method: str,
    identification_details: Dict[str, Any],
    config: Optional[ValidationConfig] = None
) -> RefusalResult:
    """
    Validate that causal identification was successful.

    Args:
        method: Identification method used ('backdoor', 'frontdoor', etc.)
        identification_details: Details from identification procedure
        config: ValidationConfig

    Returns:
        RefusalResult indicating validity
    """
    config = config or ValidationConfig()
    details = {'method': method, 'identification_details': identification_details}
    recommendations = []

    # Check if identification succeeded
    if identification_details.get('error'):
        details['error'] = identification_details['error']
        recommendations.append("Causal identification failed")
        recommendations.append("Check graph structure and available variables")
        return RefusalResult(
            status="refuse",
            reason=RefusalReason.IDENTIFICATION_FAILURE,
            details=details,
            recommendations=recommendations
        )

    # Check criterion verification
    criterion_verification = identification_details.get('criterion_verification', {})

    if criterion_verification.get('issues'):
        details['criterion_issues'] = criterion_verification['issues']
        recommendations.append("Identification criterion not satisfied")

        if method == 'backdoor':
            recommendations.append("Check adjustment set for descendants of treatment")
            recommendations.append("Ensure all backdoor paths are blocked")
        elif method == 'frontdoor':
            recommendations.append("Verify mediator intercepts all causal paths")
            recommendations.append("Check for unblocked confounding paths")

        return RefusalResult(
            status="refuse",
            reason=RefusalReason.IDENTIFICATION_FAILURE,
            details=details,
            recommendations=recommendations
        )

    # Check number of strata/witnesses used
    n_witnesses = identification_details.get('n_witnesses_total', 0)
    if n_witnesses < config.min_witnesses:
        details['n_witnesses'] = n_witnesses
        recommendations.append(f"Insufficient witnesses ({n_witnesses} < {config.min_witnesses})")
        return RefusalResult(
            status="refuse",
            reason=RefusalReason.INSUFFICIENT_WITNESSES,
            details=details,
            recommendations=recommendations
        )

    return RefusalResult(
        status="ok",
        details=details
    )


def create_refusal_report(
    results: List[RefusalResult],
    include_all_details: bool = False
) -> Dict[str, Any]:
    """
    Create a comprehensive refusal report from multiple validation results.

    Args:
        results: List of RefusalResult objects
        include_all_details: Whether to include full details

    Returns:
        Dictionary report
    """
    failed = [r for r in results if not r.is_valid]
    passed = [r for r in results if r.is_valid]

    report = {
        'overall_status': 'ok' if not failed else 'refuse',
        'checks_passed': len(passed),
        'checks_failed': len(failed),
        'failure_reasons': [r.reason.value for r in failed if r.reason],
        'all_recommendations': []
    }

    for r in results:
        report['all_recommendations'].extend(r.recommendations)

    report['all_recommendations'] = list(set(report['all_recommendations']))

    if include_all_details:
        report['detailed_results'] = [r.to_dict() for r in results]

    return report


class CausalClaimValidator:
    """
    High-level validator for causal claims.

    Provides a clean interface for validating end-to-end causal inference.
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self.validation_history: List[RefusalResult] = []

    def validate(
        self,
        region_x0: SphericalConvexHull,
        region_x1: SphericalConvexHull,
        identification_details: Optional[Dict[str, Any]] = None,
        method: str = 'backdoor'
    ) -> RefusalResult:
        """
        Perform full validation of a causal claim.

        Args:
            region_x0: Region under baseline treatment
            region_x1: Region under intervention
            identification_details: Details from identification procedure
            method: Identification method used

        Returns:
            RefusalResult with validation outcome
        """
        results = []

        # Validate identification if details provided
        if identification_details:
            id_result = validate_identification(method, identification_details, self.config)
            results.append(id_result)
            if not id_result.is_valid:
                self.validation_history.append(id_result)
                return id_result

        # Validate causal claim (regions)
        claim_result = validate_causal_claim(region_x0, region_x1, self.config)
        results.append(claim_result)

        # Store and return final result
        self.validation_history.extend(results)
        return claim_result

    def get_validation_report(self) -> Dict[str, Any]:
        """Get report of all validations performed."""
        return create_refusal_report(self.validation_history, include_all_details=True)

    def reset(self):
        """Clear validation history."""
        self.validation_history = []
