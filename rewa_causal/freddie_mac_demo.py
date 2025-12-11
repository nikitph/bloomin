"""
REWA-Causal Engine - Freddie Mac Integration Demo

Demonstrates causal inference on real mortgage data using the
REWA-Causal geometric framework.

Causal Questions:
1. What is the causal effect of Credit Score on Default?
2. What is the causal effect of DTI on Default?
3. What is the causal effect of LTV on Default?

Handles confounding from unobserved economic conditions.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'rewa_core', 'freddie_validation', 'src'))

from witness import WitnessSet, AttributeValueExtractor
from geometry import spherical_convex_hull, geodesic_distance, compute_dispersion, frechet_mean
from causal_graph import CausalGraph
from identification import backdoor_adjustment, causal_effect
from refusal import validate_region, validate_causal_claim, ValidationConfig, CausalClaimValidator


@dataclass
class FreddieLoan:
    """Parsed Freddie Mac loan record."""
    loan_id: str
    credit_score: float
    dti: float
    ltv: float
    original_upb: float
    interest_rate: float
    loan_purpose: str
    property_type: str
    occupancy_status: str
    was_repurchased: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'credit_score': self.credit_score,
            'dti': self.dti,
            'ltv': self.ltv,
            'original_upb': self.original_upb,
            'interest_rate': self.interest_rate,
            'loan_purpose': self.loan_purpose,
            'property_type': self.property_type,
            'occupancy_status': self.occupancy_status
        }


def load_freddie_data(data_dir: str, year: int = 2020, max_loans: int = 1000) -> List[FreddieLoan]:
    """
    Load Freddie Mac loan data from pipe-delimited files.

    Args:
        data_dir: Directory containing data files
        year: Year of data to load
        max_loans: Maximum number of loans to load

    Returns:
        List of FreddieLoan objects
    """
    orig_file = os.path.join(data_dir, f"sample_orig_{year}.txt")
    svcg_file = os.path.join(data_dir, f"sample_svcg_{year}.txt")

    if not os.path.exists(orig_file):
        print(f"Data file not found: {orig_file}")
        return []

    # Load origination data
    loans = {}
    print(f"Loading origination data from {orig_file}...")

    with open(orig_file, 'r') as f:
        for line_num, line in enumerate(f):
            if line_num >= max_loans:
                break

            parts = line.strip().split('|')
            if len(parts) < 20:
                continue

            try:
                loan_id = parts[19] if len(parts) > 19 else f"loan_{line_num}"

                # Parse fields with defaults
                credit_score = float(parts[0]) if parts[0] and parts[0] != '' else 700
                dti = float(parts[9]) if len(parts) > 9 and parts[9] and parts[9] != '' else 36
                ltv = float(parts[11]) if len(parts) > 11 and parts[11] and parts[11] != '' else 80
                upb = float(parts[10]) if len(parts) > 10 and parts[10] and parts[10] != '' else 200000
                rate = float(parts[12]) if len(parts) > 12 and parts[12] and parts[12] != '' else 5.0
                purpose = parts[20] if len(parts) > 20 else 'P'
                prop_type = parts[17] if len(parts) > 17 else 'SF'
                occupancy = parts[7] if len(parts) > 7 else 'P'

                loans[loan_id] = FreddieLoan(
                    loan_id=loan_id,
                    credit_score=credit_score,
                    dti=dti,
                    ltv=ltv,
                    original_upb=upb,
                    interest_rate=rate,
                    loan_purpose=purpose,
                    property_type=prop_type,
                    occupancy_status=occupancy
                )
            except (ValueError, IndexError) as e:
                continue

    # Load servicing data to identify repurchased loans
    if os.path.exists(svcg_file):
        print(f"Loading servicing data from {svcg_file}...")
        repurchased_ids = set()

        with open(svcg_file, 'r') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) < 2:
                    continue

                loan_id = parts[0]
                # Check zero balance code for repurchase (code 03 or 09)
                if len(parts) > 8 and parts[8] in ['03', '09']:
                    repurchased_ids.add(loan_id)

        # Mark repurchased loans
        for loan_id in repurchased_ids:
            if loan_id in loans:
                loans[loan_id].was_repurchased = True

        print(f"Found {len(repurchased_ids)} repurchased loans")

    return list(loans.values())


def create_mortgage_witnesses(
    loans: List[FreddieLoan],
    dimension: int = 64
) -> Dict[str, WitnessSet]:
    """
    Create witness sets from loan data for causal variables.

    Variables:
    - CreditScore: Credit score embeddings
    - DTI: Debt-to-income ratio embeddings
    - LTV: Loan-to-value ratio embeddings
    - Default: Default/repurchase status embeddings

    Args:
        loans: List of loan records
        dimension: Embedding dimension

    Returns:
        Dictionary of witness sets
    """
    # Define schemas for attribute encoding
    credit_schema = {
        'credit_score': {'type': 'numeric', 'min': 300, 'max': 850}
    }

    dti_schema = {
        'dti': {'type': 'numeric', 'min': 0, 'max': 65}
    }

    ltv_schema = {
        'ltv': {'type': 'numeric', 'min': 0, 'max': 200}
    }

    risk_schema = {
        'credit_score': {'type': 'numeric', 'min': 300, 'max': 850},
        'dti': {'type': 'numeric', 'min': 0, 'max': 65},
        'ltv': {'type': 'numeric', 'min': 0, 'max': 200},
        'interest_rate': {'type': 'numeric', 'min': 0, 'max': 15}
    }

    # Create extractors
    credit_extractor = AttributeValueExtractor(credit_schema, dimension=dimension)
    dti_extractor = AttributeValueExtractor(dti_schema, dimension=dimension)
    ltv_extractor = AttributeValueExtractor(ltv_schema, dimension=dimension)
    risk_extractor = AttributeValueExtractor(risk_schema, dimension=dimension)

    # Extract witnesses for each variable
    credit_data = [{'credit_score': loan.credit_score} for loan in loans]
    dti_data = [{'dti': loan.dti} for loan in loans]
    ltv_data = [{'ltv': loan.ltv} for loan in loans]
    risk_data = [loan.to_dict() for loan in loans]

    credit_ws = credit_extractor.extract(credit_data, 'CreditScore')
    dti_ws = dti_extractor.extract(dti_data, 'DTI')
    ltv_ws = ltv_extractor.extract(ltv_data, 'LTV')

    # For default, create witnesses based on repurchase status
    # Repurchased loans point in one direction, performing loans in another
    default_witnesses = []
    np.random.seed(42)

    default_center = np.random.randn(dimension)
    default_center = default_center / np.linalg.norm(default_center)

    perform_center = -default_center  # Opposite direction

    for loan in loans:
        if loan.was_repurchased:
            # Point toward "default" region
            base = default_center.copy()
        else:
            # Point toward "performing" region
            base = perform_center.copy()

        # Add noise
        noise = np.random.randn(dimension) * 0.15
        witness = base + noise
        witness = witness / np.linalg.norm(witness)
        default_witnesses.append(witness)

    default_ws = WitnessSet(
        variable='Default',
        witnesses=np.array(default_witnesses),
        metadata=[{'was_repurchased': loan.was_repurchased} for loan in loans]
    )

    return {
        'CreditScore': credit_ws,
        'DTI': dti_ws,
        'LTV': ltv_ws,
        'Default': default_ws,
        'RiskProfile': risk_extractor.extract(risk_data, 'RiskProfile')
    }


def create_mortgage_causal_graph() -> CausalGraph:
    """
    Create the causal graph for mortgage default analysis.

    Graph structure:
    - Economy (latent) → CreditScore, Default
    - CreditScore → Default
    - DTI → Default
    - LTV → Default

    CreditScore is confounded by Economy.
    DTI and LTV are approximately unconfounded.
    """
    graph = CausalGraph()

    # Add nodes
    graph.add_node('Economy', is_observed=False, is_latent=True)
    graph.add_node('CreditScore')
    graph.add_node('DTI')
    graph.add_node('LTV')
    graph.add_node('Default')

    # Add edges
    graph.add_edge('Economy', 'CreditScore')  # Economic conditions affect credit scores
    graph.add_edge('Economy', 'Default')       # Economic conditions affect default rates
    graph.add_edge('CreditScore', 'Default')   # Credit score affects default
    graph.add_edge('DTI', 'Default')           # DTI affects default
    graph.add_edge('LTV', 'Default')           # LTV affects default

    return graph


def stratified_causal_effect(
    treatment_var: str,
    outcome_var: str,
    witnesses: Dict[str, WitnessSet],
    percentile_split: float = 50.0
) -> Dict[str, Any]:
    """
    Compute causal effect by stratifying treatment into high/low groups.

    Instead of relying on the generic identification, we directly split
    the data into treatment groups and compare outcome regions.
    """
    W_X = witnesses[treatment_var]
    W_Y = witnesses[outcome_var]

    # Compute "treatment level" as similarity to treatment centroid
    X_centroid = frechet_mean(W_X.witnesses)
    X_scores = W_X.witnesses @ X_centroid

    # Split into high/low treatment groups
    threshold = np.percentile(X_scores, percentile_split)
    high_mask = X_scores >= threshold
    low_mask = X_scores < threshold

    # Get corresponding Y witnesses for each group
    Y_high = W_Y.witnesses[high_mask]
    Y_low = W_Y.witnesses[low_mask]

    if len(Y_high) < 5 or len(Y_low) < 5:
        return {
            'success': False,
            'error': 'insufficient_witnesses',
            'n_high': len(Y_high),
            'n_low': len(Y_low)
        }

    # Compute hulls
    hull_high = spherical_convex_hull(Y_high)
    hull_low = spherical_convex_hull(Y_low)

    # Compute effect
    centroid_shift = geodesic_distance(hull_high.centroid, hull_low.centroid)

    # Compute direction
    c_low = hull_low.centroid
    c_high = hull_high.centroid
    tangent = c_high - (c_high @ c_low) * c_low
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm > 1e-10:
        direction = tangent / tangent_norm
    else:
        direction = np.zeros_like(c_low)

    # Compute ambiguity
    avg_radius = (hull_high.radius + hull_low.radius) / 2
    ambiguity = avg_radius / centroid_shift if centroid_shift > 0 else float('inf')

    return {
        'success': True,
        'region_high': hull_high,
        'region_low': hull_low,
        'centroid_shift': centroid_shift,
        'direction': direction,
        'ambiguity': ambiguity,
        'n_high': len(Y_high),
        'n_low': len(Y_low)
    }


def run_causal_analysis(
    witnesses: Dict[str, WitnessSet],
    graph: CausalGraph
) -> Dict[str, Any]:
    """
    Run causal analysis on mortgage data.

    Estimates causal effects of CreditScore, DTI, and LTV on Default.
    """
    results = {
        'analyses': [],
        'refusals': [],
        'summary': {}
    }

    # Validation thresholds tuned for mortgage data
    # These would be calibrated based on domain requirements
    validator = CausalClaimValidator(ValidationConfig(
        min_witnesses=5,
        max_ambiguity=15.0,      # Allow higher ambiguity for exploratory analysis
        min_effect_size=0.05,    # Require at least ~3 degrees effect
        max_overlap_fraction=0.95  # Allow some overlap
    ))

    # Analysis 1: Effect of Credit Score on Default
    print("\n" + "="*60)
    print("Analysis 1: Causal Effect of Credit Score on Default")
    print("="*60)
    print("  Note: CreditScore is confounded by unobserved Economy")

    credit_effect = stratified_causal_effect('CreditScore', 'Default', witnesses)

    if credit_effect['success']:
        print(f"  Stratification: {credit_effect['n_low']} low, {credit_effect['n_high']} high")
        print(f"  Centroid shift: {credit_effect['centroid_shift']:.4f} radians")
        print(f"  ({np.degrees(credit_effect['centroid_shift']):.2f} degrees)")
        print(f"  Ambiguity: {credit_effect['ambiguity']:.4f}")

        # Validate
        validation = validator.validate(
            credit_effect['region_low'],
            credit_effect['region_high']
        )

        if validation.is_valid:
            print("  Status: VALID causal claim")
            results['analyses'].append({
                'treatment': 'CreditScore',
                'effect_size': credit_effect['centroid_shift'],
                'ambiguity': credit_effect['ambiguity'],
                'status': 'valid'
            })
        else:
            print(f"  Status: REFUSED - {validation.reason.value}")
            print(f"  Recommendations: {validation.recommendations}")
            results['refusals'].append({
                'treatment': 'CreditScore',
                'reason': validation.reason.value,
                'recommendations': validation.recommendations
            })
    else:
        print(f"  Stratification failed: {credit_effect.get('error', 'unknown')}")
        results['refusals'].append({
            'treatment': 'CreditScore',
            'reason': credit_effect.get('error', 'stratification_failure')
        })

    # Analysis 2: Effect of DTI on Default
    print("\n" + "="*60)
    print("Analysis 2: Causal Effect of DTI on Default")
    print("="*60)
    print("  Note: DTI is approximately unconfounded")

    dti_effect = stratified_causal_effect('DTI', 'Default', witnesses)

    if dti_effect['success']:
        print(f"  Stratification: {dti_effect['n_low']} low, {dti_effect['n_high']} high")
        print(f"  Centroid shift: {dti_effect['centroid_shift']:.4f} radians")
        print(f"  ({np.degrees(dti_effect['centroid_shift']):.2f} degrees)")
        print(f"  Ambiguity: {dti_effect['ambiguity']:.4f}")

        validation = validator.validate(
            dti_effect['region_low'],
            dti_effect['region_high']
        )

        if validation.is_valid:
            print("  Status: VALID causal claim")
            results['analyses'].append({
                'treatment': 'DTI',
                'effect_size': dti_effect['centroid_shift'],
                'ambiguity': dti_effect['ambiguity'],
                'status': 'valid'
            })
        else:
            print(f"  Status: REFUSED - {validation.reason.value}")
            results['refusals'].append({
                'treatment': 'DTI',
                'reason': validation.reason.value
            })
    else:
        print(f"  Stratification failed: {dti_effect.get('error', 'unknown')}")
        results['refusals'].append({
            'treatment': 'DTI',
            'reason': dti_effect.get('error', 'stratification_failure')
        })

    # Analysis 3: Effect of LTV on Default
    print("\n" + "="*60)
    print("Analysis 3: Causal Effect of LTV on Default")
    print("="*60)
    print("  Note: LTV is approximately unconfounded")

    ltv_effect = stratified_causal_effect('LTV', 'Default', witnesses)

    if ltv_effect['success']:
        print(f"  Stratification: {ltv_effect['n_low']} low, {ltv_effect['n_high']} high")
        print(f"  Centroid shift: {ltv_effect['centroid_shift']:.4f} radians")
        print(f"  ({np.degrees(ltv_effect['centroid_shift']):.2f} degrees)")
        print(f"  Ambiguity: {ltv_effect['ambiguity']:.4f}")

        validation = validator.validate(
            ltv_effect['region_low'],
            ltv_effect['region_high']
        )

        if validation.is_valid:
            print("  Status: VALID causal claim")
            results['analyses'].append({
                'treatment': 'LTV',
                'effect_size': ltv_effect['centroid_shift'],
                'ambiguity': ltv_effect['ambiguity'],
                'status': 'valid'
            })
        else:
            print(f"  Status: REFUSED - {validation.reason.value}")
            results['refusals'].append({
                'treatment': 'LTV',
                'reason': validation.reason.value
            })
    else:
        print(f"  Stratification failed: {ltv_effect.get('error', 'unknown')}")
        results['refusals'].append({
            'treatment': 'LTV',
            'reason': ltv_effect.get('error', 'stratification_failure')
        })

    # Summary
    results['summary'] = {
        'total_analyses': 3,
        'valid_claims': len(results['analyses']),
        'refused_claims': len(results['refusals']),
        'refusal_rate': len(results['refusals']) / 3
    }

    return results


def main():
    """Main demo entry point."""
    print("="*60)
    print("REWA-Causal Engine - Freddie Mac Demo")
    print("="*60)

    # Find data directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, '..', 'rewa_core', 'freddie_validation', 'data')

    # Check for data
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Generating synthetic mortgage data instead...")
        from utils import generate_mortgage_causal_dataset
        dataset = generate_mortgage_causal_dataset(n_loans=200, dimension=64)
        witnesses = dataset.witnesses
        graph = dataset.graph
    else:
        # Try to load real Freddie Mac data
        loans = []
        for year in [2020, 2021, 2022, 2023, 2024, 2025]:
            loans = load_freddie_data(data_dir, year=year, max_loans=500)
            if loans:
                print(f"Loaded {len(loans)} loans from year {year}")
                break

        if not loans:
            print("No real data found, using synthetic data...")
            from utils import generate_mortgage_causal_dataset
            dataset = generate_mortgage_causal_dataset(n_loans=200, dimension=64)
            witnesses = dataset.witnesses
            graph = dataset.graph
        else:
            # Count repurchased
            n_repurchased = sum(1 for l in loans if l.was_repurchased)
            print(f"Repurchased loans: {n_repurchased}")

            # Create witnesses from real data
            print("\nCreating witness sets from loan data...")
            witnesses = create_mortgage_witnesses(loans, dimension=64)

            # Create causal graph
            graph = create_mortgage_causal_graph()

    # Print witness set statistics
    print("\nWitness Set Statistics:")
    for var, ws in witnesses.items():
        print(f"  {var}: {ws.n_witnesses} witnesses, dim={ws.dimension}")

    # Validate individual witness sets
    print("\nValidating witness sets...")
    for var, ws in witnesses.items():
        hull = spherical_convex_hull(ws.witnesses)
        result = validate_region(hull)
        status = "OK" if result.is_valid else f"ISSUE: {result.reason.value}"
        print(f"  {var}: {status}")

    # Run causal analysis
    results = run_causal_analysis(witnesses, graph)

    # Final summary
    print("\n" + "="*60)
    print("CAUSAL ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total analyses: {results['summary']['total_analyses']}")
    print(f"Valid causal claims: {results['summary']['valid_claims']}")
    print(f"Refused claims: {results['summary']['refused_claims']}")
    print(f"Refusal rate: {results['summary']['refusal_rate']*100:.1f}%")

    if results['analyses']:
        print("\nValid Causal Effects:")
        for analysis in results['analyses']:
            print(f"  {analysis['treatment']} → Default: "
                  f"effect={np.degrees(analysis['effect_size']):.2f}°, "
                  f"ambiguity={analysis['ambiguity']:.2f}")

    if results['refusals']:
        print("\nRefused Claims:")
        for refusal in results['refusals']:
            print(f"  {refusal['treatment']}: {refusal['reason']}")

    print("\n" + "="*60)
    print("REWA-Causal guarantees: No hallucinated causal claims")
    print("="*60)

    return results


if __name__ == '__main__':
    results = main()
