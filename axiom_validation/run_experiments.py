"""
Main runner for Axiom Validation Experiments

Validates key axioms from "Axioms of Semantic Geometry" (Phadke, Dec 2025):
- Experiment 2: Axiom 3.2 (Antipodal Negation)
- Experiment 3: Axiom 4.1 (No Linear Aggregation)
- Experiment 4: Theorem 11 (Semantic Drift Under Iteration)
- Experiment 5: Energy Landscape Visualization
"""

import os
import sys
import json
from datetime import datetime
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def main():
    print("="*70)
    print("AXIOM VALIDATION EXPERIMENT SUITE")
    print("="*70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    results_summary = {}

    # Run Experiment 2: Antipodal Negation
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT 2: ANTIPODAL NEGATION (AXIOM 3.2)")
    print("="*70)

    try:
        from experiment2_antipodal_negation import main as exp2_main
        exp2_results, exp2_stats = exp2_main()
        results_summary['experiment2'] = {
            'axiom': '3.2 (Antipodal Negation)',
            'validated': exp2_stats['validated'],
            'mean_angle_pi': exp2_stats['mean_pi'],
            'success_rate': exp2_stats['success_rate'],
            'std_angle_pi': exp2_stats['std_pi'],
        }
        print(f"\n[COMPLETE] Experiment 2 completed. Validated: {exp2_stats['validated']}")
    except Exception as e:
        print(f"\n[ERROR] Experiment 2 failed: {e}")
        import traceback
        traceback.print_exc()
        results_summary['experiment2'] = {'error': str(e)}

    # Run Experiment 3: Linear Aggregation Failure
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT 3: LINEAR AGGREGATION FAILURE (AXIOM 4.1)")
    print("="*70)

    try:
        from experiment3_linear_aggregation import main as exp3_main
        exp3_results, exp3_stats = exp3_main()
        results_summary['experiment3'] = {
            'axiom': '4.1 (No Linear Aggregation)',
            'validated': exp3_stats['validated'],
            'high_energy_rate': exp3_stats['high_energy_rate'],
            'high_curvature_rate': exp3_stats['high_curvature_rate'],
            'outside_region_rate': exp3_stats['outside_rate'],
            'rewa_refusal_rate': exp3_stats['refusal_rate'],
        }
        print(f"\n[COMPLETE] Experiment 3 completed. Validated: {exp3_stats['validated']}")
    except Exception as e:
        print(f"\n[ERROR] Experiment 3 failed: {e}")
        import traceback
        traceback.print_exc()
        results_summary['experiment3'] = {'error': str(e)}

    # Run Experiment 4: Semantic Drift
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT 4: SEMANTIC DRIFT (THEOREM 11)")
    print("="*70)

    try:
        from experiment4_semantic_drift import main as exp4_main
        exp4_results, exp4_stats, exp4_comparison = exp4_main()
        results_summary['experiment4'] = {
            'theorem': '11 (Semantic Drift Under Iteration)',
            'validated': exp4_stats['validated'],
            'positive_drift_rate': exp4_stats['positive_drift_rate'],
            'mean_increase': exp4_stats['mean_increase'],
            'energy_increase_rate': exp4_stats['energy_increase_rate'],
            'mean_drift_rate': exp4_stats['mean_drift_rate'],
        }
        print(f"\n[COMPLETE] Experiment 4 completed. Validated: {exp4_stats['validated']}")
    except Exception as e:
        print(f"\n[ERROR] Experiment 4 failed: {e}")
        import traceback
        traceback.print_exc()
        results_summary['experiment4'] = {'error': str(e)}

    # Run Experiment 5: Energy Landscape
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT 5: ENERGY LANDSCAPE VISUALIZATION")
    print("="*70)

    try:
        from experiment5_energy_landscape import main as exp5_main
        exp5_analyses, exp5_stats, exp5_contradiction = exp5_main()
        results_summary['experiment5'] = {
            'objective': 'Energy Landscape Visualization',
            'validated': exp5_stats['energy_variance'] > 0.01,
            'admissible_rate': exp5_stats['admissible_rate'],
            'mean_energy': exp5_stats['mean_energy'],
            'energy_variance': exp5_stats['energy_variance'],
        }
        print(f"\n[COMPLETE] Experiment 5 completed. Validated: {results_summary['experiment5']['validated']}")
    except Exception as e:
        print(f"\n[ERROR] Experiment 5 failed: {e}")
        import traceback
        traceback.print_exc()
        results_summary['experiment5'] = {'error': str(e)}

    # Final Summary
    print("\n" + "="*70)
    print("FINAL EXPERIMENT SUMMARY")
    print("="*70)

    for exp_name, exp_data in results_summary.items():
        print(f"\n{exp_name.upper()}:")
        if 'error' in exp_data:
            print(f"  [ERROR] {exp_data['error']}")
        else:
            print(f"  Axiom: {exp_data['axiom']}")
            print(f"  Validated: {'YES' if exp_data['validated'] else 'NO'}")
            for key, value in exp_data.items():
                if key not in ['axiom', 'validated']:
                    if isinstance(value, float):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")

    # Save results to JSON
    results_file = os.path.join(script_dir, 'experiment_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2, cls=NumpyEncoder)
    print(f"\n[SAVED] Results saved to {results_file}")

    # Overall validation
    all_validated = all(
        exp_data.get('validated', False)
        for exp_data in results_summary.values()
        if 'error' not in exp_data
    )

    print("\n" + "="*70)
    if all_validated:
        print("OVERALL: ALL AXIOMS VALIDATED")
    else:
        print("OVERALL: SOME AXIOMS NOT FULLY VALIDATED")
    print("="*70)

    return results_summary

if __name__ == "__main__":
    main()
