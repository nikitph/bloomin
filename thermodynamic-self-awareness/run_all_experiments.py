#!/usr/bin/env python3
"""
Master Runner for All Thermodynamic Self-Awareness Experiments

Runs all four experiments (A, B, C, D) sequentially and generates
a comprehensive summary report.
"""

import sys
import os
import time
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments import experiment_a_self_discovery
from experiments import experiment_b_mirror
from experiments import experiment_c_tom
from experiments import experiment_d_freewill


def run_all_experiments(seed: int = 42, output_dir: str = 'logs'):
    """
    Run all four experiments sequentially.
    
    Args:
        seed: Random seed for reproducibility
        output_dir: Directory to save results
    """
    print("\n" + "="*70)
    print("THERMODYNAMIC SELF-AWARENESS - FULL EXPERIMENTAL SUITE")
    print("="*70)
    print(f"Seed: {seed}")
    print(f"Output Directory: {output_dir}")
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    results_summary = {
        'seed': seed,
        'start_time': datetime.now().isoformat(),
        'experiments': {}
    }
    
    # Experiment A: Self-Discovery
    print("\n" + "▶"*35)
    print("EXPERIMENT A: SELF-DISCOVERY")
    print("▶"*35 + "\n")
    start = time.time()
    try:
        results_a = experiment_a_self_discovery.run_experiment(
            seed=seed,
            enable_topos=True,
            enable_ricci=True,
            output_dir=output_dir
        )
        duration_a = time.time() - start
        results_summary['experiments']['A_self_discovery'] = {
            'status': 'success',
            'duration_seconds': duration_a,
            'rule_discovered': results_a.get('rule_discovery_epoch') is not None,
            'discovery_epoch': results_a.get('rule_discovery_epoch'),
            'mean_free_energy': float(sum(results_a['free_energy']) / len(results_a['free_energy']))
        }
        print(f"\n✓ Experiment A completed in {duration_a:.1f}s")
    except Exception as e:
        print(f"\n✗ Experiment A failed: {e}")
        results_summary['experiments']['A_self_discovery'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    # Experiment B: Mirror Test
    print("\n" + "▶"*35)
    print("EXPERIMENT B: MIRROR TEST")
    print("▶"*35 + "\n")
    start = time.time()
    try:
        results_b = experiment_b_mirror.run_experiment(
            seed=seed,
            output_dir=output_dir
        )
        duration_b = time.time() - start
        accuracy_b = results_b['accuracies'][0] if results_b['accuracies'] else 0
        results_summary['experiments']['B_mirror_test'] = {
            'status': 'success',
            'duration_seconds': duration_b,
            'accuracy': float(accuracy_b),
            'success': accuracy_b >= 0.9
        }
        print(f"\n✓ Experiment B completed in {duration_b:.1f}s")
    except Exception as e:
        print(f"\n✗ Experiment B failed: {e}")
        results_summary['experiments']['B_mirror_test'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    # Experiment C: Theory-of-Mind
    print("\n" + "▶"*35)
    print("EXPERIMENT C: THEORY-OF-MIND")
    print("▶"*35 + "\n")
    start = time.time()
    try:
        results_c = experiment_c_tom.run_experiment(
            seed=seed,
            output_dir=output_dir
        )
        duration_c = time.time() - start
        accuracy_c = results_c['accuracies'][0] if results_c['accuracies'] else 0
        results_summary['experiments']['C_theory_of_mind'] = {
            'status': 'success',
            'duration_seconds': duration_c,
            'accuracy': float(accuracy_c),
            'success': accuracy_c >= 0.85
        }
        print(f"\n✓ Experiment C completed in {duration_c:.1f}s")
    except Exception as e:
        print(f"\n✗ Experiment C failed: {e}")
        results_summary['experiments']['C_theory_of_mind'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    # Experiment D: Free-Will
    print("\n" + "▶"*35)
    print("EXPERIMENT D: FREE-WILL / NOVEL ACTIONS")
    print("▶"*35 + "\n")
    start = time.time()
    try:
        results_d = experiment_d_freewill.run_experiment(
            seed=seed,
            output_dir=output_dir
        )
        duration_d = time.time() - start
        validity_rate = results_d['validity_rate'][0] if results_d['validity_rate'] else 0
        results_summary['experiments']['D_free_will'] = {
            'status': 'success',
            'duration_seconds': duration_d,
            'validity_rate': float(validity_rate),
            'success': validity_rate >= 0.10
        }
        print(f"\n✓ Experiment D completed in {duration_d:.1f}s")
    except Exception as e:
        print(f"\n✗ Experiment D failed: {e}")
        results_summary['experiments']['D_free_will'] = {
            'status': 'failed',
            'error': str(e)
        }
    
    # Final summary
    results_summary['end_time'] = datetime.now().isoformat()
    results_summary['total_duration_seconds'] = sum(
        exp.get('duration_seconds', 0) 
        for exp in results_summary['experiments'].values()
    )
    
    # Save summary
    summary_path = os.path.join(output_dir, f'summary_all_experiments_seed{seed}.json')
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Print final summary
    print("\n" + "="*70)
    print("EXPERIMENTAL SUITE COMPLETE")
    print("="*70)
    print(f"\nTotal Duration: {results_summary['total_duration_seconds']:.1f}s")
    print(f"\nResults Summary:")
    
    for exp_name, exp_results in results_summary['experiments'].items():
        status_symbol = "✓" if exp_results['status'] == 'success' else "✗"
        print(f"\n  {status_symbol} {exp_name}:")
        if exp_results['status'] == 'success':
            if 'accuracy' in exp_results:
                print(f"      Accuracy: {exp_results['accuracy']:.2%}")
                print(f"      Success: {'Yes' if exp_results['success'] else 'No'}")
            elif 'validity_rate' in exp_results:
                print(f"      Validity Rate: {exp_results['validity_rate']:.2%}")
                print(f"      Success: {'Yes' if exp_results['success'] else 'No'}")
            elif 'rule_discovered' in exp_results:
                print(f"      Rule Discovered: {'Yes' if exp_results['rule_discovered'] else 'No'}")
                if exp_results['rule_discovered']:
                    print(f"      Discovery Epoch: {exp_results['discovery_epoch']}")
        else:
            print(f"      Error: {exp_results.get('error', 'Unknown')}")
    
    print(f"\n  Summary saved to: {summary_path}")
    print("\n" + "="*70 + "\n")
    
    return results_summary


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run all thermodynamic self-awareness experiments'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='logs', 
                       help='Output directory')
    
    args = parser.parse_args()
    
    summary = run_all_experiments(seed=args.seed, output_dir=args.output_dir)
    
    print("✓ All experiments complete!")
