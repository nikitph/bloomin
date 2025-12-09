"""
Run Intrinsic Dimension Sweep
==============================

Main runner script for the intrinsic dimension sweep experiment.
"""

from experiment_intrinsic_dim_sweep import IntrinsicDimensionExperiment
from model_configs import MODELS_TO_TEST, QUICK_TEST_MODELS
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run intrinsic dimension sweep')
    parser.add_argument('--quick', action='store_true', help='Run quick test with subset of models')
    parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.90, 0.95, 0.99],
                       help='Variance thresholds to measure')
    parser.add_argument('--skip-download', action='store_true', 
                       help='Skip models that are not already cached (faster)')
    
    args = parser.parse_args()
    
    # Select models
    models = QUICK_TEST_MODELS if args.quick else MODELS_TO_TEST
    
    print(f"\n{'='*70}")
    print(f"INTRINSIC DIMENSION SWEEP")
    print(f"{'='*70}")
    print(f"Mode: {'QUICK TEST' if args.quick else 'FULL SWEEP'}")
    print(f"Models: {len(models)}")
    print(f"Variance thresholds: {args.thresholds}")
    print(f"Skip uncached: {args.skip_download}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*70}\n")
    
    # Run experiment
    experiment = IntrinsicDimensionExperiment(output_dir=args.output_dir)
    experiment.run_all_models(models, variance_thresholds=args.thresholds, skip_download=args.skip_download)
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
