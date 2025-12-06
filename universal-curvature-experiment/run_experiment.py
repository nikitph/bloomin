"""
Run Universal Curvature Experiment
===================================

Main execution script.
"""

import argparse
from experiment import CurvatureExperiment
from model_configs import MODELS_TO_TEST, QUICK_TEST_MODELS


def main():
    parser = argparse.ArgumentParser(description='Measure Gaussian curvature across embedding models')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick test on 2 models only')
    parser.add_argument('--model', type=str, default=None,
                       help='Run single model by name')
    parser.add_argument('--triangles', type=int, default=1000,
                       help='Number of triangles to sample per model')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize experiment
    exp = CurvatureExperiment(output_dir=args.output_dir)
    
    # Select models
    if args.model:
        # Single model
        if args.model not in MODELS_TO_TEST:
            print(f"Error: Model '{args.model}' not found")
            print(f"Available models: {list(MODELS_TO_TEST.keys())}")
            return
        
        models = {args.model: MODELS_TO_TEST[args.model]}
        print(f"Running single model: {args.model}")
        
    elif args.quick_test:
        # Quick test
        models = QUICK_TEST_MODELS
        print(f"Running quick test on {len(models)} models")
        
    else:
        # All models
        models = MODELS_TO_TEST
        print(f"Running full experiment on {len(models)} models")
    
    # Run experiment
    exp.run_all_models(models, n_triangles=args.triangles)
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
