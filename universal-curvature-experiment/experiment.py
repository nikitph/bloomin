"""
Universal Curvature Experiment
===============================

Main experiment class to measure K across all models.
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List

from curvature_measurement import measure_curvature
from model_loaders import load_model_and_get_embeddings
from corpus_loaders import load_corpus


class CurvatureExperiment:
    """
    Orchestrates curvature measurement across all models.
    """
    
    def __init__(self, output_dir: str = './results'):
        self.output_dir = output_dir
        self.results = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Results file
        self.results_file = os.path.join(output_dir, 'curvature_results.csv')
        
        # Load existing results if available
        if os.path.exists(self.results_file):
            print(f"Loading existing results from {self.results_file}")
            df = pd.read_csv(self.results_file)
            self.results = df.to_dict('records')
            print(f"  Loaded {len(self.results)} previous results")
    
    def run_single_model(
        self,
        model_name: str,
        model_config: Dict,
        n_triangles: int = 1000
    ) -> Dict:
        """
        Run experiment for a single model.
        
        Args:
            model_name: Name of the model
            model_config: Model configuration dict
            n_triangles: Number of triangles to sample
            
        Returns:
            Results dictionary
        """
        print(f"\n{'='*70}")
        print(f"Testing: {model_name}")
        print(f"{'='*70}")
        print(f"Model: {model_config['model']}")
        print(f"Year: {model_config['year']}, Dim: {model_config['dim']}, Arch: {model_config['arch']}")
        print(f"Domain: {model_config['domain']}, Samples: {model_config['n_samples']}")
        
        # Check if already computed
        for result in self.results:
            if result['model'] == model_name:
                print(f"\n⚠️  Model already computed, skipping...")
                print(f"   K = {result['K_mean']:.3f} ± {result['K_std']:.3f}")
                return result
        
        try:
            # Load corpus
            print(f"\nStep 1: Loading corpus...")
            corpus_type = model_config['corpus_type']
            n_samples = model_config['n_samples']
            texts = load_corpus(corpus_type, n_samples)
            print(f"  ✓ Loaded {len(texts)} samples")
            
            # Get embeddings
            print(f"\nStep 2: Extracting embeddings...")
            embeddings = load_model_and_get_embeddings(model_config, texts)
            print(f"  ✓ Embeddings shape: {embeddings.shape}")
            
            # Measure curvature
            print(f"\nStep 3: Measuring curvature...")
            curvature_results = measure_curvature(
                embeddings,
                n_triangles=n_triangles,
                random_seed=42,
                verbose=True
            )
            
            # Store results
            result_row = {
                'model': model_name,
                'model_path': model_config['model'],
                'year': model_config['year'],
                'dim': model_config['dim'],
                'arch': model_config['arch'],
                'K_mean': curvature_results['K_mean'],
                'K_std': curvature_results['K_std'],
                'K_median': curvature_results['K_median'],
                'K_min': curvature_results['K_min'],
                'K_max': curvature_results['K_max'],
                'n_samples': len(embeddings),
                'n_triangles': curvature_results['n_valid_triangles'],
                'domain': model_config['domain'],
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result_row)
            
            # Save immediately (checkpoint)
            self.save_results()
            
            # Print summary
            print(f"\n{'='*70}")
            print(f"RESULTS FOR {model_name}")
            print(f"{'='*70}")
            print(f"K = {curvature_results['K_mean']:.3f} ± {curvature_results['K_std']:.3f}")
            print(f"Median: {curvature_results['K_median']:.3f}")
            print(f"Range: [{curvature_results['K_min']:.3f}, {curvature_results['K_max']:.3f}]")
            print(f"Valid triangles: {curvature_results['n_valid_triangles']}/{n_triangles}")
            print(f"{'='*70}")
            
            return result_row
            
        except Exception as e:
            print(f"\n❌ Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Store error
            error_row = {
                'model': model_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            
            return error_row
    
    def run_all_models(self, models_dict: Dict, n_triangles: int = 1000):
        """
        Run experiment on all models.
        
        Args:
            models_dict: Dictionary of model configurations
            n_triangles: Number of triangles per model
        """
        print(f"\n{'#'*70}")
        print(f"UNIVERSAL CURVATURE MEASUREMENT EXPERIMENT")
        print(f"{'#'*70}")
        print(f"Models to test: {len(models_dict)}")
        print(f"Triangles per model: {n_triangles}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'#'*70}\n")
        
        for i, (model_name, config) in enumerate(models_dict.items(), 1):
            print(f"\n[{i}/{len(models_dict)}] Processing: {model_name}")
            
            self.run_single_model(model_name, config, n_triangles)
        
        # Final summary
        self.print_summary()
    
    def save_results(self):
        """Save results to CSV."""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(self.results_file, index=False)
        print(f"\n💾 Results saved to {self.results_file}")
    
    def print_summary(self):
        """Print summary statistics."""
        if not self.results:
            print("\nNo results to summarize.")
            return
        
        df = pd.DataFrame(self.results)
        
        # Filter out errors
        df = df[df['K_mean'].notna()]
        
        if len(df) == 0:
            print("\nNo valid results.")
            return
        
        print(f"\n{'='*70}")
        print(f"SUMMARY: Gaussian Curvature Across All Models")
        print(f"{'='*70}")
        
        print(f"\nOverall Statistics:")
        print(f"  Models tested: {len(df)}")
        print(f"  Mean K: {df['K_mean'].mean():.3f}")
        print(f"  Std K:  {df['K_mean'].std():.3f}")
        print(f"  95% CI: [{df['K_mean'].mean() - 1.96*df['K_mean'].std()/np.sqrt(len(df)):.3f}, "
              f"{df['K_mean'].mean() + 1.96*df['K_mean'].std()/np.sqrt(len(df)):.3f}]")
        
        print(f"\nBy Architecture:")
        arch_stats = df.groupby('arch')['K_mean'].agg(['mean', 'std', 'count'])
        print(arch_stats)
        
        print(f"\nBy Year:")
        year_stats = df.groupby('year')['K_mean'].agg(['mean', 'std', 'count'])
        print(year_stats)
        
        print(f"\nBy Dimension:")
        dim_stats = df.groupby('dim')['K_mean'].agg(['mean', 'std', 'count'])
        print(dim_stats)
        
        print(f"\nBy Domain:")
        domain_stats = df.groupby('domain')['K_mean'].agg(['mean', 'std', 'count'])
        print(domain_stats)
        
        print(f"\n{'='*70}")
        
        # Save summary
        summary = {
            'overall': {
                'n_models': len(df),
                'mean_K': float(df['K_mean'].mean()),
                'std_K': float(df['K_mean'].std()),
                'ci_lower': float(df['K_mean'].mean() - 1.96*df['K_mean'].std()/np.sqrt(len(df))),
                'ci_upper': float(df['K_mean'].mean() + 1.96*df['K_mean'].std()/np.sqrt(len(df))),
            },
            'by_architecture': arch_stats.to_dict(),
            'by_year': year_stats.to_dict(),
            'by_dimension': dim_stats.to_dict(),
            'by_domain': domain_stats.to_dict(),
        }
        
        summary_file = os.path.join(self.output_dir, 'summary_statistics.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Summary saved to {summary_file}")
