"""
Intrinsic Dimension Sweep Experiment
=====================================

Measures intrinsic dimensionality and variance explained across all embedding models.
Similar structure to the universal curvature experiment.
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List
from sklearn.decomposition import PCA
from tqdm import tqdm

from model_loaders import load_model_and_get_embeddings
from corpus_loaders import load_corpus


def measure_intrinsic_dimension(embeddings: np.ndarray, variance_thresholds: List[float] = [0.90, 0.95, 0.99]):
    """
    Measure intrinsic dimensionality using PCA.
    
    Args:
        embeddings: (N, D) array of embeddings
        variance_thresholds: List of variance thresholds to measure
        
    Returns:
        Dictionary with intrinsic dimension measurements
    """
    # Normalize embeddings (consistent with spherical geometry)
    X = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Fit PCA
    pca = PCA()
    pca.fit(X)
    
    # Cumulative variance
    cumsum = pca.explained_variance_ratio_.cumsum()
    
    # Compute intrinsic dimensions for different thresholds
    results = {
        'full_dim': embeddings.shape[1],
        'n_samples': embeddings.shape[0],
    }
    
    for threshold in variance_thresholds:
        d_intrinsic = np.sum(cumsum < threshold) + 1
        compression = embeddings.shape[1] / d_intrinsic
        
        results[f'd_intrinsic_{int(threshold*100)}'] = d_intrinsic
        results[f'compression_{int(threshold*100)}'] = compression
    
    # Additional statistics
    results['variance_first_pc'] = pca.explained_variance_ratio_[0]
    results['variance_top10_pcs'] = pca.explained_variance_ratio_[:10].sum()
    results['variance_top50_pcs'] = pca.explained_variance_ratio_[:50].sum() if len(pca.explained_variance_ratio_) >= 50 else pca.explained_variance_ratio_.sum()
    
    # Effective rank (participation ratio)
    var_ratios = pca.explained_variance_ratio_
    effective_rank = np.exp(-np.sum(var_ratios * np.log(var_ratios + 1e-10)))
    results['effective_rank'] = effective_rank
    
    # Store full variance spectrum (first 100 components)
    results['variance_spectrum'] = pca.explained_variance_ratio_[:100].tolist()
    
    return results


class IntrinsicDimensionExperiment:
    """
    Orchestrates intrinsic dimension measurement across all models.
    """
    
    def __init__(self, output_dir: str = './results'):
        self.output_dir = output_dir
        self.results = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Results file
        self.results_file = os.path.join(output_dir, 'intrinsic_dim_results.csv')
        
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
        variance_thresholds: List[float] = [0.90, 0.95, 0.99],
        skip_download: bool = False
    ) -> Dict:
        """
        Run experiment for a single model.
        
        Args:
            model_name: Name of the model
            model_config: Model configuration dict
            variance_thresholds: Variance thresholds to measure
            skip_download: If True, skip models that aren't already cached
            
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
                print(f"   d_intrinsic (95%): {result.get('d_intrinsic_95', 'N/A')}")
                return result
        
        # Check if model is cached (if skip_download is enabled)
        if skip_download:
            from transformers import AutoModel, AutoTokenizer
            from sentence_transformers import SentenceTransformer
            import os
            
            try:
                model_path = model_config['model']
                source = model_config['source']
                
                # Check cache based on source
                if source == 'transformers':
                    cache_dir = os.path.expanduser('~/.cache/huggingface/hub')
                    # Simple heuristic: check if cache directory exists
                    model_cache_name = f"models--{model_path.replace('/', '--')}"
                    if not os.path.exists(os.path.join(cache_dir, model_cache_name)):
                        print(f"\n⏭️  Model not cached, skipping (use --no-skip-download to download)...")
                        return {'model': model_name, 'skipped': True, 'reason': 'not_cached'}
                elif source == 'sentence-transformers':
                    cache_dir = os.path.expanduser('~/.cache/torch/sentence_transformers')
                    model_cache_name = model_path.replace('/', '_')
                    if not os.path.exists(os.path.join(cache_dir, model_cache_name)):
                        print(f"\n⏭️  Model not cached, skipping (use --no-skip-download to download)...")
                        return {'model': model_name, 'skipped': True, 'reason': 'not_cached'}
            except Exception as e:
                print(f"\n⚠️  Could not check cache status: {e}")
                # Continue anyway if cache check fails
        
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
            
            # Measure intrinsic dimension
            print(f"\nStep 3: Measuring intrinsic dimension...")
            dim_results = measure_intrinsic_dimension(embeddings, variance_thresholds)
            
            # Store results
            result_row = {
                'model': model_name,
                'model_path': model_config['model'],
                'year': model_config['year'],
                'dim': model_config['dim'],
                'arch': model_config['arch'],
                'domain': model_config['domain'],
                'n_samples': dim_results['n_samples'],
                'full_dim': dim_results['full_dim'],
                'd_intrinsic_90': dim_results['d_intrinsic_90'],
                'd_intrinsic_95': dim_results['d_intrinsic_95'],
                'd_intrinsic_99': dim_results['d_intrinsic_99'],
                'compression_90': dim_results['compression_90'],
                'compression_95': dim_results['compression_95'],
                'compression_99': dim_results['compression_99'],
                'variance_first_pc': dim_results['variance_first_pc'],
                'variance_top10_pcs': dim_results['variance_top10_pcs'],
                'variance_top50_pcs': dim_results['variance_top50_pcs'],
                'effective_rank': dim_results['effective_rank'],
                'timestamp': datetime.now().isoformat()
            }
            
            self.results.append(result_row)
            
            # Save variance spectrum separately
            spectrum_file = os.path.join(self.output_dir, f'variance_spectrum_{model_name}.json')
            with open(spectrum_file, 'w') as f:
                json.dump({
                    'model': model_name,
                    'variance_spectrum': dim_results['variance_spectrum']
                }, f, indent=2)
            
            # Save immediately (checkpoint)
            self.save_results()
            
            # Print summary
            print(f"\n{'='*70}")
            print(f"RESULTS FOR {model_name}")
            print(f"{'='*70}")
            print(f"Full Dimension: {dim_results['full_dim']}")
            print(f"Intrinsic Dim (90%): {dim_results['d_intrinsic_90']} ({dim_results['compression_90']:.2f}x compression)")
            print(f"Intrinsic Dim (95%): {dim_results['d_intrinsic_95']} ({dim_results['compression_95']:.2f}x compression)")
            print(f"Intrinsic Dim (99%): {dim_results['d_intrinsic_99']} ({dim_results['compression_99']:.2f}x compression)")
            print(f"First PC explains: {dim_results['variance_first_pc']:.1%}")
            print(f"Top 10 PCs explain: {dim_results['variance_top10_pcs']:.1%}")
            print(f"Effective Rank: {dim_results['effective_rank']:.1f}")
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
    
    def run_all_models(self, models_dict: Dict, variance_thresholds: List[float] = [0.90, 0.95, 0.99], skip_download: bool = False):
        """
        Run experiment on all models.
        
        Args:
            models_dict: Dictionary of model configurations
            variance_thresholds: Variance thresholds to measure
            skip_download: If True, skip models that aren't already cached
        """
        print(f"\n{'#'*70}")
        print(f"INTRINSIC DIMENSION SWEEP EXPERIMENT")
        print(f"{'#'*70}")
        print(f"Models to test: {len(models_dict)}")
        print(f"Variance thresholds: {variance_thresholds}")
        print(f"Skip uncached models: {skip_download}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'#'*70}\n")
        
        for i, (model_name, config) in enumerate(models_dict.items(), 1):
            print(f"\n[{i}/{len(models_dict)}] Processing: {model_name}")
            
            self.run_single_model(model_name, config, variance_thresholds, skip_download)
        
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
        df = df[df['d_intrinsic_95'].notna()]
        
        if len(df) == 0:
            print("\nNo valid results.")
            return
        
        print(f"\n{'='*70}")
        print(f"SUMMARY: Intrinsic Dimension Across All Models")
        print(f"{'='*70}")
        
        print(f"\nOverall Statistics (95% variance threshold):")
        print(f"  Models tested: {len(df)}")
        print(f"  Mean intrinsic dim: {df['d_intrinsic_95'].mean():.1f}")
        print(f"  Std intrinsic dim:  {df['d_intrinsic_95'].std():.1f}")
        print(f"  Mean compression: {df['compression_95'].mean():.2f}x")
        print(f"  Mean effective rank: {df['effective_rank'].mean():.1f}")
        
        print(f"\nBy Architecture:")
        arch_stats = df.groupby('arch').agg({
            'd_intrinsic_95': ['mean', 'std'],
            'compression_95': 'mean',
            'effective_rank': 'mean'
        }).round(2)
        print(arch_stats)
        
        print(f"\nBy Dimension:")
        dim_stats = df.groupby('dim').agg({
            'd_intrinsic_95': ['mean', 'std'],
            'compression_95': 'mean',
            'effective_rank': 'mean'
        }).round(2)
        print(dim_stats)
        
        print(f"\nBy Domain:")
        domain_stats = df.groupby('domain').agg({
            'd_intrinsic_95': ['mean', 'std'],
            'compression_95': 'mean',
            'effective_rank': 'mean'
        }).round(2)
        print(domain_stats)
        
        print(f"\n{'='*70}")
        
        # Save summary - flatten MultiIndex columns
        def flatten_stats(stats_df):
            """Convert pandas groupby result with MultiIndex columns to JSON-serializable dict."""
            result = {}
            for idx in stats_df.index:
                result[str(idx)] = {}
                for col in stats_df.columns:
                    # Handle MultiIndex columns (e.g., ('d_intrinsic_95', 'mean'))
                    if isinstance(col, tuple):
                        col_name = '_'.join(str(c) for c in col)
                    else:
                        col_name = str(col)
                    val = stats_df.loc[idx, col]
                    result[str(idx)][col_name] = float(val) if not pd.isna(val) else None
            return result
        
        summary = {
            'overall': {
                'n_models': int(len(df)),
                'mean_d_intrinsic_95': float(df['d_intrinsic_95'].mean()),
                'std_d_intrinsic_95': float(df['d_intrinsic_95'].std()),
                'mean_compression_95': float(df['compression_95'].mean()),
                'mean_effective_rank': float(df['effective_rank'].mean()),
            },
            'by_architecture': flatten_stats(arch_stats),
            'by_dimension': flatten_stats(dim_stats),
            'by_domain': flatten_stats(domain_stats),
        }
        
        summary_file = os.path.join(self.output_dir, 'intrinsic_dim_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n💾 Summary saved to {summary_file}")
