"""
Geometry Diagnostics Module

Visualizations and metrics for geometric analysis:
- Curvature heatmaps
- Geodesic vs Euclidean distortion
- Intrinsic dimension estimation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from .fisher import FisherMetric, compute_scalar_curvature

class GeometryDiagnostics:
    """Diagnostic tools for geometric analysis"""
    
    def __init__(self, metrics: List[FisherMetric]):
        self.metrics = metrics
        self.doc_ids = [m.doc_id for m in metrics]
        
    def compute_curvature_distribution(self) -> Dict[str, float]:
        """Compute curvature statistics."""
        curvatures = [compute_scalar_curvature(m.metric) for m in self.metrics]
        
        return {
            'mean': np.mean(curvatures),
            'std': np.std(curvatures),
            'min': np.min(curvatures),
            'max': np.max(curvatures),
            'median': np.median(curvatures)
        }
    
    def compute_distortion(self) -> Tuple[float, float]:
        """
        Compute geodesic vs Euclidean distance distortion.
        
        Returns:
            (mean_distortion, std_distortion)
        """
        distortions = []
        
        for i in range(len(self.metrics)):
            for j in range(i + 1, min(i + 10, len(self.metrics))):
                # Geodesic distance
                d_geo = self.metrics[i].geodesic_distance(self.metrics[j])
                
                # Euclidean distance
                d_euc = np.linalg.norm(
                    self.metrics[i].embedding - self.metrics[j].embedding
                )
                
                if d_euc > 1e-10:
                    distortion = abs(d_geo - d_euc) / d_euc
                    distortions.append(distortion)
        
        return np.mean(distortions), np.std(distortions)
    
    def plot_curvature_heatmap(self, save_path: Optional[str] = None):
        """Plot curvature heatmap over documents."""
        curvatures = [compute_scalar_curvature(m.metric) for m in self.metrics]
        
        plt.figure(figsize=(12, 4))
        plt.plot(curvatures, marker='o', markersize=3, linewidth=0.5)
        plt.xlabel('Document Index')
        plt.ylabel('Scalar Curvature')
        plt.title('Curvature Distribution Across Documents')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_embedding_space(
        self,
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """Plot 2D projection of embedding space colored by curvature."""
        from sklearn.decomposition import PCA
        
        # Get embeddings
        embeddings = np.array([m.embedding for m in self.metrics])
        curvatures = [compute_scalar_curvature(m.metric) for m in self.metrics]
        
        # PCA to 2D
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=curvatures,
            cmap='viridis',
            s=50,
            alpha=0.6
        )
        plt.colorbar(scatter, label='Scalar Curvature')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        plt.title('Embedding Space (colored by curvature)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def generate_report(self) -> str:
        """Generate text report of geometric diagnostics."""
        curvature_stats = self.compute_curvature_distribution()
        mean_distortion, std_distortion = self.compute_distortion()
        
        report = []
        report.append("=== Geometry Diagnostics Report ===\n")
        report.append(f"Number of documents: {len(self.metrics)}\n")
        report.append(f"Embedding dimension: {len(self.metrics[0].embedding)}\n")
        report.append("\nCurvature Statistics:")
        report.append(f"  Mean: {curvature_stats['mean']:.6f}")
        report.append(f"  Std:  {curvature_stats['std']:.6f}")
        report.append(f"  Min:  {curvature_stats['min']:.6f}")
        report.append(f"  Max:  {curvature_stats['max']:.6f}")
        report.append(f"\nGeodesic Distortion:")
        report.append(f"  Mean: {mean_distortion:.4f}")
        report.append(f"  Std:  {std_distortion:.4f}")
        
        return "\n".join(report)
