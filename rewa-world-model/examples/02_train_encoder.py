"""
Example 2: Contrastive Encoder Training

Demonstrates:
- Training contrastive encoder on hierarchical data
- Computing Fisher geometry
- Curvature diagnostics and visualization
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from data import HierarchicalGaussianGenerator
from neural import ContrastiveEncoder, ContrastiveTrainer
from geometry import FisherGeometryEstimator, GeometryDiagnostics, estimate_intrinsic_dimension
from witnesses import estimate_witness_distribution

def main():
    print("=== Contrastive Encoder Training ===")
    print()
    
    # 1. Generate hierarchical dataset
    print("1. Generating hierarchical Gaussian dataset...")
    generator = HierarchicalGaussianGenerator(
        n_levels=3,
        branching_factor=3,
        dim=32
    )
    documents = generator.generate(docs_per_leaf=10)
    print(f"   Generated {len(documents)} documents")
    print()
    
    # 2. Extract embeddings and labels
    print("2. Preparing training data...")
    embeddings = np.array([doc.embedding for doc in documents])
    labels = np.array([doc.metadata['cluster_id'] for doc in documents])
    doc_ids = [doc.id for doc in documents]
    
    # Create label mapping
    unique_labels = list(set(labels))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_map[label] for label in labels])
    
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Number of clusters: {len(unique_labels)}")
    print()
    
    # 3. Train contrastive encoder
    print("3. Training contrastive encoder...")
    input_dim = embeddings.shape[1]
    output_dim = 16
    
    encoder = ContrastiveEncoder(
        input_dim=input_dim,
        hidden_dim=64,
        output_dim=output_dim,
        num_layers=3
    )
    
    trainer = ContrastiveTrainer(
        encoder=encoder,
        temperature=0.07,
        learning_rate=1e-3
    )
    
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        loss = trainer.train_epoch(
            embeddings,
            numeric_labels,
            batch_size=16,
            delta=0.1
        )
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}")
    
    print()
    
    # 4. Encode documents
    print("4. Encoding documents with trained encoder...")
    encoded_embeddings = trainer.encode(embeddings)
    print(f"   Encoded shape: {encoded_embeddings.shape}")
    print()
    
    # 5. Compute Fisher geometry
    print("5. Computing Fisher geometry...")
    
    # Create simple witness distributions (use cluster membership as witness)
    witness_dists = []
    for doc in documents:
        cluster = doc.metadata['cluster_id']
        witness_dists.append({cluster: 1.0})
    
    geometry_estimator = FisherGeometryEstimator(encoder)
    fisher_metrics = geometry_estimator.compute_all_metrics(
        encoded_embeddings,
        doc_ids,
        witness_dists
    )
    
    print(f"   Computed Fisher metrics for {len(fisher_metrics)} documents")
    print()
    
    # 6. Run diagnostics
    print("6. Running geometry diagnostics...")
    diagnostics = GeometryDiagnostics(fisher_metrics)
    
    # Curvature statistics
    curvature_stats = diagnostics.compute_curvature_distribution()
    print("   Curvature Statistics:")
    print(f"     Mean: {curvature_stats['mean']:.6f}")
    print(f"     Std:  {curvature_stats['std']:.6f}")
    print(f"     Range: [{curvature_stats['min']:.6f}, {curvature_stats['max']:.6f}]")
    print()
    
    # Distortion
    mean_dist, std_dist = diagnostics.compute_distortion()
    print(f"   Geodesic Distortion: {mean_dist:.4f} ± {std_dist:.4f}")
    print()
    
    # Intrinsic dimension
    d_intrinsic = estimate_intrinsic_dimension(fisher_metrics, delta=0.1)
    print(f"   Estimated Intrinsic Dimension: {d_intrinsic:.2f}")
    print(f"   (vs. embedding dimension: {output_dim})")
    print()
    
    # 7. Generate visualizations
    print("7. Generating visualizations...")
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    diagnostics.plot_curvature_heatmap('results/curvature_heatmap.png')
    print("   Saved: results/curvature_heatmap.png")
    
    diagnostics.plot_embedding_space(save_path='results/embedding_space.png')
    print("   Saved: results/embedding_space.png")
    print()
    
    # 8. Generate report
    print("8. Geometry Report:")
    print(diagnostics.generate_report())
    print()
    
    print("✅ Contrastive encoder training and geometry analysis complete!")

if __name__ == "__main__":
    main()
