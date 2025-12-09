"""
Example 4: Semantic RG Coarse-Graining

Demonstrates:
- Multiscale witness coarse-graining
- Renormalized metric computation
- Compression ratio measurement
- Mutual information preservation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from data import HierarchicalGaussianGenerator
from witnesses import WitnessExtractor, WitnessType, estimate_witness_distribution
from geometry import FisherMetric
from semantic_rg import SemanticRG

def main():
    print("=== Semantic RG Coarse-Graining ===")
    print()
    
    # 1. Generate dataset
    print("1. Generating hierarchical dataset...")
    generator = HierarchicalGaussianGenerator(
        n_levels=3,
        branching_factor=3,
        dim=16
    )
    documents = generator.generate(docs_per_leaf=5)
    print(f"   Generated {len(documents)} documents")
    print()
    
    # 2. Extract witnesses
    print("2. Extracting witnesses...")
    extractor = WitnessExtractor([WitnessType.BOOLEAN])
    
    witness_dists = []
    for doc in documents:
        witnesses = extractor.extract({'id': doc.id, 'text': doc.text})
        dist = estimate_witness_distribution(witnesses)
        witness_dists.append(dist)
    
    print(f"   Extracted {len(witness_dists)} witness distributions")
    print()
    
    # 3. Create simple Fisher metrics
    print("3. Creating Fisher metrics...")
    embeddings = np.array([doc.embedding for doc in documents])
    doc_ids = [doc.id for doc in documents]
    
    metrics = []
    for emb in embeddings:
        d = len(emb)
        g = np.eye(d) + np.random.randn(d, d) * 0.05
        g = (g + g.T) / 2
        g += np.eye(d) * 0.01
        metrics.append(g)
    
    print(f"   Created {len(metrics)} Fisher metrics")
    print()
    
    # 4. Run Semantic RG
    print("4. Running Semantic RG coarse-graining...")
    rg = SemanticRG(num_scales=3, block_size_base=2)
    
    rg_flow = rg.build_rg_flow(witness_dists, metrics, doc_ids)
    
    print()
    print("RG Flow Results:")
    for i, scale in enumerate(rg_flow.scales):
        print(f"  Scale {i}:")
        print(f"    Witness blocks: {len(scale.witness_blocks)}")
        print(f"    Compression ratio: {scale.compression_ratio:.2f}x")
        if i < len(rg_flow.mutual_information):
            print(f"    MI preservation: {rg_flow.mutual_information[i]:.4f}")
    
    print()
    
    # 5. Analyze compression
    print("5. Compression Analysis:")
    total_compression = 1.0
    for scale in rg_flow.scales[1:]:
        total_compression *= scale.compression_ratio
    
    print(f"   Total compression: {total_compression:.2f}x")
    print(f"   Final scale size: {len(rg_flow.scales[-1].witness_blocks)} blocks")
    print()
    
    print("✅ Semantic RG coarse-graining complete!")

if __name__ == "__main__":
    main()
