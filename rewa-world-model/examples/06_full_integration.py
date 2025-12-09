"""
Example 6: Full Integration (Phases 1-6)

Demonstrates the complete Multiscale Geometric REWA World-Model:
1. Data Generation (Hierarchical + Compositional)
2. Phase 1: Witness Extraction & REWA Encoding
3. Phase 2: Neural Encoding & Fisher Geometry
4. Phase 3: Semantic RG Coarse-Graining
5. Phase 4: Topos Logic & Consistency
6. Phase 6: Unified Multiscale Retrieval

This script runs the end-to-end pipeline and queries the world-model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import time
from typing import List, Dict

# Import all modules
from data import HierarchicalGaussianGenerator, SyntheticDocument
from witnesses import WitnessExtractor, WitnessType, Witness, estimate_witness_distribution
from encoding import REWAEncoder, REWAConfig
from retrieval import REWARetriever, MultiscaleRetriever
from neural import ContrastiveEncoder, ContrastiveTrainer
from geometry import FisherGeometryEstimator, GeometryDiagnostics
from semantic_rg import SemanticRG
from topos import ToposLogic

def main():
    start_time = time.time()
    print("=== Multiscale Geometric REWA World-Model: Full Integration ===")
    print("===============================================================")
    print()
    
    # ---------------------------------------------------------
    # 1. Data Generation
    # ---------------------------------------------------------
    print("1. Data Generation")
    print("------------------")
    # Use hierarchical generator but add metadata for Topos logic
    generator = HierarchicalGaussianGenerator(
        n_levels=2,
        branching_factor=4,
        dim=32,
        seed=42
    )
    documents = generator.generate(docs_per_leaf=5)
    
    # Enrich metadata for Topos logic (simulating compositional attributes)
    colors = ['red', 'blue', 'green', 'yellow']
    shapes = ['cube', 'sphere', 'pyramid', 'cylinder']
    
    for i, doc in enumerate(documents):
        # Assign attributes based on cluster to maintain consistency
        cluster_hash = hash(doc.metadata['cluster_id'])
        color = colors[cluster_hash % len(colors)]
        shape = shapes[(cluster_hash // len(colors)) % len(shapes)]
        
        doc.metadata.update({
            'color': color,
            'shape': shape,
            'cluster_id': doc.metadata['cluster_id']
        })
        doc.text = f"{doc.text}. A {color} {shape}."
    
    print(f"   Generated {len(documents)} documents with hierarchical and compositional structure")
    print()
    
    # ---------------------------------------------------------
    # 2. Phase 1: Witnesses & REWA Encoding
    # ---------------------------------------------------------
    print("2. Phase 1: Witnesses & REWA Encoding")
    print("-------------------------------------")
    
    # Extract witnesses
    extractor = WitnessExtractor([WitnessType.BOOLEAN, WitnessType.NATURAL])
    witness_dists = []
    
    for doc in documents:
        witnesses = extractor.extract({'id': doc.id, 'text': doc.text, 'metadata': doc.metadata})
        
        # Add metadata attributes as strong witnesses
        for k, v in doc.metadata.items():
            if isinstance(v, str):
                witnesses.append(Witness(
                    id=f"{doc.id}_{k}_{v}",
                    feature=f"{k}_{v}",
                    value=1.0,
                    witness_type=WitnessType.BOOLEAN
                ))
        
        dist = estimate_witness_distribution(witnesses)
        witness_dists.append(dist)
    
    print(f"   Extracted witness distributions for {len(documents)} docs")
    
    # REWA Encoding
    config = REWAConfig(
        input_dim=10000,  # Estimated unique witnesses
        num_positions=2048, 
        num_hashes=2,
        delta_gap=0.1
    )
    rewa_encoder = REWAEncoder(config)
    
    signatures = []
    for dist in witness_dists:
        # Convert dist to list of witnesses for encoding
        # (Simplified: just treat as Boolean for signature)
        witnesses = [
            Witness(
                id=f"w_{w}", 
                feature=w, 
                value=1.0, 
                witness_type=WitnessType.BOOLEAN
            ) 
            for w in dist.keys()
        ]
        sig = rewa_encoder.encode(witnesses)
        signatures.append(sig)
    
    signatures = np.array(signatures)
    
    # Build REWA Retriever
    rewa_retriever = REWARetriever(WitnessType.BOOLEAN)
    for i, doc in enumerate(documents):
        rewa_retriever.add(doc.id, signatures[i])
        
    print(f"   Built REWA index: {rewa_retriever.size()} documents, {rewa_retriever.memory_usage():.2f} KB")
    print()
    
    # ---------------------------------------------------------
    # 3. Phase 2: Neural & Geometry
    # ---------------------------------------------------------
    print("3. Phase 2: Neural Encoding & Geometry")
    print("--------------------------------------")
    
    # Prepare data
    raw_embeddings = np.array([doc.embedding for doc in documents])
    labels = np.array([hash(doc.metadata['cluster_id']) % 100 for doc in documents])
    
    # Train Contrastive Encoder
    print("   Training contrastive encoder...")
    input_dim = raw_embeddings.shape[1]
    
    neural_encoder = ContrastiveEncoder(input_dim=input_dim, output_dim=16)
    trainer = ContrastiveTrainer(neural_encoder, learning_rate=1e-3)
    
    # Quick training
    loss = trainer.train_epoch(raw_embeddings, labels, batch_size=32)
    for _ in range(4): # 5 epochs total
        loss = trainer.train_epoch(raw_embeddings, labels, batch_size=32)
        
    print(f"   Training complete (Loss: {loss:.4f})")
    
    # Encode
    learned_embeddings = trainer.encode(raw_embeddings)
    
    # Compute Fisher Geometry
    print("   Computing Fisher geometry...")
    geo_estimator = FisherGeometryEstimator(neural_encoder)
    doc_ids = [doc.id for doc in documents]
    
    fisher_metrics = geo_estimator.compute_all_metrics(
        learned_embeddings,
        doc_ids,
        witness_dists
    )
    
    # Diagnostics
    diagnostics = GeometryDiagnostics(fisher_metrics)
    curv_stats = diagnostics.compute_curvature_distribution()
    print(f"   Mean curvature: {curv_stats['mean']:.4f}")
    print()
    
    # ---------------------------------------------------------
    # 4. Phase 3: Semantic RG
    # ---------------------------------------------------------
    print("4. Phase 3: Semantic RG")
    print("-----------------------")
    
    semantic_rg = SemanticRG(num_scales=2, block_size_base=2)
    rg_flow = semantic_rg.build_rg_flow(witness_dists, [m.metric for m in fisher_metrics], doc_ids)
    
    print(f"   Built RG flow with {len(rg_flow.scales)} scales")
    print(f"   Coarsest scale compression: {rg_flow.scales[-1].compression_ratio:.2f}x")
    print()
    
    # ---------------------------------------------------------
    # 5. Phase 4: Topos Logic
    # ---------------------------------------------------------
    print("5. Phase 4: Topos Logic")
    print("-----------------------")
    
    topos = ToposLogic(confidence_threshold=0.5)
    
    for doc, dist in zip(documents, witness_dists):
        topos.build_section(doc.id, dist)
        
    print(f"   Built {len(topos.sections)} local logical sections")
    print()
    
    # ---------------------------------------------------------
    # 6. Phase 6: Unified Retrieval
    # ---------------------------------------------------------
    print("6. Phase 6: Unified Multiscale Retrieval")
    print("----------------------------------------")
    
    # Initialize Unified Retriever
    retriever = MultiscaleRetriever(
        rewa_encoder=rewa_encoder,
        rewa_retriever=rewa_retriever,
        rg_flow=rg_flow,
        fisher_metrics=fisher_metrics,
        topos=topos
    )
    
    # Run a query
    # Find a "blue cube"
    query_text = "blue cube"
    print(f"   Query: '{query_text}'")
    
    # Create query signature (simulated)
    query_witnesses = [
        Witness(id="q_w1", feature="blue", value=1.0, witness_type=WitnessType.BOOLEAN),
        Witness(id="q_w2", feature="cube", value=1.0, witness_type=WitnessType.BOOLEAN),
        Witness(id="q_w3", feature="color_blue", value=1.0, witness_type=WitnessType.BOOLEAN),
        Witness(id="q_w4", feature="shape_cube", value=1.0, witness_type=WitnessType.BOOLEAN)
    ]
    query_signature = rewa_encoder.encode(query_witnesses)
    
    # Create query embedding (random for demo, ideally encoded)
    # In real system, we'd embed the query text
    query_embedding = np.random.randn(16)
    query_embedding /= np.linalg.norm(query_embedding)
    
    # Retrieve
    results = retriever.retrieve(
        query_signature=query_signature,
        query_embedding=query_embedding,
        query_id="query_0",
        k=5
    )
    
    print()
    print("   Top Results:")
    print("   Rank | Doc ID | Final Score | REWA | Geo  | Consist.")
    print("   " + "-"*55)
    
    for i, res in enumerate(results):
        # Get actual doc info
        doc = next(d for d in documents if d.id == res.doc_id)
        cluster = doc.metadata['cluster_id']
        color = doc.metadata['color']
        shape = doc.metadata['shape']
        
        print(f"   #{i+1:<3} | {res.doc_id:<6} | {res.final_score:.4f}      | {res.rewa_score:.2f} | {res.geometric_score:.2f} | {res.consistency_score:.2f}")
        print(f"        Values: {color} {shape} (Cluster: {cluster})")
    
    elapsed = time.time() - start_time
    print()
    print("===============================================================")
    print(f"✅ Full integration complete in {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()
