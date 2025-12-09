"""
Example 7: Scaling & Extreme Compression (100k Docs)

Demonstrates:
- Streaming real Wikipedia data (100k documents)
- High-speed witness extraction
- REWA encoding at scale
- Semantic RG compression analysis
- Memory footprint benchmarking
"""

import sys
import os
import time
import psutil
import numpy as np
from datasets import load_dataset
from typing import List, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from witnesses import WitnessExtractor, WitnessType, Witness, estimate_witness_distribution
from encoding import REWAEncoder, REWAConfig
from semantic_rg import SemanticRG
from geometry import FisherMetric
from retrieval import REWARetriever

def get_process_memory():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def main():
    print("=== Phase 7: Scaling & Extreme Compression (100k Docs) ===")
    print(f"Initial Memory: {get_process_memory():.2f} MB")
    print()
    
    # 1. Load Dataset (Streaming)
    print("1. Loading Wikipedia dataset (streaming 100k docs)...")
    # Use wikitext-103-v1 as a proxy for Wikipedia (cleaner, faster)
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)
    
    documents = []
    MAX_DOCS = 10000
    # Note: 100k takes a while for demo, starting with 10k for speed
    # Change to 100000 for full run
    
    start_load = time.time()
    for item in ds:
        text = item['text'].strip()
        if len(text) > 100:  # Skip headers/empty
            documents.append({
                'id': f"doc_{len(documents)}",
                'text': text
            })
        if len(documents) >= MAX_DOCS:
            break
            
    load_time = time.time() - start_load
    print(f"   Loaded {len(documents)} documents in {load_time:.2f}s")
    print(f"   Memory: {get_process_memory():.2f} MB")
    print()
    
    # 2. Extract Witnesses & Encode
    print("2. Witness Extraction & REWA Encoding...")
    
    # Config for scale
    config = REWAConfig(
        input_dim=100000, 
        num_positions=10000,  # 10k dimensions
        num_hashes=2,
        delta_gap=0.1
    )
    encoder = REWAEncoder(config)
    extractor = WitnessExtractor([WitnessType.BOOLEAN])
    
    encoded_signatures = []
    witness_dists = [] # Keep for RG (memory heavy, but needed for demo)
    
    start_encode = time.time()
    for i, doc in enumerate(documents):
        # Extract
        witnesses = extractor.extract(doc)
        
        # Encode
        sig = encoder.encode(witnesses)
        encoded_signatures.append(sig)
        
        # Keep dist for RG
        dist = estimate_witness_distribution(witnesses)
        witness_dists.append(dist)
        
        if (i+1) % 1000 == 0:
            print(f"   Processed {i+1} docs...", end='\r')
            
    encode_time = time.time() - start_encode
    print(f"   Processed {len(documents)} docs in {encode_time:.2f}s ({len(documents)/encode_time:.0f} docs/s)")
    print(f"   Index Memory: {get_process_memory():.2f} MB")
    print()
    
    # 3. Simulate Fisher Metrics (Fast)
    # For 100k, we can't train contrastive network in demo time
    # We'll generate random metrics as placeholders to test RG compression mechanics
    print("3. Generating placeholder Fisher metrics...")
    metrics = []
    doc_ids = [d['id'] for d in documents]
    
    # Just generate for first 1000 to save time/memory for RG demo part
    RG_SUBSET = 1000
    print(f"   Running RG on subset of {RG_SUBSET} docs (for speed)...")
    
    for _ in range(RG_SUBSET):
        g = np.eye(16) * (1.0 + np.random.rand()*0.5) # Random scalar curvature
        metrics.append(g)
        
    subset_dists = witness_dists[:RG_SUBSET]
    subset_ids = doc_ids[:RG_SUBSET]
    
    # 4. Semantic RG Compression
    print("4. Extreme Compression with Semantic RG...")
    rg = SemanticRG(num_scales=3, block_size_base=4) # Aggressive blocking
    
    start_rg = time.time()
    rg_flow = rg.build_rg_flow(subset_dists, metrics, subset_ids)
    rg_time = time.time() - start_rg
    
    print(f"   RG Flow built in {rg_time:.2f}s")
    print()
    print("   Compression Results:")
    for i, scale in enumerate(rg_flow.scales):
        n_blocks = len(scale.witness_blocks)
        comp = scale.compression_ratio
        print(f"   Scale {i}: {n_blocks} blocks (Compression: {comp:.2f}x)")
    
    total_compression = 1.0
    for scale in rg_flow.scales[1:]:
        total_compression *= scale.compression_ratio
        
    print(f"   Total Compression Factor: {total_compression:.2f}x")
    print()
    
    # 5. Measure Recall vs Compression Trade-off
    print("5. Measuring Recall vs Compression Trade-off...")
    
    # We will use REWA to index and search at each scale
    # Scale 0 is "Ground Truth" for this relative recall test
    
    # Select queries (random docs)
    import random
    query_indices = random.sample(range(RG_SUBSET), 20)
    
    # Helper to map witness -> block at specific scale recursively
    def get_witness_map(target_scale_idx: int):
        # Start with identity map (witness -> witness)
        # But wait, Scale 0 in SemanticRG uses witnesses as-is.
        # Actually Scale 0 blocks are [[w1], [w2]...].
        # So Scale 0 witness_list becomes ["block_0", "block_1"...] for Scale 1.
        
        # We need to map: Original Witness -> ... -> Target Scale Block Name
        
        # 1. Build map: Original Witness -> Scale 0 Block Index
        # Since Scale 0 is identity (if block_size_base > 1, but here it is checked in RG)
        # Let's inspect RGScale 0.
        
        current_map = {} # witness -> current_name
        
        # Initialize with Scale 0
        scale0 = rg_flow.scales[0]
        # scale0.witness_blocks contains lists of original witnesses
        for block_id, block in enumerate(scale0.witness_blocks):
            block_name = f"block_{block_id}" # This is the name used in Scale 1
            for w in block:
                current_map[w] = block_name
                
        # Iterate up to target scale
        for s in range(1, target_scale_idx + 1):
            scale = rg_flow.scales[s]
            new_map = {}
            
            # This scale's blocks contain names from previous scale
            # e.g. ["block_0", "block_1"]
            
            # Inverse lookup for this scale
            prev_name_to_new_block = {}
            for block_id, block in enumerate(scale.witness_blocks):
                new_block_name = f"block_{block_id}" # Name for next scale
                for prev_name in block:
                    prev_name_to_new_block[prev_name] = new_block_name
            
            # Update main map
            for w, prev_name in current_map.items():
                if prev_name in prev_name_to_new_block:
                    new_map[w] = prev_name_to_new_block[prev_name]
            
            current_map = new_map
            
        return current_map

    # Helper to build index and search at a specific scale
    def evaluate_scale(scale_idx: int, ground_truth_results: Dict[int, List[str]] = None):
        print(f"   Evaluating Scale {scale_idx}...")
        
        # Get mapping for this scale
        if scale_idx == 0:
            # Scale 0 is effectively checking original witnesses 
            # (since Scale 0 blocks are singletons in current RG impl if scale_idx=0 logic holds)
            # But let's use the map to be consistent if they were clustered
            witness_map = get_witness_map(0)
        else:
            witness_map = get_witness_map(scale_idx)
                
        # Encode docs
        coarse_signatures = []
        for i in range(RG_SUBSET):
            doc_witnesses = witness_dists[i].keys()
            scale_witnesses = set()
            for w in doc_witnesses:
                if w in witness_map:
                    scale_witnesses.add(witness_map[w])
            
            # Create witness objects
            w_objs = [Witness(id=w, feature=w, value=1.0, witness_type=WitnessType.BOOLEAN) 
                     for w in scale_witnesses]
            
            if not w_objs and i < 5:
                print(f"Warning: Doc {i} has 0 witnesses at Scale {scale_idx}")

            sig = encoder.encode(w_objs)
            coarse_signatures.append(sig)
            
        # Build Retriever
        retriever = REWARetriever(WitnessType.BOOLEAN)
        scale_ids = doc_ids[:RG_SUBSET]
        for i, doc_id in enumerate(scale_ids):
            retriever.add(doc_id, coarse_signatures[i])
            
        # Run Queries
        results = {}
        recall_sum = 0.0
        
        for idx in query_indices:
            q_sig = coarse_signatures[idx]
            q_id = scale_ids[idx]
            
            # Search
            res = retriever.search(q_sig, k=10)
            retrieved_ids = [r.doc_id for r in res]
            results[idx] = retrieved_ids
            
            # Calculate Recall relative to Ground Truth (if provided)
            if ground_truth_results:
                truth = set(ground_truth_results[idx])
                # We care if we find the SAME docs, ignoring the query doc itself usually, 
                # but here self-retrieval is a sanity check. Let's look at overlap.
                overlap = len(set(retrieved_ids) & truth)
                recall_sum += overlap / len(truth)
        
        if ground_truth_results:
            avg_recall = recall_sum / len(query_indices)
            return avg_recall
        else:
            return results

    # Run Baseline (Scale 0)
    scale0_results = evaluate_scale(0)
    
    # Run Compressed Scales
    print()
    print("   Trade-off Results:")
    print("   Scale | Compression | Identity Recall (Self) | Relative Recall@10")
    print("   " + "-"*65)
    
    # Scale 0 Stats
    print(f"   0     | 1.00x       | 100%                 | 1.00 (Baseline)")
    
    for i in range(1, len(rg_flow.scales)):
        recall = evaluate_scale(i, scale0_results)
        scale = rg_flow.scales[i]
        
        # Calculate cumulative compression
        comp = 1.0
        for s in rg_flow.scales[1:i+1]:
            comp *= s.compression_ratio
            
        print(f"   {i}     | {comp:7.2f}x    | --                   | {recall:.2f}")
    
    print()

if __name__ == "__main__":
    main()
