use pei::index::pei_index::PEIIndex;
use pei::storage::item_store::{ItemStore, ItemMetadata};
use pei::evidence::dag::EvidenceDAG;
use pei::evidence::operator::{EvidenceOperator, EvidenceResult};
use pei::evidence::multimodal_ops::{AspectRatioOp, ColorOp, CoarseClipOp}; // FullClipOp implemented locally for counting
use pei::query::search::search;
use pei::{Query, ItemId};
use std::collections::HashMap;
use std::time::Instant;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use rand::Rng;

// Instrumented Full Clip Op to count calls
pub struct InstrumentedFullClipOp {
    pub id: usize,
    pub counter: Arc<AtomicUsize>,
}

impl EvidenceOperator for InstrumentedFullClipOp {
    fn id(&self) -> usize { self.id }
    fn cost(&self) -> f64 { 1.0 }

    fn apply(&self, query: &Query, item_id: ItemId, store: &ItemStore) -> EvidenceResult {
        self.counter.fetch_add(1, Ordering::SeqCst);
        let d = store.distance(query, item_id);
        EvidenceResult {
            belief_delta: -(d as f64),
            uncertainty_delta: 0.99,
        }
    }
}

fn main() {
    println!("Running Multimodal Benchmark (Images + Text)...");
    
    // 1. Dataset Generation (N=5000)
    let n_items = 5000;
    let dim_fine = 512;
    let dim_coarse = 32;
    let mut rng = rand::thread_rng();
    
    let mut vectors = Vec::with_capacity(n_items);
    let mut metadata = Vec::with_capacity(n_items);
    
    for _ in 0..n_items {
        // Random fine vector
        let fine: Vec<f32> = (0..dim_fine).map(|_| rng.gen::<f32>()).collect();
        // Coarse is first 32 dims (simulated)
        let coarse = fine[0..dim_coarse].to_vec();
        
        // Metadata
        let aspect = if rng.gen_bool(0.5) { 1.0 } else { 0.5 }; // Square or Portrait
        let color = rng.gen_range(0..5) as u8; // 5 dominant colors
        
        vectors.push(fine);
        metadata.push(ItemMetadata {
            aspect_ratio: aspect,
            color,
            coarse_emb: coarse,
        });
    }
    
    let store = ItemStore::new(vectors, metadata);
    let store_clone = ItemStore::new(store.vectors.clone(), store.metadata.clone());
    
    // 2. Query Generation
    // Query targets Aspect=1.0, Color=2
    let mut q_vec: Vec<f32> = (0..dim_fine).map(|_| rng.gen::<f32>()).collect();
    let mut query = Query::new(q_vec.clone());
    query.aspect_hint = Some(1.0);
    query.color_hint = Some(2);
    
    // 3. PEI Setup
    // Op 0: Aspect (Cost 0.001)
    // Op 1: Color (Cost 0.001)
    // Op 2: Coarse (Cost 0.05)
    // Op 3: Full (Cost 1.0)
    
    let clip_counter = Arc::new(AtomicUsize::new(0));
    
    let ops: Vec<Box<dyn EvidenceOperator>> = vec![
        Box::new(AspectRatioOp { id: 0 }),
        Box::new(ColorOp { id: 1 }),
        Box::new(CoarseClipOp { id: 2 }),
        Box::new(InstrumentedFullClipOp { id: 3, counter: clip_counter.clone() }),
    ];
    
    // DAG: Roots {0, 1}. 0->2, 1->2. 2->3.
    // Try Metadata first. Then Coarse. Then Fine.
    let mut edges = HashMap::new();
    edges.insert(0, vec![2]);
    edges.insert(1, vec![2]);
    edges.insert(2, vec![3]);
    
    let dag = EvidenceDAG::new(vec![0, 1], edges);
    let index = PEIIndex::new(ops, dag, store_clone);
    
    // 4. Ground Truth (Filtered by Metadata - The "User Intent")
    println!("Computing Standard Search Baseline...");
    let start_std = Instant::now();
    let k = 10;
    
    // Standard Search (Naive)
    // 1. Compute all distances
    // 2. Return top K (ignoring metadata compatibility)
    // To measure "Recall" fairly, we must define what is "Relevant".
    // "Relevant" = Compatible Metadata AND High Vector Similarity.
    // So we compute the "True Top K" from the subset of compatible items.
    
    let mut valid_cands = Vec::with_capacity(n_items);
    for i in 0..n_items {
        // Check validity first (God Mode definition of Truth)
        let meta = &store.metadata[i];
        if (meta.aspect_ratio - 1.0).abs() <= 0.2 && meta.color == 2 {
             let d = store.distance(&query, i);
             valid_cands.push((i, d));
        }
    }
    valid_cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let ground_truth: Vec<usize> = valid_cands.iter().take(k).map(|(id, _)| *id).collect();
    
    // Standard search doesn't filter, but performs full scan. 
    // We count its cost (ALL items) and its ability to find these specific targets.
    // In a real naive retrieval, it might return incompatible items.
    // We assume the baseline is "Compute All Distances, then Filter" OR "Compute All, Return Mixed".
    // The benchmark goal: "Can we avoid computing distance for incompatible items?"
    let std_clip_calls = n_items;
    
    let time_std = start_std.elapsed();
    
    // 5. PEI Run
    println!("Running PEI Search...");
    clip_counter.store(0, Ordering::SeqCst);
    let start_pei = Instant::now();
    
    // Budget: Enough to Check metadata + Coarse + some Fine.
    let budget = 300.0;
    
    let results = search(&index, &query, budget, k);
    let time_pei = start_pei.elapsed();
    let pei_calls = clip_counter.load(Ordering::SeqCst);
    
    // 6. Metrics
    let mut hits = 0;
    for c in &results {
        if ground_truth.contains(&c.item_id) {
            hits += 1;
        }
    }
    let recall = hits as f64 / k as f64;
    
    println!("\n=== Results (N={}) ===", n_items);
    println!("Standard Search:");
    println!("  Time (Micro-bench): {:.2?}", time_std);
    println!("  CLIP Calls: {}", std_clip_calls);
    println!("  Recall: 1.0 (Definition)");
    
    println!("PEI Search:");
    println!("  Time (Micro-bench): {:.2?}", time_pei);
    println!("  CLIP Calls: {}", pei_calls);
    println!("  Recall: {:.2}", recall);
    
    println!("\n=== Efficiency Gains ===");
    println!("  CLIP Reduction: {:.2}x fewer calls", std_clip_calls as f64 / pei_calls as f64);
    
    // Projected Speedup assuming 10ms per CLIP call (Real World)
    // Overhead is negligible compared to 10ms.
    // Speedup ~ Calls Ratio.
    println!("  Projected Real-World Speedup (10ms/Call): ~{:.2}x", std_clip_calls as f64 / pei_calls as f64);
}
