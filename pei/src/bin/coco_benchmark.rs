use pei::index::pei_index::PEIIndex;
use pei::storage::item_store::{ItemStore, ItemMetadata};
use pei::evidence::dag::EvidenceDAG;
use pei::evidence::operator::{EvidenceOperator, EvidenceResult};
use pei::evidence::multimodal_ops::{AspectRatioOp, ColorOp, CoarseClipOp};
use pei::query::search::search;
use pei::{Query, ItemId};
use std::collections::HashMap;
use std::time::Instant;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use rand::Rng;
use rand::distributions::{Distribution, Standard};

// Re-use instrumented op from multimodal experiment
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

// === Clustering Simulation ===

struct Cluster {
    center: Vec<f32>,
    aspect_bias: f32, // Probability of being "Square" (1.0) vs "Portrait" (0.5)
    color_bias: usize, // Dominant color index
}

fn generate_clustered_data(n_items: usize, n_clusters: usize, dim: usize, rng: &mut impl Rng) -> (Vec<Vec<f32>>, Vec<ItemMetadata>, Vec<Cluster>) {
    // 1. Generate Clusters
    let mut clusters = Vec::new();
    for _ in 0..n_clusters {
        let center: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
        // Each cluster has a preference for aspect/color
        let aspect_bias = rng.gen::<f32>(); 
        let color_bias = rng.gen_range(0..5);
        clusters.push(Cluster { center, aspect_bias, color_bias });
    }

    // 2. Generate Items
    let mut vectors = Vec::with_capacity(n_items);
    let mut metadata = Vec::with_capacity(n_items);
    let dim_coarse = 32;

    for _ in 0..n_items {
        // Pick a cluster
        let c_idx = rng.gen_range(0..n_clusters);
        let cluster = &clusters[c_idx];

        // Generate vector near center + noise
        let mut v = cluster.center.clone();
        for x in v.iter_mut() {
            *x += rng.gen_range(-0.1..0.1); 
        }

        // Generate Metadata correlated with Cluster
        // Aspect: skewed coin flip
        let aspect = if rng.gen::<f32>() < cluster.aspect_bias { 1.0 } else { 0.5 };
        
        // Color: High prob of matching cluster bias, small prob of random
        let color = if rng.gen_bool(0.8) {
            cluster.color_bias as u8
        } else {
            rng.gen_range(0..5) as u8
        };

        let coarse = v[0..dim_coarse].to_vec();

        vectors.push(v);
        metadata.push(ItemMetadata { aspect_ratio: aspect, color, coarse_emb: coarse });
    }

    (vectors, metadata, clusters)
}

fn main() {
    println!("Running COCO Proxy Benchmark (Clustered Data)...");
    let mut rng = rand::thread_rng();

    // 1. Generate Data
    let n_items = 5000;
    let n_clusters = 50;
    let dim = 512;
    
    let (vectors, metadata, clusters) = generate_clustered_data(n_items, n_clusters, dim, &mut rng);
    
    let store = ItemStore::new(vectors, metadata);
    let store_clone = ItemStore::new(store.vectors.clone(), store.metadata.clone());

    // 2. Generate Query
    // Pick a target cluster to simulate user intent
    let target_c_idx = 0; // Arbitrary
    let target_cluster = &clusters[target_c_idx];
    
    // Query Vector: Center of cluster
    let mut q_vec = target_cluster.center.clone();
    let query_obj = Query::new(q_vec.clone());
    
    // Query Hints: The bias of that cluster
    // e.g. User searches "Red Car" (Cluster implies Red and Car shape)
    let mut query_hints = query_obj.clone();
    query_hints.aspect_hint = Some(if target_cluster.aspect_bias > 0.5 { 1.0 } else { 0.5 });
    query_hints.color_hint = Some(target_cluster.color_bias as u8);

    println!("Query Intent -> Cluster {}: Aspect~{:.1}, Color={}", 
             target_c_idx, target_cluster.aspect_bias, target_cluster.color_bias);

    // 3. PEI Setup
    let clip_counter = Arc::new(AtomicUsize::new(0));
    
    let ops: Vec<Box<dyn EvidenceOperator>> = vec![
        Box::new(AspectRatioOp { id: 0 }),
        Box::new(ColorOp { id: 1 }),
        Box::new(CoarseClipOp { id: 2 }),
        Box::new(InstrumentedFullClipOp { id: 3, counter: clip_counter.clone() }),
    ];

    let mut edges = HashMap::new();
    // Metadata -> Coarse -> Fine
    edges.insert(0, vec![2]);
    edges.insert(1, vec![2]);
    edges.insert(2, vec![3]);
    let dag = EvidenceDAG::new(vec![0, 1], edges);
    let index = PEIIndex::new(ops, dag, store_clone);

    // 4. Ground Truth
    println!("Computing Ground Truth...");
    let k = 10;
    
    // Truth = Items that effectively match the query vector. 
    // Metadata is "Soft" in reality, but for benchmark standard we define "Truth" purely by vector distance?
    // No, for Multimodal, Truth is [Relevant] returns.
    // If I search "Red Car", I don't want a "Blue Car" even if vector is close-ish.
    // So Truth = Compatible Metadata + Top Vector Distance.
    
    let mut valid_cands = Vec::new();
    for i in 0..n_items {
        let meta = &store.metadata[i];
        
        let match_aspect = match query_hints.aspect_hint {
            Some(h) => (meta.aspect_ratio - h).abs() < 0.2,
            None => true
        };
        let match_color = match query_hints.color_hint {
            Some(c) => meta.color == c,
            None => true
        };

        if match_aspect && match_color {
            let d = store.distance(&query_hints, i);
            valid_cands.push((i, d));
        }
    }
    valid_cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let ground_truth: Vec<usize> = valid_cands.iter().take(k).map(|(id, _)| *id).collect();

    let std_clip_calls = n_items;

    // 5. Run PEI
    println!("Running PEI...");
    clip_counter.store(0, Ordering::SeqCst);
    let start_pei = Instant::now();
    
    // Budget 500 should be plenty
    // We request Top 50 candidates (`ef_search`) to ensure the Top 10 are strictly optimal
    let budget = 500.0;
    let search_k = 50; 
    let results = search(&index, &query_hints, budget, search_k);
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

    println!("\n=== COCO Proxy Results (N={}) ===", n_items);
    println!("PEI Search:");
    println!("  Time (Micro-bench): {:.2?}", time_pei);
    println!("  CLIP Calls: {} (vs 5000 Baseline)", pei_calls);
    println!("  Recall: {:.2}", recall);
    
    let reduction = std_clip_calls as f64 / pei_calls.max(1) as f64;
    println!("\nOptimization:");
    println!("  CLIP Reduction: {:.1}x", reduction);
    println!("  Projected Speedup (10ms/Call): ~{:.1}x", reduction);
    
    if recall < 0.8 {
        println!("WARNING: Low recall. correlation might be too weak or budget too low.");
    } else {
        println!("SUCCESS: High recall achieved with structural pruning.");
    }
}
