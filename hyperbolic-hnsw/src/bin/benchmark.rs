use hyperbolic_hnsw::embedding::{HyperbolicEmbedder, TreeNode};
use hyperbolic_hnsw::hnsw::HyperbolicHNSW;
use ndarray::{Array1, ArrayView1};
use std::collections::{BinaryHeap, HashMap};
use std::time::Instant;
use ordered_float::OrderedFloat;
use std::cmp::Reverse;

fn main() {
    let dim = 10;
    let curvature = 1.0;
    let embedder = HyperbolicEmbedder::new(dim, curvature);

    println!("Generating synthetic tree (Depth=5, Branch=6)...");
    // 1 + 6 + 36 + 216 + 1296 + 7776 = 9331 nodes
    let tree = build_tree(5, 6, "root".to_string());

    println!("Embedding tree into hyperbolic space...");
    let start_embed = Instant::now();
    let embeddings = embedder.embed_tree(&tree, 0.5);
    println!("Embedded {} nodes in {:.2?}s", embeddings.len(), start_embed.elapsed());

    // Convert map to list for indexing
    let mut data_points = Vec::new();
    let mut ids = Vec::new();
    for (id, point) in &embeddings {
        ids.push(id);
        data_points.push(point.clone());
    }

    println!("Building Hyperbolic HNSW index...");
    // M=16, ef_construction=200
    let ml = 1.0 / (2.0f64).ln();
    let mut index = HyperbolicHNSW::new(dim, curvature, 16, 16, 32, 200, ml);
    
    let start_build = Instant::now();
    for point in &data_points {
        index.insert(point.clone());
    }
    println!("Index built in {:.2?}s", start_build.elapsed());

    // Generate queries
    // Use some existing points as queries (leave-one-out style logic, but here just searching for themselves/neighbors)
    // Or simpler: just use first 100 points
    let num_queries = 100;
    let queries = &data_points[0..num_queries];
    let k = 10;

    println!("Computing Ground Truth (Brute Force) for {} queries...", num_queries);
    let start_gt = Instant::now();
    let ground_truth: Vec<Vec<usize>> = queries.iter().map(|q| {
        brute_force_search(&embedder, &data_points, q, k)
    }).collect();
    println!("Ground truth computed in {:.2?}s", start_gt.elapsed());

    println!("\nRunning HNSW Benchmark...");
    println!("| ef | Recall@{} | Avg Latency (ms) | QPS |", k);
    println!("|----|-----------|------------------|-----|");

    let ef_values = vec![10, 20, 50, 100, 200];

    for &ef in &ef_values {
        let start_bench = Instant::now();
        let mut total_recall = 0.0;

        for (i, query) in queries.iter().enumerate() {
            let results = index.search(query, k, ef);
            let result_indices: Vec<usize> = results.iter().map(|(idx, _)| *idx).collect();
            
            // Calculate overlap with ground truth
            let gt_set = &ground_truth[i];
            let mut matches = 0;
            for idx in &result_indices {
                // In exact search, we match indices. 
                // Note: index.data order matches insertion order?
                // HNSW insertion: we iterated `data_points` and called `insert`.
                // `index.data` grows sequentially. So `index.data[i]` == `data_points[i]`.
                // Yes, valid assumption.
                if gt_set.contains(idx) {
                    matches += 1;
                }
            }
            total_recall += (matches as f64) / (k as f64);
        }

        let duration = start_bench.elapsed();
        let avg_recall = total_recall / (num_queries as f64);
        let avg_latency = duration.as_secs_f64() * 1000.0 / (num_queries as f64);
        let qps = (num_queries as f64) / duration.as_secs_f64();

        println!("| {:<2} | {:.4}    | {:.4}           | {:.0} |", ef, avg_recall, avg_latency, qps);
    }
}

// Helper to build tree
fn build_tree(depth: usize, branch_factor: usize, prefix: String) -> TreeNode {
    if depth == 0 {
        return TreeNode { id: prefix, children: Vec::new() };
    }
    
    let mut children = Vec::new();
    for i in 0..branch_factor {
        let child_prefix = format!("{}.{}", prefix, i);
        children.push(build_tree(depth - 1, branch_factor, child_prefix));
    }
    
    TreeNode { id: prefix, children }
}

// Exact search
fn brute_force_search(
    embedder: &HyperbolicEmbedder, 
    data: &[Array1<f64>], 
    query: &Array1<f64>, 
    k: usize
) -> Vec<usize> {
    let mut heap = BinaryHeap::new(); // Max heap of (dist, idx) to keep smallest k
    
    for (i, point) in data.iter().enumerate() {
        let dist = embedder.space.distance(&query.view(), &point.view());
        
        heap.push((OrderedFloat(dist), i));
        if heap.len() > k {
            heap.pop();
        }
    }
    
    let mut result = Vec::new();
    while let Some((_, idx)) = heap.pop() {
        result.push(idx);
    }
    // Result is in reverse order (largest dist first due to pop), but set intersection doesn't care
    result
}
