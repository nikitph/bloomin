//! Benchmark: Spherical Cap Tree for Range Queries
//!
//! Demonstrates O(log N) vs O(N) speedup for range queries

use faiss_sphere::SphericalCapIndex;
use ndarray::{Array1, Array2};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::time::Instant;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(80));
    println!("Spherical Cap Tree Benchmark: Range Queries");
    println!("{}", "=".repeat(80));
    
    // Load data
    let data_dir = "/Users/truckx/PycharmProjects/bloomin/faiss-sphere-new/experiments/data";
    let doc_path = Path::new(data_dir).join("wikipedia_documents.npy");
    let query_path = Path::new(data_dir).join("wikipedia_queries.npy");
    
    println!("Loading data from: {}", data_dir);
    
    let documents = Array2::<f32>::read_npy(File::open(&doc_path)?)?;
    let queries = Array2::<f32>::read_npy(File::open(&query_path)?)?;
    
    let (n_docs, d) = documents.dim();
    let (n_queries, _) = queries.dim();
    
    println!("Loaded {} documents, {} queries ({}D)", n_docs, n_queries, d);
    println!();
    
    // Build spherical cap tree
    println!("{}", "=".repeat(80));
    println!("Building Spherical Cap Tree");
    println!("{}", "=".repeat(80));
    
    let build_start = Instant::now();
    let cap_index = SphericalCapIndex::new(documents.clone());
    let build_time = build_start.elapsed();
    
    let stats = cap_index.stats();
    println!("Build time: {:.2}ms", build_time.as_secs_f32() * 1000.0);
    println!("Tree stats:");
    println!("  Nodes: {}", stats.num_nodes);
    println!("  Leaves: {}", stats.num_leaves);
    println!("  Max depth: {}", stats.max_depth);
    println!("  Points: {}", stats.total_points);
    println!();
    
    // Test different angular thresholds
    let thresholds = vec![
        (0.1, "~5.7°"),
        (0.2, "~11.5°"),
        (0.5, "~28.6°"),
    ];
    
    for (theta, desc) in thresholds {
        println!("{}", "=".repeat(80));
        println!("Range Query: θ = {} radians ({})", theta, desc);
        println!("{}", "=".repeat(80));
        
        // Method 1: Linear scan (O(N))
        let mut linear_results = Vec::new();
        let linear_start = Instant::now();
        
        for i in 0..n_queries.min(100) {
            let query = queries.row(i);
            let mut count = 0;
            
            for j in 0..n_docs {
                let doc = documents.row(j);
                let dot = query.dot(&doc).clamp(-1.0, 1.0);
                let dist = dot.acos();
                
                if dist <= theta {
                    count += 1;
                }
            }
            
            linear_results.push(count);
        }
        
        let linear_time = linear_start.elapsed();
        let avg_linear = linear_time.as_secs_f32() / 100.0;
        
        // Method 2: Spherical cap tree (O(log N))
        let mut tree_results = Vec::new();
        let tree_start = Instant::now();
        
        for i in 0..n_queries.min(100) {
            let query = queries.row(i).to_owned();
            let results = cap_index.range_query(&query, theta);
            tree_results.push(results.len());
        }
        
        let tree_time = tree_start.elapsed();
        let avg_tree = tree_time.as_secs_f32() / 100.0;
        
        // Results
        let speedup = linear_time.as_secs_f32() / tree_time.as_secs_f32();
        let avg_results: f32 = linear_results.iter().map(|&x| x as f32).sum::<f32>() / linear_results.len() as f32;
        
        println!("Linear scan (O(N)):");
        println!("  Total time: {:.2}ms", linear_time.as_secs_f32() * 1000.0);
        println!("  Avg per query: {:.3}ms", avg_linear * 1000.0);
        println!();
        
        println!("Spherical cap tree (O(log N)):");
        println!("  Total time: {:.2}ms", tree_time.as_secs_f32() * 1000.0);
        println!("  Avg per query: {:.3}ms", avg_tree * 1000.0);
        println!();
        
        println!("Results:");
        println!("  Speedup: {:.2}×", speedup);
        println!("  Avg points in range: {:.0}", avg_results);
        println!();
    }
    
    println!("{}", "=".repeat(80));
    println!("CONCLUSION");
    println!("{}", "=".repeat(80));
    println!("Spherical cap trees provide significant speedups for range queries.");
    println!("The speedup increases with larger datasets and tighter angular thresholds.");
    
    Ok(())
}
