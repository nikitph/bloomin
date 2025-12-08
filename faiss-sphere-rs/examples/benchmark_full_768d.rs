//! Benchmark: Full 768D with Rust Optimizations (No Projection)
//!
//! Tests pure Rust performance on full-dimensional data to isolate
//! the impact of dimensional reduction vs other optimizations

use faiss_sphere::SphericalIndex;
use ndarray::{Array2, s};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::time::Instant;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(80));
    println!("Full 768D Benchmark: Rust Optimizations Without Projection");
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
    let k = 10;
    
    println!("Loaded {} documents, {} queries ({}D)", n_docs, n_queries, d);
    println!();
    
    // Build index with full 768D data
    println!("{}", "=".repeat(80));
    println!("Full 768D Search (Rust Optimized)");
    println!("{}", "=".repeat(80));
    
    let mut index = SphericalIndex::new(d);
    index.add(documents.clone())?;
    
    // Warm-up run
    let _ = index.search_parallel(&queries.slice(s![..10, ..]).to_owned(), k)?;
    
    // Benchmark
    let start = Instant::now();
    let (_, indices) = index.search_parallel(&queries, k)?;
    let time_full = start.elapsed();
    
    let memory_full = (n_docs * d * 4) as f32 / 1e6;
    
    println!("Search time: {:.2}ms ({:.3}ms per query)", 
        time_full.as_secs_f32() * 1000.0,
        time_full.as_secs_f32() * 1000.0 / n_queries as f32
    );
    println!("Memory: {:.1} MB", memory_full);
    println!("Throughput: {:.0} QPS", n_queries as f32 / time_full.as_secs_f32());
    println!();
    
    // Compare with baseline from previous benchmarks
    println!("{}", "=".repeat(80));
    println!("COMPARISON");
    println!("{}", "=".repeat(80));
    
    // Reference times from benchmark_wiki.rs
    let baseline_time_ms = 13.8; // Baseline from previous runs
    let projected_time_ms = 4.6; // FAISS-Sphere with 320D projection
    
    let rust_full_time_ms = time_full.as_secs_f32() * 1000.0 / n_queries as f32;
    
    println!("Configuration                    | Time/Query | Speedup vs Baseline");
    println!("---------------------------------|------------|-------------------");
    println!("Baseline (768D, Python-style)    | {:6.2}ms   | 1.00×", baseline_time_ms);
    println!("Rust Full (768D, optimized)      | {:6.2}ms   | {:.2}×", 
        rust_full_time_ms, 
        baseline_time_ms / rust_full_time_ms
    );
    println!("FAISS-Sphere (320D, projected)   | {:6.2}ms   | {:.2}×", 
        projected_time_ms,
        baseline_time_ms / projected_time_ms
    );
    println!();
    
    println!("{}", "=".repeat(80));
    println!("ANALYSIS");
    println!("{}", "=".repeat(80));
    
    let rust_speedup = baseline_time_ms / rust_full_time_ms;
    let projection_speedup = baseline_time_ms / projected_time_ms;
    let projection_benefit = rust_full_time_ms / projected_time_ms;
    
    println!("Rust optimizations alone:        {:.2}× speedup", rust_speedup);
    println!("Projection (320D) alone:          {:.2}× additional speedup", projection_benefit);
    println!("Combined (Rust + Projection):     {:.2}× total speedup", projection_speedup);
    println!();
    
    if rust_speedup > 1.5 {
        println!("✓ Rust optimizations provide significant benefit even without projection");
    } else {
        println!("✓ Most speedup comes from dimensional reduction, not Rust optimizations");
    }
    
    println!();
    println!("Key Rust optimizations applied:");
    println!("  • Pure inner product (k=1 spherical geometry)");
    println!("  • Rayon parallel search");
    println!("  • LLVM auto-vectorization (SIMD)");
    println!("  • Zero-copy memory handling");
    
    Ok(())
}
