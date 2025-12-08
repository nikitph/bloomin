//! Baseline Benchmark for FAISS-Sphere (Rust)
//!
//! Benchmarks raw search performance (768D) without projection.
//! Compares Serial vs Parallel execution on 100k documents.

use faiss_sphere::SphericalIndex;
use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::time::Instant;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(80));
    println!("FAISS-Sphere Rust Benchmark: Baseline (No Projection)");
    println!("{}", "=".repeat(80));
    
    // Paths to data
    let data_dir = "/Users/truckx/PycharmProjects/bloomin/faiss-sphere-new/experiments/data";
    let doc_path = Path::new(data_dir).join("wikipedia_documents.npy");
    let query_path = Path::new(data_dir).join("wikipedia_queries.npy");
    
    println!("Loading data from: {}", data_dir);
    
    // Load data
    let documents = Array2::<f32>::read_npy(File::open(&doc_path)?)?;
    let queries = Array2::<f32>::read_npy(File::open(&query_path)?)?;
    
    let (n_docs, d_ambient) = documents.dim();
    let (n_queries, _) = queries.dim();
    let k = 10;
    
    println!("Loaded {} documents, {} queries ({}D)", n_docs, n_queries, d_ambient);
    println!();
    
    // Create index
    let mut index = SphericalIndex::new(d_ambient);
    index.add(documents.clone())?;
    
    // 1. Serial Search
    println!("{}", "=".repeat(80));
    println!("1. Serial Search (Single Thread)");
    println!("{}", "=".repeat(80));
    
    let start_serial = Instant::now();
    let _ = index.search(&queries, k)?;
    let time_serial = start_serial.elapsed();
    
    println!("Time: {:.2}ms ({:.3}ms per query)", 
        time_serial.as_secs_f32() * 1000.0,
        time_serial.as_secs_f32() * 1000.0 / n_queries as f32
    );
    println!();
    
    // 2. Parallel Search
    println!("{}", "=".repeat(80));
    println!("2. Parallel Search (Rayon)");
    println!("{}", "=".repeat(80));
    
    let start_parallel = Instant::now();
    let _ = index.search_parallel(&queries, k)?;
    let time_parallel = start_parallel.elapsed();
    
    println!("Time: {:.2}ms ({:.3}ms per query)", 
        time_parallel.as_secs_f32() * 1000.0,
        time_parallel.as_secs_f32() * 1000.0 / n_queries as f32
    );
    println!();
    
    // Results
    println!("{}", "=".repeat(80));
    println!("RESULTS: Baseline Performance (100k Docs, 768D)");
    println!("{}", "=".repeat(80));
    
    let speedup = time_serial.as_secs_f32() / time_parallel.as_secs_f32();
    
    println!("Serial Time:   {:.2}ms", time_serial.as_secs_f32() * 1000.0);
    println!("Parallel Time: {:.2}ms", time_parallel.as_secs_f32() * 1000.0);
    println!("Parallel Speedup: {:.2}×", speedup);
    
    Ok(())
}
