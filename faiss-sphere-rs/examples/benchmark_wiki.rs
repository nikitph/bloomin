//! Wikipedia + BERT Benchmark for FAISS-Sphere (Rust)
//!
//! Benchmarks performance on real Wikipedia embeddings (10k docs, 768D)

use faiss_sphere::{IntrinsicProjector, SphericalIndex};
use ndarray::{Array2, s};
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::time::Instant;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(80));
    println!("FAISS-Sphere Rust Benchmark: Wikipedia + BERT 10k");
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
    
    // Baseline: Full 768D search
    println!("{}", "=".repeat(80));
    println!("1. Baseline: Full {}D Search", d_ambient);
    println!("{}", "=".repeat(80));
    
    let mut index_baseline = SphericalIndex::new(d_ambient);
    index_baseline.add(documents.clone())?;
    
    let start = Instant::now();
    let (_, indices_gt) = index_baseline.search_parallel(&queries, k)?;
    let time_baseline = start.elapsed();
    
    let memory_baseline = (n_docs * d_ambient * 4) as f32 / 1e6;
    
    println!("Search time: {:.2}ms ({:.3}ms per query)", 
        time_baseline.as_secs_f32() * 1000.0,
        time_baseline.as_secs_f32() * 1000.0 / n_queries as f32
    );
    println!("Memory: {:.1} MB", memory_baseline);
    println!();
    
    // Optimized: Intrinsic projection
    let d_intrinsic = 320; // Using 320D as determined optimal
    
    println!("{}", "=".repeat(80));
    println!("2. FAISS-Sphere: Intrinsic Projection ({}D → {}D)", d_ambient, d_intrinsic);
    println!("{}", "=".repeat(80));
    
    // Train projector
    let mut projector = IntrinsicProjector::new(d_ambient, d_intrinsic);
    let train_start = Instant::now();
    // Use first 2000 docs for training
    projector.train(&documents.slice(s![..2000, ..]).to_owned())?;
    let train_time = train_start.elapsed();
    
    println!("Training time: {:.2}ms", train_time.as_secs_f32() * 1000.0);
    
    let stats = projector.stats();
    println!("Variance explained: {:.4}", stats.variance_explained);
    println!();
    
    // Project database
    println!("Projecting database...");
    let project_start = Instant::now();
    let data_projected = projector.project_parallel(&documents)?;
    let project_time = project_start.elapsed();
    
    println!("Projection time: {:.2}ms", project_time.as_secs_f32() * 1000.0);
    
    // Build index
    let mut index_sphere = SphericalIndex::new(d_intrinsic);
    index_sphere.add(data_projected)?;
    
    // Search
    let start = Instant::now();
    let queries_projected = projector.project_parallel(&queries)?;
    let (_, indices_sphere) = index_sphere.search_parallel(&queries_projected, k)?;
    let time_sphere = start.elapsed();
    
    let memory_sphere = (n_docs * d_intrinsic * 4) as f32 / 1e6;
    
    println!("Search time: {:.2}ms ({:.3}ms per query)", 
        time_sphere.as_secs_f32() * 1000.0,
        time_sphere.as_secs_f32() * 1000.0 / n_queries as f32
    );
    println!("Memory: {:.1} MB", memory_sphere);
    
    // Compute recall
    let recall = compute_recall(&indices_gt, &indices_sphere);
    println!("Recall@{}: {:.3}", k, recall);
    println!();
    
    // Results
    println!("{}", "=".repeat(80));
    println!("RESULTS: Wikipedia + BERT 10k");
    println!("{}", "=".repeat(80));
    
    let speedup = time_baseline.as_secs_f32() / time_sphere.as_secs_f32();
    let memory_reduction = memory_baseline / memory_sphere;
    
    println!("Speedup: {:.2}×", speedup);
    println!("Memory reduction: {:.2}×", memory_reduction);
    println!("Recall: {:.1}%", recall * 100.0);
    
    Ok(())
}

fn compute_recall(gt: &Array2<usize>, results: &Array2<usize>) -> f32 {
    faiss_sphere::utils::compute_recall(gt, results)
}
