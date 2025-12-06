//! Benchmark example for FAISS-Sphere
//!
//! Demonstrates 2-3× speedup with Rust implementation

use faiss_sphere::{IntrinsicProjector, SphericalIndex};
use ndarray::{Array2, s};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", "=".repeat(80));
    println!("FAISS-Sphere Rust Benchmark");
    println!("{}", "=".repeat(80));
    
    // Configuration
    let n_samples = 50_000;
    let n_queries = 100;
    let d_ambient = 768;
    let d_intrinsic = 320;
    let k = 10;
    
    println!("Samples: {}", n_samples);
    println!("Queries: {}", n_queries);
    println!("Ambient dimension: {}D", d_ambient);
    println!("Intrinsic dimension: {}D", d_intrinsic);
    println!("k: {}", k);
    println!();
    
    // Generate data with intrinsic structure
    println!("Generating data with {}D intrinsic structure...", 350);
    let data = generate_intrinsic_data(n_samples, d_ambient, 350);
    let queries = data.slice(s![..n_queries, ..]).to_owned();
    
    println!("✓ Generated {} embeddings", n_samples);
    println!();
    
    // Baseline: Full 768D search
    println!("{}", "=".repeat(80));
    println!("1. Baseline: Full {}D Search", d_ambient);
    println!("{}", "=".repeat(80));
    
    let mut index_baseline = SphericalIndex::new(d_ambient);
    index_baseline.add(data.clone())?;
    
    let start = Instant::now();
    let (_, indices_gt) = index_baseline.search_parallel(&queries, k)?;
    let time_baseline = start.elapsed();
    
    let memory_baseline = (n_samples * d_ambient * 4) as f32 / 1e6;
    
    println!("Search time: {:.2}ms ({:.3}ms per query)", 
        time_baseline.as_secs_f32() * 1000.0,
        time_baseline.as_secs_f32() * 1000.0 / n_queries as f32
    );
    println!("Memory: {:.1} MB", memory_baseline);
    println!();
    
    // Optimized: Intrinsic projection
    println!("{}", "=".repeat(80));
    println!("2. FAISS-Sphere: Intrinsic Projection ({}D → {}D)", d_ambient, d_intrinsic);
    println!("{}", "=".repeat(80));
    
    // Train projector
    let mut projector = IntrinsicProjector::new(d_ambient, d_intrinsic);
    let train_start = Instant::now();
    projector.train(&data.slice(s![..5000, ..]).to_owned())?;
    let train_time = train_start.elapsed();
    
    println!("Training time: {:.2}ms", train_time.as_secs_f32() * 1000.0);
    
    let stats = projector.stats();
    println!("Variance explained: {:.4}", stats.variance_explained);
    println!();
    
    // Project database
    println!("Projecting database...");
    let project_start = Instant::now();
    let data_projected = projector.project_parallel(&data)?;
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
    
    let memory_sphere = (n_samples * d_intrinsic * 4) as f32 / 1e6;
    
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
    println!("RESULTS");
    println!("{}", "=".repeat(80));
    
    let speedup = time_baseline.as_secs_f32() / time_sphere.as_secs_f32();
    let memory_reduction = memory_baseline / memory_sphere;
    
    println!("Speedup: {:.2}×", speedup);
    println!("Memory reduction: {:.2}×", memory_reduction);
    println!("Recall: {:.1}%", recall * 100.0);
    
    if speedup >= 2.0 {
        println!("\n✅ TARGET ACHIEVED: {:.2}× speedup!", speedup);
    } else if speedup >= 1.5 {
        println!("\n✅ GOOD PERFORMANCE: {:.2}× speedup", speedup);
    }
    
    Ok(())
}

fn generate_intrinsic_data(n: usize, d_ambient: usize, d_intrinsic: usize) -> Array2<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    // Generate in intrinsic space
    let intrinsic = Array2::from_shape_fn((n, d_intrinsic), |_| {
        rng.gen::<f32>() - 0.5
    });
    
    // Random projection to ambient space
    let projection = Array2::from_shape_fn((d_intrinsic, d_ambient), |_| {
        rng.gen::<f32>() / (d_intrinsic as f32).sqrt()
    });
    
    let ambient = intrinsic.dot(&projection);
    
    // Normalize
    faiss_sphere::utils::normalize_vectors(&ambient)
}

fn compute_recall(gt: &Array2<usize>, results: &Array2<usize>) -> f32 {
    faiss_sphere::utils::compute_recall(gt, results)
}
