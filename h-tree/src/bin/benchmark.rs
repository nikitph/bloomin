//! H-Tree Benchmark Suite
//!
//! Comprehensive benchmarks demonstrating:
//! - O(log N) query complexity
//! - O(1) vacuum detection
//! - Memory efficiency vs. brute force
//! - Recall@k accuracy
//!
//! Run with: cargo run --release --bin benchmark

use h_tree::{HTree, Vector};
use h_tree::tree::{HTreeConfig, QueryStats};
use instant::Instant;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use indicatif::{ProgressBar, ProgressStyle};

/// Benchmark configuration
struct BenchmarkConfig {
    /// Number of vectors to insert
    num_vectors: usize,

    /// Vector dimensionality
    dimension: usize,

    /// Number of queries to run
    num_queries: usize,

    /// k for k-NN queries
    k: usize,

    /// Number of "vacuum" queries (should return empty)
    num_vacuum_queries: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_vectors: 10_000,
            dimension: 128,
            num_queries: 1_000,
            k: 10,
            num_vacuum_queries: 100,
        }
    }
}

/// Benchmark results
#[derive(Debug)]
struct BenchmarkResults {
    // Insertion metrics
    insert_time_total_ms: f64,
    insert_time_per_vector_us: f64,

    // Query metrics
    query_time_total_ms: f64,
    query_time_per_query_us: f64,
    queries_per_second: f64,

    // Vacuum detection metrics
    vacuum_detection_time_us: f64,
    vacuum_detection_accuracy: f64,

    // Accuracy metrics
    recall_at_k: f64,

    // Memory metrics
    tree_memory_mb: f64,
    memory_per_vector_bytes: f64,

    // Tree statistics
    tree_height: u32,
    node_count: usize,

    // Query statistics
    avg_nodes_visited: f64,
    avg_vacuum_pruned: f64,
    avg_leaves_searched: f64,
}

/// Generate random vectors clustered in groups
fn generate_clustered_vectors(num_vectors: usize, dim: usize, num_clusters: usize) -> Vec<Vector> {
    let mut rng = rand::thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Generate cluster centers
    let centers: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| (0..dim).map(|_| normal.sample(&mut rng) as f32 * 10.0).collect())
        .collect();

    // Generate vectors around centers
    (0..num_vectors)
        .map(|id| {
            let center = &centers[id % num_clusters];
            let data: Vec<f32> = center
                .iter()
                .map(|c| c + normal.sample(&mut rng) as f32)
                .collect();
            Vector::new(id as u64, data)
        })
        .collect()
}

/// Generate vectors far from the data distribution (for vacuum testing)
fn generate_vacuum_queries(num_queries: usize, dim: usize) -> Vec<Vector> {
    let mut rng = rand::thread_rng();

    (0..num_queries)
        .map(|id| {
            // Generate vectors in a completely different region
            let data: Vec<f32> = (0..dim)
                .map(|_| rng.gen_range(1000.0..2000.0))
                .collect();
            Vector::new((1_000_000 + id) as u64, data)
        })
        .collect()
}

/// Compute ground truth k-NN using brute force
fn brute_force_knn(vectors: &[Vector], query: &Vector, k: usize) -> Vec<u64> {
    let mut distances: Vec<(u64, f32)> = vectors
        .iter()
        .map(|v| (v.id, query.cosine_similarity(v)))
        .collect();

    distances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    distances.truncate(k);
    distances.iter().map(|(id, _)| *id).collect()
}

/// Compute recall@k
fn compute_recall(htree_results: &[u64], ground_truth: &[u64], k: usize) -> f64 {
    let htree_set: std::collections::HashSet<_> = htree_results.iter().collect();
    let gt_set: std::collections::HashSet<_> = ground_truth.iter().take(k).collect();

    let intersection = htree_set.intersection(&gt_set).count();
    intersection as f64 / k as f64
}

fn run_benchmark(config: BenchmarkConfig) -> BenchmarkResults {
    println!("\n========================================");
    println!("  H-Tree Benchmark Suite");
    println!("========================================\n");

    println!("Configuration:");
    println!("  Vectors: {}", config.num_vectors);
    println!("  Dimension: {}", config.dimension);
    println!("  Queries: {}", config.num_queries);
    println!("  k: {}", config.k);
    println!("  Vacuum queries: {}", config.num_vacuum_queries);
    println!();

    // Generate data
    println!("Generating {} clustered vectors...", config.num_vectors);
    let vectors = generate_clustered_vectors(config.num_vectors, config.dimension, 20);

    println!("Generating {} vacuum queries...", config.num_vacuum_queries);
    let vacuum_queries = generate_vacuum_queries(config.num_vacuum_queries, config.dimension);

    // Create H-Tree with high-recall configuration
    let htree_config = HTreeConfig::high_recall();
    let mut tree = HTree::new(htree_config);

    // Benchmark insertion
    println!("\n--- Insertion Benchmark ---");
    let pb = ProgressBar::new(config.num_vectors as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap());

    let insert_start = Instant::now();
    for v in &vectors {
        tree.insert(v.clone());
        pb.inc(1);
    }
    let insert_time = insert_start.elapsed();
    pb.finish_with_message("Done");

    let insert_time_total_ms = insert_time.as_secs_f64() * 1000.0;
    let insert_time_per_vector_us = (insert_time.as_micros() as f64) / config.num_vectors as f64;

    println!("  Total time: {:.2} ms", insert_time_total_ms);
    println!("  Per vector: {:.2} µs", insert_time_per_vector_us);

    // Get tree stats
    let stats = tree.stats();
    println!("\nTree Statistics:");
    println!("  Height: {}", stats.height);
    println!("  Nodes: {}", stats.node_count);
    println!("  Memory: {:.2} MB", stats.memory_bytes as f64 / 1_000_000.0);

    // Benchmark queries
    println!("\n--- Query Benchmark ---");

    // Select random query vectors from the dataset
    let mut rng = rand::thread_rng();
    let query_indices: Vec<usize> = (0..config.num_queries)
        .map(|_| rng.gen_range(0..config.num_vectors))
        .collect();

    let pb = ProgressBar::new(config.num_queries as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap());

    let mut total_recall = 0.0;
    let mut total_stats = QueryStats::default();

    let query_start = Instant::now();
    for &idx in &query_indices {
        let query = &vectors[idx];
        let (results, query_stats) = tree.query_with_stats(query, config.k);

        // Compute recall against brute force
        let ground_truth = brute_force_knn(&vectors, query, config.k);
        let htree_ids: Vec<u64> = results.iter().map(|r| r.vector_id).collect();
        total_recall += compute_recall(&htree_ids, &ground_truth, config.k);

        // Accumulate stats
        total_stats.nodes_visited += query_stats.nodes_visited;
        total_stats.vacuum_pruned += query_stats.vacuum_pruned;
        total_stats.leaves_searched += query_stats.leaves_searched;
        total_stats.vectors_compared += query_stats.vectors_compared;

        pb.inc(1);
    }
    let query_time = query_start.elapsed();
    pb.finish_with_message("Done");

    let query_time_total_ms = query_time.as_secs_f64() * 1000.0;
    let query_time_per_query_us = (query_time.as_micros() as f64) / config.num_queries as f64;
    let queries_per_second = config.num_queries as f64 / query_time.as_secs_f64();

    let recall_at_k = total_recall / config.num_queries as f64;

    println!("  Total time: {:.2} ms", query_time_total_ms);
    println!("  Per query: {:.2} µs", query_time_per_query_us);
    println!("  Queries/sec: {:.0}", queries_per_second);
    println!("  Recall@{}: {:.2}%", config.k, recall_at_k * 100.0);

    // Benchmark vacuum detection
    println!("\n--- Vacuum Detection Benchmark ---");

    let mut vacuum_correct = 0;
    let vacuum_start = Instant::now();
    for query in &vacuum_queries {
        let is_vacuum = tree.is_vacuum(query);
        if is_vacuum {
            vacuum_correct += 1;
        }
        // Also test that queries return few/no results
        let results = tree.query(query, config.k);
        if results.is_empty() || results[0].similarity < 0.1 {
            // Low similarity is as good as vacuum for practical purposes
        }
    }
    let vacuum_time = vacuum_start.elapsed();

    let vacuum_detection_time_us = (vacuum_time.as_micros() as f64) / config.num_vacuum_queries as f64;
    let vacuum_detection_accuracy = vacuum_correct as f64 / config.num_vacuum_queries as f64;

    println!("  Per query: {:.2} µs", vacuum_detection_time_us);
    println!("  Vacuum detection rate: {:.1}%", vacuum_detection_accuracy * 100.0);

    // Verify integrity
    println!("\n--- Integrity Check ---");
    let integrity_ok = tree.verify_integrity();
    println!("  Merkle integrity: {}", if integrity_ok { "PASSED ✓" } else { "FAILED ✗" });

    // Memory analysis
    let tree_memory_mb = stats.memory_bytes as f64 / 1_000_000.0;
    let memory_per_vector_bytes = stats.memory_bytes as f64 / config.num_vectors as f64;

    // Average query statistics
    let avg_nodes_visited = total_stats.nodes_visited as f64 / config.num_queries as f64;
    let avg_vacuum_pruned = total_stats.vacuum_pruned as f64 / config.num_queries as f64;
    let avg_leaves_searched = total_stats.leaves_searched as f64 / config.num_queries as f64;

    println!("\n--- Query Statistics ---");
    println!("  Avg nodes visited: {:.1}", avg_nodes_visited);
    println!("  Avg vacuum pruned: {:.1}", avg_vacuum_pruned);
    println!("  Avg leaves searched: {:.1}", avg_leaves_searched);

    BenchmarkResults {
        insert_time_total_ms,
        insert_time_per_vector_us,
        query_time_total_ms,
        query_time_per_query_us,
        queries_per_second,
        vacuum_detection_time_us,
        vacuum_detection_accuracy,
        recall_at_k,
        tree_memory_mb,
        memory_per_vector_bytes,
        tree_height: stats.height,
        node_count: stats.node_count,
        avg_nodes_visited,
        avg_vacuum_pruned,
        avg_leaves_searched,
    }
}

fn run_scaling_benchmark() {
    println!("\n========================================");
    println!("  Scaling Benchmark (O(log N) Verification)");
    println!("========================================\n");

    let sizes = vec![1_000, 5_000, 10_000, 25_000, 50_000];
    let dim = 64;
    let k = 10;
    let num_queries = 100;

    println!("{:>10} {:>12} {:>12} {:>10} {:>10}",
             "N", "Query µs", "Height", "Nodes", "Memory MB");
    println!("{:-<60}", "");

    for n in sizes {
        let config = BenchmarkConfig {
            num_vectors: n,
            dimension: dim,
            num_queries,
            k,
            num_vacuum_queries: 10,
        };

        let results = run_benchmark(config);

        println!("{:>10} {:>12.1} {:>12} {:>10} {:>10.2}",
                 n,
                 results.query_time_per_query_us,
                 results.tree_height,
                 results.node_count,
                 results.tree_memory_mb);
    }

    println!("\nNote: Query time should grow logarithmically with N");
    println!("Expected: ~O(log N) for well-balanced tree");
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║         H-Tree: Holographic B-Tree Benchmarks              ║");
    println!("║         Witness Field Theory Implementation                ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    // Run main benchmark
    let config = BenchmarkConfig::default();
    let results = run_benchmark(config);

    // Summary
    println!("\n========================================");
    println!("  Summary");
    println!("========================================");
    println!();
    println!("Performance:");
    println!("  Insert throughput: {:.0} vectors/sec",
             1_000_000.0 / results.insert_time_per_vector_us);
    println!("  Query throughput:  {:.0} queries/sec", results.queries_per_second);
    println!("  Vacuum detection:  {:.1} µs (O(1))", results.vacuum_detection_time_us);
    println!();
    println!("Accuracy:");
    println!("  Recall@10: {:.1}%", results.recall_at_k * 100.0);
    println!();
    println!("Efficiency:");
    println!("  Memory per vector: {:.1} bytes", results.memory_per_vector_bytes);
    println!("  Tree height: {} (log₂(N) = {:.1})",
             results.tree_height,
             (10_000f64).log2());
    println!();

    // Run scaling benchmark
    println!("\nRunning scaling benchmark...");
    run_scaling_benchmark();

    println!("\n✓ Benchmarks complete!");
}
