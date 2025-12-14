//! Benchmark: Witness-LDPC vs Brute Force
//!
//! This is the honest comparison - Rust vs Rust.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;
use witness_retrieval::{brute_force_batch_search, WitnessLDPCIndex};
use witness_retrieval::hierarchical::HierarchicalWitnessLDPC;

fn generate_clustered_data(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let vectors_per_cluster = n / n_clusters;

    let mut vectors = Vec::with_capacity(n);

    for _ in 0..n_clusters {
        // Random cluster center
        let center: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 10.0 - 5.0).collect();

        // Samples around center
        for _ in 0..vectors_per_cluster {
            let sample: Vec<f32> = center
                .iter()
                .map(|&c| c + (rng.gen::<f32>() - 0.5) * 2.0)
                .collect();
            vectors.push(sample);
        }
    }

    // Normalize all vectors
    for v in &mut vectors {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
    }

    vectors
}

fn compute_recall(retrieved: &[Vec<(u32, f32)>], ground_truth: &[Vec<(u32, f32)>], k: usize) -> f64 {
    let mut total_recall = 0.0;

    for (ret, gt) in retrieved.iter().zip(ground_truth.iter()) {
        let gt_set: std::collections::HashSet<u32> =
            gt.iter().take(k).map(|(id, _)| *id).collect();
        let ret_set: std::collections::HashSet<u32> =
            ret.iter().take(k).map(|(id, _)| *id).collect();

        let overlap = gt_set.intersection(&ret_set).count();
        total_recall += overlap as f64 / k as f64;
    }

    total_recall / retrieved.len() as f64
}

#[derive(Debug)]
struct BenchmarkResult {
    method: String,
    n_vectors: usize,
    dim: usize,
    build_time_ms: f64,
    memory_mb: f64,
    avg_query_time_us: f64,
    p50_query_time_us: f64,
    p99_query_time_us: f64,
    recall_at_1: f64,
    recall_at_10: f64,
    recall_at_100: f64,
}

fn benchmark_brute_force(
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground_truth: &[Vec<(u32, f32)>],
) -> BenchmarkResult {
    println!("\n[Brute Force]");

    let n = vectors.len();
    let dim = vectors[0].len();
    let n_queries = queries.len();

    // Memory: just vectors
    let memory_mb = (n * dim * 4) as f64 / 1024.0 / 1024.0;

    // Query timing
    println!("  Running {} queries...", n_queries);
    let mut query_times = Vec::with_capacity(n_queries);
    let mut all_results = Vec::with_capacity(n_queries);

    for query in queries {
        let start = Instant::now();
        let results = witness_retrieval::brute_force_search(vectors, query, 100);
        query_times.push(start.elapsed().as_micros() as f64);
        all_results.push(results);
    }

    query_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let recall_1 = compute_recall(&all_results, ground_truth, 1);
    let recall_10 = compute_recall(&all_results, ground_truth, 10);
    let recall_100 = compute_recall(&all_results, ground_truth, 100);

    let result = BenchmarkResult {
        method: "Brute Force".to_string(),
        n_vectors: n,
        dim,
        build_time_ms: 0.0,
        memory_mb,
        avg_query_time_us: query_times.iter().sum::<f64>() / n_queries as f64,
        p50_query_time_us: query_times[n_queries / 2],
        p99_query_time_us: query_times[(n_queries as f64 * 0.99) as usize],
        recall_at_1: recall_1,
        recall_at_10: recall_10,
        recall_at_100: recall_100,
    };

    println!(
        "  Memory: {:.1}MB, Query: avg={:.0}us, p50={:.0}us, p99={:.0}us",
        result.memory_mb, result.avg_query_time_us, result.p50_query_time_us, result.p99_query_time_us
    );
    println!(
        "  Recall: @1={:.3}, @10={:.3}, @100={:.3}",
        result.recall_at_1, result.recall_at_10, result.recall_at_100
    );

    result
}

fn benchmark_hierarchical(
    vectors: Vec<Vec<f32>>,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<(u32, f32)>],
) -> BenchmarkResult {
    println!("\n[Hierarchical Witness-LDPC] coarse=512bits, fine=2048bits");

    let n = vectors.len();
    let dim = vectors[0].len();
    let n_queries = queries.len();

    // Build index
    let mut index = HierarchicalWitnessLDPC::new(dim);

    let start = Instant::now();
    index.add(vectors, true);
    let build_time_ms = start.elapsed().as_millis() as f64;

    let memory_mb = index.memory_bytes() as f64 / 1024.0 / 1024.0;

    // Query timing
    println!("  Running {} queries...", n_queries);
    let max_coarse_candidates = 2000.min(n);  // More candidates for better recall
    let mut query_times = Vec::with_capacity(n_queries);
    let mut all_results = Vec::with_capacity(n_queries);

    for query in queries {
        let start = Instant::now();
        let results = index.search(query, 100, max_coarse_candidates);
        query_times.push(start.elapsed().as_micros() as f64);
        all_results.push(results);
    }

    query_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let recall_1 = compute_recall(&all_results, ground_truth, 1);
    let recall_10 = compute_recall(&all_results, ground_truth, 10);
    let recall_100 = compute_recall(&all_results, ground_truth, 100);

    let result = BenchmarkResult {
        method: "Hierarchical(c=512,f=2048)".to_string(),
        n_vectors: n,
        dim,
        build_time_ms,
        memory_mb,
        avg_query_time_us: query_times.iter().sum::<f64>() / n_queries as f64,
        p50_query_time_us: query_times[n_queries / 2],
        p99_query_time_us: query_times[(n_queries as f64 * 0.99) as usize],
        recall_at_1: recall_1,
        recall_at_10: recall_10,
        recall_at_100: recall_100,
    };

    println!(
        "  Build: {:.0}ms, Memory: {:.1}MB",
        result.build_time_ms, result.memory_mb
    );
    println!(
        "  Query: avg={:.0}us, p50={:.0}us, p99={:.0}us",
        result.avg_query_time_us, result.p50_query_time_us, result.p99_query_time_us
    );
    println!(
        "  Recall: @1={:.3}, @10={:.3}, @100={:.3}",
        result.recall_at_1, result.recall_at_10, result.recall_at_100
    );

    result
}

fn benchmark_witness_ldpc(
    vectors: Vec<Vec<f32>>,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<(u32, f32)>],
    code_length: usize,
    num_hashes: usize,
    num_witnesses: usize,
    store_vectors: bool,
) -> BenchmarkResult {
    println!(
        "\n[Witness-LDPC] code_length={}, num_hashes={}, num_witnesses={}, store_vectors={}",
        code_length, num_hashes, num_witnesses, store_vectors
    );

    let n = vectors.len();
    let dim = vectors[0].len();
    let n_queries = queries.len();

    // Build index
    let mut index = WitnessLDPCIndex::new(dim, code_length, num_hashes, num_witnesses, true);

    let start = Instant::now();
    index.add(vectors, store_vectors);
    let build_time_ms = start.elapsed().as_millis() as f64;

    let memory_mb = index.memory_bytes() as f64 / 1024.0 / 1024.0;

    // Query timing
    println!("  Running {} queries...", n_queries);
    let n_candidates = 500.min(n);  // More candidates for better recall
    let mut query_times = Vec::with_capacity(n_queries);
    let mut all_results = Vec::with_capacity(n_queries);

    for query in queries {
        let start = Instant::now();
        let results = index.search(query, 100, n_candidates);
        query_times.push(start.elapsed().as_micros() as f64);
        all_results.push(results);
    }

    query_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let recall_1 = compute_recall(&all_results, ground_truth, 1);
    let recall_10 = compute_recall(&all_results, ground_truth, 10);
    let recall_100 = compute_recall(&all_results, ground_truth, 100);

    let result = BenchmarkResult {
        method: format!(
            "Witness-LDPC(m={},K={},w={}{})",
            code_length,
            num_hashes,
            num_witnesses,
            if store_vectors { "" } else { ",compact" }
        ),
        n_vectors: n,
        dim,
        build_time_ms,
        memory_mb,
        avg_query_time_us: query_times.iter().sum::<f64>() / n_queries as f64,
        p50_query_time_us: query_times[n_queries / 2],
        p99_query_time_us: query_times[(n_queries as f64 * 0.99) as usize],
        recall_at_1: recall_1,
        recall_at_10: recall_10,
        recall_at_100: recall_100,
    };

    println!(
        "  Build: {:.0}ms, Memory: {:.1}MB",
        result.build_time_ms, result.memory_mb
    );
    println!(
        "  Query: avg={:.0}us, p50={:.0}us, p99={:.0}us",
        result.avg_query_time_us, result.p50_query_time_us, result.p99_query_time_us
    );
    println!(
        "  Recall: @1={:.3}, @10={:.3}, @100={:.3}",
        result.recall_at_1, result.recall_at_10, result.recall_at_100
    );

    result
}

fn print_summary(results: &[BenchmarkResult]) {
    println!("\n{}", "=".repeat(120));
    println!("BENCHMARK SUMMARY");
    println!("{}", "=".repeat(120));

    println!(
        "\n{:<45} {:>12} {:>15} {:>12} {:>12} {:>12}",
        "Method", "Memory (MB)", "Query (us)", "Recall@1", "Recall@10", "Recall@100"
    );
    println!("{}", "-".repeat(120));

    for r in results {
        println!(
            "{:<45} {:>12.1} {:>15.1} {:>12.3} {:>12.3} {:>12.3}",
            r.method, r.memory_mb, r.avg_query_time_us, r.recall_at_1, r.recall_at_10, r.recall_at_100
        );
    }

    println!("{}", "-".repeat(120));

    // Compute speedups vs brute force
    if let Some(brute) = results.iter().find(|r| r.method == "Brute Force") {
        println!("\nSpeedups vs Brute Force:");
        for r in results {
            if r.method == "Brute Force" {
                continue;
            }
            let speedup = brute.avg_query_time_us / r.avg_query_time_us;
            let memory_ratio = brute.memory_mb / r.memory_mb;
            println!(
                "  {}: {:.1}x faster, {:.1}x memory ratio (recall@10={:.1}%)",
                r.method,
                speedup,
                memory_ratio,
                r.recall_at_10 * 100.0
            );
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let n_vectors: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let dim: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(768);
    let n_queries: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(500);

    println!("{}", "=".repeat(120));
    println!("WITNESS-LDPC BENCHMARK (Rust)");
    println!("{}", "=".repeat(120));
    println!("\nConfiguration:");
    println!("  - Vectors: {}", n_vectors);
    println!("  - Dimensions: {}", dim);
    println!("  - Queries: {}", n_queries);

    // Generate data
    println!("\nGenerating clustered data...");
    let vectors = generate_clustered_data(n_vectors, dim, 100, 42);

    // Generate queries (perturbed versions of random vectors)
    let mut rng = ChaCha8Rng::seed_from_u64(123);
    let query_indices: Vec<usize> = (0..n_queries)
        .map(|_| rng.gen_range(0..n_vectors))
        .collect();

    let queries: Vec<Vec<f32>> = query_indices
        .iter()
        .map(|&i| {
            let mut q = vectors[i].clone();
            for x in &mut q {
                *x += (rng.gen::<f32>() - 0.5) * 0.2;
            }
            // Normalize
            let norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut q {
                *x /= norm;
            }
            q
        })
        .collect();

    // Compute ground truth
    println!("\nComputing ground truth (brute force)...");
    let ground_truth = brute_force_batch_search(&vectors, &queries, 100);

    // Run benchmarks
    let mut results = Vec::new();

    // Brute force baseline
    results.push(benchmark_brute_force(&vectors, &queries, &ground_truth));

    // Hierarchical Witness-LDPC (the fast one!)
    results.push(benchmark_hierarchical(vectors.clone(), &queries, &ground_truth));

    // Witness-LDPC variants - finding the sweet spot
    // Small/fast config
    results.push(benchmark_witness_ldpc(
        vectors.clone(),
        &queries,
        &ground_truth,
        1024,
        4,
        32,
        true,
    ));

    // Medium config - the sweet spot
    results.push(benchmark_witness_ldpc(
        vectors.clone(),
        &queries,
        &ground_truth,
        2048,
        4,
        64,
        true,
    ));

    // Higher recall config
    results.push(benchmark_witness_ldpc(
        vectors.clone(),
        &queries,
        &ground_truth,
        2048,
        6,
        96,
        true,
    ));

    // Max recall (but slower)
    results.push(benchmark_witness_ldpc(
        vectors.clone(),
        &queries,
        &ground_truth,
        4096,
        4,
        64,
        true,
    ));

    // Summary
    print_summary(&results);
}
