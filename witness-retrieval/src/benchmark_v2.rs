//! Production Benchmark v2: Scaling Test
//!
//! Tests the full stack at 50K, 100K, 200K, 500K vectors
//! Compares: Baseline flat, Hierarchical v1, Production v2

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

use witness_retrieval::production::{ProductionWitnessLDPC, brute_force_batch_search};
use witness_retrieval::WitnessLDPCIndex;
use witness_retrieval::hierarchical::HierarchicalWitnessLDPC;

/// Generate clustered data - this is realistic for similarity search
/// Random uniform data is nearly orthogonal in high dimensions (curse of dimensionality)
fn generate_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n_clusters = 100; // 100 clusters
    let vectors_per_cluster = n / n_clusters;

    let mut vectors = Vec::with_capacity(n);

    for _ in 0..n_clusters {
        // Random cluster center
        let center: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 10.0 - 5.0).collect();

        // Samples around center with some noise
        for _ in 0..vectors_per_cluster {
            let sample: Vec<f32> = center
                .iter()
                .map(|&c| c + (rng.gen::<f32>() - 0.5) * 2.0)
                .collect();
            vectors.push(sample);
        }
    }

    // Fill remaining if n not divisible by n_clusters
    while vectors.len() < n {
        let cluster_idx = rng.gen_range(0..n_clusters);
        let center_vec = &vectors[cluster_idx * vectors_per_cluster];
        let sample: Vec<f32> = center_vec
            .iter()
            .map(|&c| c + (rng.gen::<f32>() - 0.5) * 2.0)
            .collect();
        vectors.push(sample);
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

/// Generate query vectors as perturbed versions of database vectors
fn generate_queries(vectors: &[Vec<f32>], n_queries: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = vectors.len();

    (0..n_queries)
        .map(|_| {
            let idx = rng.gen_range(0..n);
            let mut q = vectors[idx].clone();
            // Add small perturbation
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
        .collect()
}

fn compute_recall(predictions: &[(u32, f32)], ground_truth: &[(u32, f32)], k: usize) -> f64 {
    let pred_set: std::collections::HashSet<u32> = predictions.iter().take(k).map(|(id, _)| *id).collect();
    let gt_set: std::collections::HashSet<u32> = ground_truth.iter().take(k).map(|(id, _)| *id).collect();

    let intersection = pred_set.intersection(&gt_set).count();
    intersection as f64 / k as f64
}

fn benchmark_scale(n: usize, dim: usize, n_queries: usize, k: usize) {
    println!("\n{}", "=".repeat(70));
    println!("BENCHMARK: {} vectors (dim={})", n, dim);
    println!("{}", "=".repeat(70));

    // Generate data
    println!("Generating {} clustered vectors...", n);
    let vectors = generate_vectors(n, dim, 42);
    let queries = generate_queries(&vectors, n_queries, 123);

    // Ground truth
    println!("Computing ground truth ({} queries)...", n_queries);
    let gt_start = Instant::now();
    let ground_truth = brute_force_batch_search(&vectors, &queries, k);
    let gt_time = gt_start.elapsed();
    println!("  Brute force: {:.2}ms per query", gt_time.as_secs_f64() * 1000.0 / n_queries as f64);

    // === Baseline: Flat index (m=4096) ===
    println!("\n--- Baseline (flat, m=4096) ---");
    let mut baseline = WitnessLDPCIndex::new(dim, 4096, 4, 64, true);

    let build_start = Instant::now();
    baseline.add(vectors.clone(), true);
    let build_time = build_start.elapsed();
    println!("  Build time: {:.2}s", build_time.as_secs_f64());

    // Use 500 candidates like original benchmark
    let search_start = Instant::now();
    let baseline_results = baseline.batch_search(&queries, k, 500);
    let search_time = search_start.elapsed();

    let baseline_recall: f64 = queries.iter().enumerate()
        .map(|(i, _)| compute_recall(&baseline_results[i], &ground_truth[i], k))
        .sum::<f64>() / n_queries as f64;

    let baseline_speedup = gt_time.as_secs_f64() / search_time.as_secs_f64();
    println!("  Recall@{}: {:.1}%", k, baseline_recall * 100.0);
    println!("  Query time: {:.3}ms per query", search_time.as_secs_f64() * 1000.0 / n_queries as f64);
    println!("  Speedup: {:.1}x", baseline_speedup);
    println!("  Memory: {:.1} MB", baseline.memory_bytes() as f64 / 1024.0 / 1024.0);

    // === Hierarchical v1 ===
    println!("\n--- Hierarchical v1 ---");
    let mut hier = HierarchicalWitnessLDPC::new(dim);

    let build_start = Instant::now();
    hier.add(vectors.clone(), true);
    let build_time = build_start.elapsed();
    println!("  Build time: {:.2}s", build_time.as_secs_f64());

    // Use more candidates for better recall
    let search_start = Instant::now();
    let hier_results: Vec<Vec<(u32, f32)>> = queries.iter()
        .map(|q| hier.search(q, k, 2000))
        .collect();
    let search_time = search_start.elapsed();

    let hier_recall: f64 = queries.iter().enumerate()
        .map(|(i, _)| compute_recall(&hier_results[i], &ground_truth[i], k))
        .sum::<f64>() / n_queries as f64;

    let hier_speedup = gt_time.as_secs_f64() / search_time.as_secs_f64();
    println!("  Recall@{}: {:.1}%", k, hier_recall * 100.0);
    println!("  Query time: {:.3}ms per query", search_time.as_secs_f64() * 1000.0 / n_queries as f64);
    println!("  Speedup: {:.1}x", hier_speedup);
    println!("  Memory: {:.1} MB", hier.memory_bytes() as f64 / 1024.0 / 1024.0);

    // === Production v2 (without IDF) ===
    println!("\n--- Production v2 (no IDF) ---");
    let mut prod_no_idf = ProductionWitnessLDPC::new(dim);

    let build_start = Instant::now();
    prod_no_idf.add(vectors.clone(), true, false);
    let build_time = build_start.elapsed();
    println!("  Build time: {:.2}s", build_time.as_secs_f64());

    // Use more candidates for better recall
    // Coarse should return many, medium narrows, fine is precise
    let coarse_cand = 5000.min(n);
    let medium_cand = 1000.min(n);

    let search_start = Instant::now();
    let prod_results = prod_no_idf.batch_search(&queries, k, coarse_cand, medium_cand);
    let search_time = search_start.elapsed();

    let prod_recall: f64 = queries.iter().enumerate()
        .map(|(i, _)| compute_recall(&prod_results[i], &ground_truth[i], k))
        .sum::<f64>() / n_queries as f64;

    let prod_speedup = gt_time.as_secs_f64() / search_time.as_secs_f64();
    println!("  Recall@{}: {:.1}%", k, prod_recall * 100.0);
    println!("  Query time: {:.3}ms per query", search_time.as_secs_f64() * 1000.0 / n_queries as f64);
    println!("  Speedup: {:.1}x", prod_speedup);
    println!("  Memory: {:.1} MB", prod_no_idf.memory_bytes() as f64 / 1024.0 / 1024.0);

    // === Production v2 (with IDF) ===
    println!("\n--- Production v2 (with IDF) ---");
    let mut prod_idf = ProductionWitnessLDPC::new(dim);

    let build_start = Instant::now();
    prod_idf.add(vectors.clone(), true, true);
    let build_time = build_start.elapsed();
    println!("  Build time: {:.2}s", build_time.as_secs_f64());
    println!("{}", prod_idf.config_summary());

    let search_start = Instant::now();
    let prod_idf_results = prod_idf.batch_search(&queries, k, coarse_cand, medium_cand);
    let search_time = search_start.elapsed();

    let prod_idf_recall: f64 = queries.iter().enumerate()
        .map(|(i, _)| compute_recall(&prod_idf_results[i], &ground_truth[i], k))
        .sum::<f64>() / n_queries as f64;

    let prod_idf_speedup = gt_time.as_secs_f64() / search_time.as_secs_f64();
    println!("  Recall@{}: {:.1}%", k, prod_idf_recall * 100.0);
    println!("  Query time: {:.3}ms per query", search_time.as_secs_f64() * 1000.0 / n_queries as f64);
    println!("  Speedup: {:.1}x", prod_idf_speedup);

    // Summary
    println!("\n--- SUMMARY ---");
    println!("{:<25} {:>10} {:>12}", "Method", "Recall@10", "Speedup");
    println!("{:-<49}", "");
    println!("{:<25} {:>9.1}% {:>11.1}x", "Baseline (flat)", baseline_recall * 100.0, baseline_speedup);
    println!("{:<25} {:>9.1}% {:>11.1}x", "Hierarchical v1", hier_recall * 100.0, hier_speedup);
    println!("{:<25} {:>9.1}% {:>11.1}x", "Production v2 (no IDF)", prod_recall * 100.0, prod_speedup);
    println!("{:<25} {:>9.1}% {:>11.1}x", "Production v2 (with IDF)", prod_idf_recall * 100.0, prod_idf_speedup);
}

fn main() {
    println!("========================================");
    println!("Production Witness-LDPC v2 Benchmark");
    println!("========================================");

    let dim = 768;
    let n_queries = 100;
    let k = 10;

    // Test at different scales
    for n in [50_000, 100_000, 200_000] {
        benchmark_scale(n, dim, n_queries, k);
    }

    println!("\n{}", "=".repeat(70));
    println!("BENCHMARK COMPLETE");
    println!("{}", "=".repeat(70));
}
