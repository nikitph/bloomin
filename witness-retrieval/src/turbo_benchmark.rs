//! Turbo Benchmark: Testing All Optimizations
//!
//! Tests the turbo index with:
//! - SIMD-friendly Hamming distance
//! - Early termination
//! - Cache-optimized access patterns

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;
use std::time::Instant;

use witness_retrieval::turbo_index::{TurboWitnessIndex, brute_force_batch};
use witness_retrieval::WitnessLDPCIndex;

fn generate_clustered_data(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let vectors_per_cluster = n / n_clusters;
    let mut vectors = Vec::with_capacity(n);

    for _ in 0..n_clusters {
        let center: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 10.0 - 5.0).collect();
        for _ in 0..vectors_per_cluster {
            let sample: Vec<f32> = center.iter()
                .map(|&c| c + (rng.gen::<f32>() - 0.5) * 2.0)
                .collect();
            vectors.push(sample);
        }
    }

    while vectors.len() < n {
        let idx = rng.gen_range(0..vectors.len());
        let sample: Vec<f32> = vectors[idx].iter()
            .map(|&c| c + (rng.gen::<f32>() - 0.5) * 2.0)
            .collect();
        vectors.push(sample);
    }

    for v in &mut vectors {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for x in v.iter_mut() { *x /= norm; }
        }
    }
    vectors
}

fn generate_queries(vectors: &[Vec<f32>], n_queries: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = vectors.len();
    (0..n_queries).map(|_| {
        let idx = rng.gen_range(0..n);
        let mut q = vectors[idx].clone();
        for x in &mut q { *x += (rng.gen::<f32>() - 0.5) * 0.2; }
        let norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        for x in &mut q { *x /= norm; }
        q
    }).collect()
}

fn compute_recall(pred: &[(u32, f32)], gt: &[(u32, f32)], k: usize) -> f64 {
    let pred_set: HashSet<u32> = pred.iter().take(k).map(|(id, _)| *id).collect();
    let gt_set: HashSet<u32> = gt.iter().take(k).map(|(id, _)| *id).collect();
    pred_set.intersection(&gt_set).count() as f64 / k as f64
}

fn main() {
    println!("{}", "=".repeat(80));
    println!("TURBO BENCHMARK: All Optimizations Combined");
    println!("SIMD + Early Termination + Cache Optimization");
    println!("{}", "=".repeat(80));

    let dim = 768;
    let n_queries = 100;
    let k = 10;

    let scales = vec![50_000, 100_000, 200_000, 500_000];

    // Collect brute force times
    println!("\n{}", "=".repeat(80));
    println!("BRUTE FORCE BASELINE");
    println!("{}", "=".repeat(80));

    let mut brute_times = vec![];
    let mut ground_truths = vec![];

    for &n in &scales {
        let vectors = generate_clustered_data(n, dim, 100, 42);
        let queries = generate_queries(&vectors, n_queries, 123);

        let start = Instant::now();
        let gt = brute_force_batch(&vectors, &queries, k);
        let elapsed = start.elapsed().as_secs_f64();
        let query_time = elapsed * 1000.0 / n_queries as f64;
        let qps = n_queries as f64 / elapsed;

        println!("N={}: {:.2}ms/query, {:.0} QPS", n, query_time, qps);
        brute_times.push((n, query_time, vectors, queries));
        ground_truths.push(gt);
    }

    // Test original implementation
    println!("\n{}", "=".repeat(80));
    println!("ORIGINAL IMPLEMENTATION (baseline comparison)");
    println!("{}", "=".repeat(80));

    println!("\n{:<10} {:>8} {:>8} {:>12} {:>10} {:>10} {:>12}",
             "N", "m", "cand", "Recall@10", "Speedup", "QPS", "ms/query");
    println!("{}", "-".repeat(75));

    for (i, &n) in scales.iter().enumerate() {
        let (_, brute_time, ref vectors, ref queries) = brute_times[i];
        let gt = &ground_truths[i];

        // Best config from earlier: m=16384, candidates=5000
        let m = 16384;
        let cand = 2000;

        let mut index = WitnessLDPCIndex::new(dim, m, 4, 64, true);
        index.add(vectors.clone(), true);

        let start = Instant::now();
        let results = index.batch_search(queries, k, cand);
        let elapsed = start.elapsed().as_secs_f64();
        let query_time = elapsed * 1000.0 / n_queries as f64;

        let recall: f64 = results.iter().enumerate()
            .map(|(j, r)| compute_recall(r, &gt[j], k))
            .sum::<f64>() / n_queries as f64;

        let speedup = brute_time / query_time;
        let qps = n_queries as f64 / elapsed;

        println!("{:<10} {:>8} {:>8} {:>11.1}% {:>9.1}x {:>10.0} {:>12.2}",
                 n, m, cand, recall * 100.0, speedup, qps, query_time);
    }

    // Test TURBO implementation
    println!("\n{}", "=".repeat(80));
    println!("TURBO IMPLEMENTATION (all optimizations)");
    println!("{}", "=".repeat(80));

    println!("\n{:<10} {:>8} {:>8} {:>12} {:>10} {:>10} {:>12}",
             "N", "m", "cand", "Recall@10", "Speedup", "QPS", "ms/query");
    println!("{}", "-".repeat(75));

    // Test multiple configurations
    let configs = vec![
        (8192, 64, 4, 1000),
        (8192, 64, 4, 2000),
        (16384, 64, 4, 1000),
        (16384, 64, 4, 2000),
        (16384, 96, 6, 1000),
        (16384, 96, 6, 2000),
        (32768, 64, 4, 1000),
        (32768, 96, 6, 1000),
        (32768, 96, 6, 2000),
    ];

    for (i, &n) in scales.iter().enumerate() {
        let (_, brute_time, ref vectors, ref queries) = brute_times[i];
        let gt = &ground_truths[i];

        println!("\n--- N = {} ---", n);

        for &(m, w, num_k, cand) in &configs {
            let mut index = TurboWitnessIndex::new(dim, m, num_k, w);
            index.add(vectors.clone(), true);

            let start = Instant::now();
            let results = index.batch_search_turbo(queries, k, cand);
            let elapsed = start.elapsed().as_secs_f64();
            let query_time = elapsed * 1000.0 / n_queries as f64;

            let recall: f64 = results.iter().enumerate()
                .map(|(j, r)| compute_recall(r, &gt[j], k))
                .sum::<f64>() / n_queries as f64;

            let speedup = brute_time / query_time;
            let qps = n_queries as f64 / elapsed;

            // Only print configs with reasonable recall
            if recall >= 0.90 {
                println!("{:<10} {:>8} {:>8} {:>11.1}%** {:>8.1}x {:>10.0} {:>12.3}",
                         n, m, cand, recall * 100.0, speedup, qps, query_time);
            } else if recall >= 0.80 {
                println!("{:<10} {:>8} {:>8} {:>11.1}%* {:>9.1}x {:>10.0} {:>12.3}",
                         n, m, cand, recall * 100.0, speedup, qps, query_time);
            } else if recall >= 0.70 {
                println!("{:<10} {:>8} {:>8} {:>11.1}% {:>10.1}x {:>10.0} {:>12.3}",
                         n, m, cand, recall * 100.0, speedup, qps, query_time);
            }
        }
    }

    // Best config comparison
    println!("\n{}", "=".repeat(80));
    println!("BEST CONFIGURATION COMPARISON");
    println!("{}", "=".repeat(80));

    println!("\n{:<10} {:>25} {:>25}",
             "N", "Original (m=16K, c=2K)", "Turbo (m=16K, c=2K)");
    println!("{:<10} {:>25} {:>25}",
             "", "(Recall/Speedup/QPS)", "(Recall/Speedup/QPS)");
    println!("{}", "-".repeat(65));

    for (i, &n) in scales.iter().enumerate() {
        let (_, brute_time, ref vectors, ref queries) = brute_times[i];
        let gt = &ground_truths[i];

        let m = 16384;
        let cand = 2000;

        // Original
        let mut orig = WitnessLDPCIndex::new(dim, m, 4, 64, true);
        orig.add(vectors.clone(), true);
        let start = Instant::now();
        let res_orig = orig.batch_search(queries, k, cand);
        let time_orig = start.elapsed().as_secs_f64();
        let recall_orig: f64 = res_orig.iter().enumerate().map(|(j, r)| compute_recall(r, &gt[j], k)).sum::<f64>() / n_queries as f64;
        let speedup_orig = brute_time / (time_orig * 1000.0 / n_queries as f64);
        let qps_orig = n_queries as f64 / time_orig;

        // Turbo
        let mut turbo = TurboWitnessIndex::new(dim, m, 4, 64);
        turbo.add(vectors.clone(), true);
        let start = Instant::now();
        let res_turbo = turbo.batch_search_turbo(queries, k, cand);
        let time_turbo = start.elapsed().as_secs_f64();
        let recall_turbo: f64 = res_turbo.iter().enumerate().map(|(j, r)| compute_recall(r, &gt[j], k)).sum::<f64>() / n_queries as f64;
        let speedup_turbo = brute_time / (time_turbo * 1000.0 / n_queries as f64);
        let qps_turbo = n_queries as f64 / time_turbo;

        println!("{:<10} {:>6.0}%/{:>5.1}x/{:>6.0} {:>6.0}%/{:>5.1}x/{:>6.0}",
                 n,
                 recall_orig * 100.0, speedup_orig, qps_orig,
                 recall_turbo * 100.0, speedup_turbo, qps_turbo);
    }

    println!("\n{}", "=".repeat(80));
    println!("CONCLUSION");
    println!("{}", "=".repeat(80));
    println!("\n* = 80%+ recall");
    println!("** = 90%+ recall");
    println!("\nOptimizations applied:");
    println!("  1. SIMD-friendly Hamming distance (unrolled loops)");
    println!("  2. Early termination with top-k heap");
    println!("  3. Cache-optimized contiguous storage");
    println!("  4. Partial sort for candidate selection");
}
