//! Hierarchical V2 Benchmark: Full Production System Test
//!
//! Tests the complete HierarchicalV2 system at multiple scales:
//! - 50K, 100K, 200K, 500K, 1M vectors
//! - With and without IDF-based witness extraction
//! - Compares against baseline flat index and brute force

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;
use std::time::Instant;

use witness_retrieval::hierarchical_v2::{HierarchicalV2, brute_force_batch_search};
use witness_retrieval::WitnessLDPCIndex;

/// Generate clustered data for realistic similarity search
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

    // Normalize
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

fn benchmark_scale(n: usize, dim: usize, n_queries: usize, k: usize) -> (f64, f64, f64, f64, f64, f64, f64) {
    println!("\n{}", "=".repeat(80));
    println!("BENCHMARK: {} vectors (dim={})", n, dim);
    println!("{}", "=".repeat(80));

    // Generate data
    println!("Generating {} clustered vectors...", n);
    let gen_start = Instant::now();
    let vectors = generate_clustered_data(n, dim, 100, 42);
    let queries = generate_queries(&vectors, n_queries, 123);
    println!("  Data generated in {:.2}s", gen_start.elapsed().as_secs_f64());

    // Ground truth
    println!("Computing ground truth ({} queries)...", n_queries);
    let gt_start = Instant::now();
    let ground_truth = brute_force_batch_search(&vectors, &queries, k);
    let brute_time = gt_start.elapsed().as_secs_f64() * 1000.0 / n_queries as f64;
    println!("  Brute force: {:.2}ms per query", brute_time);

    // === Baseline: Flat index (m=4096) ===
    println!("\n--- Baseline (flat, m=4096) ---");
    let mut baseline = WitnessLDPCIndex::new(dim, 4096, 4, 64, true);

    let build_start = Instant::now();
    baseline.add(vectors.clone(), true);
    let baseline_build = build_start.elapsed().as_secs_f64();
    println!("  Build time: {:.2}s", baseline_build);

    let search_start = Instant::now();
    let baseline_results = baseline.batch_search(&queries, k, 500);
    let baseline_search = search_start.elapsed().as_secs_f64() * 1000.0 / n_queries as f64;

    let baseline_recall: f64 = baseline_results.iter().enumerate()
        .map(|(i, r)| compute_recall(r, &ground_truth[i], k))
        .sum::<f64>() / n_queries as f64;

    let baseline_speedup = brute_time / baseline_search;
    let baseline_memory = baseline.memory_bytes() as f64 / 1024.0 / 1024.0;

    println!("  Recall@{}: {:.1}%", k, baseline_recall * 100.0);
    println!("  Query time: {:.3}ms per query", baseline_search);
    println!("  Speedup: {:.1}x over brute force", baseline_speedup);
    println!("  Memory: {:.1} MB", baseline_memory);

    // === HierarchicalV2 without IDF ===
    println!("\n--- HierarchicalV2 (no IDF) ---");
    let mut hier_no_idf = HierarchicalV2::new(dim, n, false);

    let build_start = Instant::now();
    hier_no_idf.add(vectors.clone(), true, false);
    let hier_build = build_start.elapsed().as_secs_f64();
    println!("  Build time: {:.2}s", hier_build);

    let search_start = Instant::now();
    let hier_results = hier_no_idf.batch_search(&queries, k);
    let hier_search = search_start.elapsed().as_secs_f64() * 1000.0 / n_queries as f64;

    let hier_recall: f64 = hier_results.iter().enumerate()
        .map(|(i, r)| compute_recall(r, &ground_truth[i], k))
        .sum::<f64>() / n_queries as f64;

    let hier_speedup = brute_time / hier_search;
    let hier_memory = hier_no_idf.memory_bytes() as f64 / 1024.0 / 1024.0;

    println!("  Recall@{}: {:.1}%", k, hier_recall * 100.0);
    println!("  Query time: {:.3}ms per query", hier_search);
    println!("  Speedup: {:.1}x over brute force", hier_speedup);
    println!("  Memory: {:.1} MB", hier_memory);

    // === HierarchicalV2 with IDF ===
    println!("\n--- HierarchicalV2 (with IDF) ---");
    let mut hier_idf = HierarchicalV2::new(dim, n, true);

    let build_start = Instant::now();
    hier_idf.add(vectors.clone(), true, true);
    let idf_build = build_start.elapsed().as_secs_f64();
    println!("  Build time: {:.2}s", idf_build);

    let search_start = Instant::now();
    let idf_results = hier_idf.batch_search(&queries, k);
    let idf_search = search_start.elapsed().as_secs_f64() * 1000.0 / n_queries as f64;

    let idf_recall: f64 = idf_results.iter().enumerate()
        .map(|(i, r)| compute_recall(r, &ground_truth[i], k))
        .sum::<f64>() / n_queries as f64;

    let idf_speedup = brute_time / idf_search;
    let idf_memory = hier_idf.memory_bytes() as f64 / 1024.0 / 1024.0;

    println!("  Recall@{}: {:.1}%", k, idf_recall * 100.0);
    println!("  Query time: {:.3}ms per query", idf_search);
    println!("  Speedup: {:.1}x over brute force", idf_speedup);
    println!("  Memory: {:.1} MB", idf_memory);

    // Summary table
    println!("\n--- SUMMARY ---");
    println!("{:<30} {:>12} {:>12} {:>12}", "Method", "Recall@10", "Speedup", "Memory(MB)");
    println!("{}", "-".repeat(70));
    println!("{:<30} {:>11.1}% {:>11.1}x {:>12.1}", "Brute Force", 100.0, 1.0, 0.0);
    println!("{:<30} {:>11.1}% {:>11.1}x {:>12.1}", "Baseline (flat, m=4096)", baseline_recall * 100.0, baseline_speedup, baseline_memory);
    println!("{:<30} {:>11.1}% {:>11.1}x {:>12.1}", "HierarchicalV2 (no IDF)", hier_recall * 100.0, hier_speedup, hier_memory);
    println!("{:<30} {:>11.1}% {:>11.1}x {:>12.1}", "HierarchicalV2 (with IDF)", idf_recall * 100.0, idf_speedup, idf_memory);

    (brute_time, baseline_recall, baseline_speedup, hier_recall, hier_speedup, idf_recall, idf_speedup)
}

fn main() {
    println!("{}", "=".repeat(80));
    println!("HIERARCHICAL V2 PRODUCTION BENCHMARK");
    println!("Full System Test with Adaptive Parameters");
    println!("{}", "=".repeat(80));

    let dim = 768;
    let n_queries = 100;
    let k = 10;

    // Test scales
    let scales = vec![50_000, 100_000, 200_000, 500_000];

    let mut results = Vec::new();

    for &n in &scales {
        let (brute, base_r, base_s, hier_r, hier_s, idf_r, idf_s) = benchmark_scale(n, dim, n_queries, k);
        results.push((n, brute, base_r, base_s, hier_r, hier_s, idf_r, idf_s));
    }

    // Final summary
    println!("\n{}", "=".repeat(80));
    println!("FINAL SUMMARY: Recall@10 and Speedup at Different Scales");
    println!("{}", "=".repeat(80));

    println!("\n{:<12} {:>15} {:>22} {:>22} {:>22}",
             "N", "Brute(ms)", "Baseline(R/S)", "HierV2(R/S)", "HierV2+IDF(R/S)");
    println!("{}", "-".repeat(95));

    for (n, brute, base_r, base_s, hier_r, hier_s, idf_r, idf_s) in &results {
        println!("{:<12} {:>15.2} {:>10.1}%/{:>5.1}x {:>10.1}%/{:>5.1}x {:>10.1}%/{:>5.1}x",
                 n, brute,
                 base_r * 100.0, base_s,
                 hier_r * 100.0, hier_s,
                 idf_r * 100.0, idf_s);
    }

    println!("\n{}", "=".repeat(80));
    println!("KEY OBSERVATIONS:");
    println!("{}", "=".repeat(80));
    println!("1. Baseline (fixed m=4096): Recall degrades as N increases (the scaling wall)");
    println!("2. HierarchicalV2: Adaptive m maintains recall at scale");
    println!("3. IDF: Prioritizing rare dimensions should improve discriminability");
    println!("\nTheoretical basis: m >= C * L^2 * log(N) / (Delta^2 * K)");
    println!("As N grows, m must grow logarithmically to maintain recall");

    println!("\n{}", "=".repeat(80));
    println!("BENCHMARK COMPLETE");
    println!("{}", "=".repeat(80));
}
