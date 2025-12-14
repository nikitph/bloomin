//! Adaptive Flat Index Benchmark
//!
//! Tests the core theoretical insight: m ∝ log(N)
//! Simple flat index with adaptive code length - no hierarchical cascade.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;
use std::time::Instant;

use witness_retrieval::WitnessLDPCIndex;
use witness_retrieval::production::brute_force_batch_search;

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

/// Compute adaptive m using information-theoretic bound
/// m = base_m * ln(N) / ln(base_N)
fn compute_adaptive_m(n: usize, target_recall: f64) -> usize {
    // For high recall (95%+), we need:
    // m ≈ 2048 * ln(N) / ln(50K) for 95%
    // m ≈ 4096 * ln(N) / ln(50K) for 98%
    let base_m = if target_recall > 0.97 { 8192.0 }
                 else if target_recall > 0.95 { 4096.0 }
                 else { 2048.0 };
    let base_n: f64 = 50_000.0;

    let m = base_m * ((n as f64).ln() / base_n.ln());

    // Round to next power of 2
    let bits = (m.log2().ceil() as u32).max(11).min(18); // min 2048, max 262144
    1usize << bits
}

/// Compute adaptive candidates (percentage of dataset that decreases with scale)
fn compute_adaptive_candidates(n: usize, target_recall: f64) -> usize {
    // Higher target recall needs more candidates
    let pct = if target_recall > 0.97 { 0.02 }
              else if target_recall > 0.95 { 0.01 }
              else { 0.005 };

    let cand = (n as f64 * pct) as usize;
    cand.max(500).min(5000)
}

fn main() {
    println!("{}", "=".repeat(80));
    println!("ADAPTIVE FLAT INDEX BENCHMARK");
    println!("Testing: m ∝ log(N) with adaptive candidates");
    println!("{}", "=".repeat(80));

    let dim = 768;
    let n_queries = 100;
    let k = 10;

    let scales = vec![50_000, 100_000, 200_000, 500_000];

    // First, show the scaling wall with fixed params
    println!("\n{}", "=".repeat(80));
    println!("PART 1: FIXED m=4096, candidates=500 (the scaling wall)");
    println!("{}", "=".repeat(80));

    println!("\n{:<12} {:>10} {:>12} {:>12} {:>10} {:>12}",
             "N", "m", "candidates", "Recall@10", "Speedup", "QPS");
    println!("{}", "-".repeat(70));

    let mut brute_times = vec![];

    for &n in &scales {
        let vectors = generate_clustered_data(n, dim, 100, 42);
        let queries = generate_queries(&vectors, n_queries, 123);

        let brute_start = Instant::now();
        let gt = brute_force_batch_search(&vectors, &queries, k);
        let brute_time = brute_start.elapsed().as_secs_f64() * 1000.0 / n_queries as f64;
        brute_times.push((n, brute_time));

        let mut index = WitnessLDPCIndex::new(dim, 4096, 4, 64, true);
        index.add(vectors, true);

        let search_start = Instant::now();
        let results = index.batch_search(&queries, k, 500);
        let search_time = search_start.elapsed().as_secs_f64();
        let query_time = search_time * 1000.0 / n_queries as f64;

        let recall: f64 = results.iter().enumerate()
            .map(|(i, r)| compute_recall(r, &gt[i], k))
            .sum::<f64>() / n_queries as f64;

        let speedup = brute_time / query_time;
        let qps = n_queries as f64 / search_time;

        println!("{:<12} {:>10} {:>12} {:>11.1}% {:>10.1}x {:>12.0}",
                 n, 4096, 500, recall * 100.0, speedup, qps);
    }

    // Now show the fix with adaptive m
    println!("\n{}", "=".repeat(80));
    println!("PART 2: ADAPTIVE m ∝ log(N) for 95% target recall");
    println!("{}", "=".repeat(80));

    println!("\nAdaptive configurations (95% target):");
    for &n in &scales {
        let m = compute_adaptive_m(n, 0.95);
        let cand = compute_adaptive_candidates(n, 0.95);
        println!("  N={}: m={}, candidates={}", n, m, cand);
    }

    println!("\n{:<12} {:>10} {:>12} {:>12} {:>10} {:>12}",
             "N", "m", "candidates", "Recall@10", "Speedup", "QPS");
    println!("{}", "-".repeat(70));

    for (i, &n) in scales.iter().enumerate() {
        let vectors = generate_clustered_data(n, dim, 100, 42);
        let queries = generate_queries(&vectors, n_queries, 123);
        let gt = brute_force_batch_search(&vectors, &queries, k);

        let m = compute_adaptive_m(n, 0.95);
        let n_candidates = compute_adaptive_candidates(n, 0.95);

        let mut index = WitnessLDPCIndex::new(dim, m, 4, 64, true);
        index.add(vectors, true);

        let search_start = Instant::now();
        let results = index.batch_search(&queries, k, n_candidates);
        let search_time = search_start.elapsed().as_secs_f64();
        let query_time = search_time * 1000.0 / n_queries as f64;

        let recall: f64 = results.iter().enumerate()
            .map(|(i, r)| compute_recall(r, &gt[i], k))
            .sum::<f64>() / n_queries as f64;

        let speedup = brute_times[i].1 / query_time;
        let qps = n_queries as f64 / search_time;

        println!("{:<12} {:>10} {:>12} {:>11.1}% {:>10.1}x {:>12.0}",
                 n, m, n_candidates, recall * 100.0, speedup, qps);
    }

    // Try for 98% target recall
    println!("\n{}", "=".repeat(80));
    println!("PART 3: ADAPTIVE m for 98% target recall");
    println!("{}", "=".repeat(80));

    println!("\nAdaptive configurations (98% target):");
    for &n in &scales {
        let m = compute_adaptive_m(n, 0.98);
        let cand = compute_adaptive_candidates(n, 0.98);
        println!("  N={}: m={}, candidates={}", n, m, cand);
    }

    println!("\n{:<12} {:>10} {:>12} {:>12} {:>10} {:>12}",
             "N", "m", "candidates", "Recall@10", "Speedup", "QPS");
    println!("{}", "-".repeat(70));

    for (i, &n) in scales.iter().enumerate() {
        let vectors = generate_clustered_data(n, dim, 100, 42);
        let queries = generate_queries(&vectors, n_queries, 123);
        let gt = brute_force_batch_search(&vectors, &queries, k);

        let m = compute_adaptive_m(n, 0.98);
        let n_candidates = compute_adaptive_candidates(n, 0.98);

        let mut index = WitnessLDPCIndex::new(dim, m, 4, 64, true);
        index.add(vectors, true);

        let search_start = Instant::now();
        let results = index.batch_search(&queries, k, n_candidates);
        let search_time = search_start.elapsed().as_secs_f64();
        let query_time = search_time * 1000.0 / n_queries as f64;

        let recall: f64 = results.iter().enumerate()
            .map(|(i, r)| compute_recall(r, &gt[i], k))
            .sum::<f64>() / n_queries as f64;

        let speedup = brute_times[i].1 / query_time;
        let qps = n_queries as f64 / search_time;

        println!("{:<12} {:>10} {:>12} {:>11.1}% {:>10.1}x {:>12.0}",
                 n, m, n_candidates, recall * 100.0, speedup, qps);
    }

    // Sweep over different m values to find optimal
    println!("\n{}", "=".repeat(80));
    println!("PART 4: m SWEEP at N=200K (finding optimal m)");
    println!("{}", "=".repeat(80));

    let n = 200_000;
    let vectors = generate_clustered_data(n, dim, 100, 42);
    let queries = generate_queries(&vectors, n_queries, 123);
    let gt = brute_force_batch_search(&vectors, &queries, k);

    let brute_time = brute_times.iter().find(|(x, _)| *x == n).unwrap().1;

    println!("\n{:<10} {:>12} {:>12} {:>10} {:>12} {:>12}",
             "m", "candidates", "Recall@10", "Speedup", "QPS", "Memory(MB)");
    println!("{}", "-".repeat(70));

    for &m in &[2048, 4096, 8192, 16384, 32768, 65536] {
        for &n_candidates in &[500, 1000, 2000] {
            let mut index = WitnessLDPCIndex::new(dim, m, 4, 64, true);
            index.add(vectors.clone(), true);

            let search_start = Instant::now();
            let results = index.batch_search(&queries, k, n_candidates);
            let search_time = search_start.elapsed().as_secs_f64();
            let query_time = search_time * 1000.0 / n_queries as f64;

            let recall: f64 = results.iter().enumerate()
                .map(|(i, r)| compute_recall(r, &gt[i], k))
                .sum::<f64>() / n_queries as f64;

            let speedup = brute_time / query_time;
            let qps = n_queries as f64 / search_time;
            let memory = index.memory_bytes() as f64 / 1024.0 / 1024.0;

            if recall > 0.90 {
                println!("{:<10} {:>12} {:>11.1}%* {:>9.1}x {:>12.0} {:>12.1}",
                         m, n_candidates, recall * 100.0, speedup, qps, memory);
            } else {
                println!("{:<10} {:>12} {:>11.1}% {:>10.1}x {:>12.0} {:>12.1}",
                         m, n_candidates, recall * 100.0, speedup, qps, memory);
            }
        }
        println!();
    }

    println!("{}", "=".repeat(80));
    println!("CONCLUSION");
    println!("{}", "=".repeat(80));
    println!("\nInformation-theoretic bound: m ≥ C · L² · log(N) / (Δ²K)");
    println!("As N grows, m must grow logarithmically to maintain recall.");
    println!("\nKey tradeoffs:");
    println!("  - Larger m → Better recall but more memory and slower search");
    println!("  - More candidates → Better recall but slower re-ranking");
    println!("  - The product m × candidates determines accuracy/speed tradeoff");
}
