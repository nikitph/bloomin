//! Adaptive M Benchmark: Testing Information-Theoretic Scaling
//!
//! From the REWA bound: m ≥ C_M · (L²/Δ²K) · (log N + log(1/δ))
//!
//! This benchmark tests:
//! 1. Fixed m (showing the scaling wall)
//! 2. Adaptive m ∝ log(N) (the fix)
//! 3. IDF-based witness extraction improvement

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;
use std::collections::HashSet;

use witness_retrieval::WitnessLDPCIndex;
use witness_retrieval::production::{
    compute_optimal_m, brute_force_batch_search, ProductionWitnessLDPC
};

/// Generate clustered data for realistic similarity search
fn generate_clustered_data(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let vectors_per_cluster = n / n_clusters;

    let mut vectors = Vec::with_capacity(n);

    for _ in 0..n_clusters {
        let center: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 10.0 - 5.0).collect();

        for _ in 0..vectors_per_cluster {
            let sample: Vec<f32> = center
                .iter()
                .map(|&c| c + (rng.gen::<f32>() - 0.5) * 2.0)
                .collect();
            vectors.push(sample);
        }
    }

    while vectors.len() < n {
        let cluster_idx = rng.gen_range(0..n_clusters);
        let center_vec = &vectors[cluster_idx * vectors_per_cluster];
        let sample: Vec<f32> = center_vec
            .iter()
            .map(|&c| c + (rng.gen::<f32>() - 0.5) * 2.0)
            .collect();
        vectors.push(sample);
    }

    // Normalize
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

fn generate_queries(vectors: &[Vec<f32>], n_queries: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = vectors.len();

    (0..n_queries)
        .map(|_| {
            let idx = rng.gen_range(0..n);
            let mut q = vectors[idx].clone();
            for x in &mut q {
                *x += (rng.gen::<f32>() - 0.5) * 0.2;
            }
            let norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut q {
                *x /= norm;
            }
            q
        })
        .collect()
}

fn compute_recall(pred: &[(u32, f32)], gt: &[(u32, f32)], k: usize) -> f64 {
    let pred_set: HashSet<u32> = pred.iter().take(k).map(|(id, _)| *id).collect();
    let gt_set: HashSet<u32> = gt.iter().take(k).map(|(id, _)| *id).collect();
    pred_set.intersection(&gt_set).count() as f64 / k as f64
}

fn compute_adaptive_m(n: usize) -> usize {
    // From information theory: m ∝ log(N)
    // Base: m=2048 at N=50K
    let base_m: f64 = 2048.0;
    let base_n: f64 = 50000.0;

    let m = base_m * ((n as f64).ln() / base_n.ln());

    // Round to next power of 2
    let m_pow2 = (m.log2().ceil() as u32).max(10); // min 1024
    1usize << m_pow2
}

fn benchmark_fixed_m(
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    ground_truth: &[Vec<(u32, f32)>],
    m: usize,
    k: usize,
    n_candidates: usize,
) -> (f64, f64, f64) {
    let dim = vectors[0].len();
    let n_queries = queries.len();

    let mut index = WitnessLDPCIndex::new(dim, m, 4, 64, true);

    let build_start = Instant::now();
    index.add(vectors.to_vec(), true);
    let _build_time = build_start.elapsed();

    let search_start = Instant::now();
    let results = index.batch_search(queries, k, n_candidates);
    let search_time = search_start.elapsed();

    let recall: f64 = results.iter().enumerate()
        .map(|(i, r)| compute_recall(r, &ground_truth[i], k))
        .sum::<f64>() / n_queries as f64;

    let query_time_ms = search_time.as_secs_f64() * 1000.0 / n_queries as f64;
    let memory_mb = index.memory_bytes() as f64 / 1024.0 / 1024.0;

    (recall, query_time_ms, memory_mb)
}

fn main() {
    println!("{}", "=".repeat(80));
    println!("ADAPTIVE M BENCHMARK: Information-Theoretic Scaling");
    println!("{}", "=".repeat(80));

    let dim = 768;
    let n_queries = 100;
    let k = 10;
    let n_candidates = 500;

    // Test at different scales
    let scales = vec![50_000, 100_000, 200_000, 500_000];

    println!("\n{}", "=".repeat(80));
    println!("PART 1: Fixed m=4096 (showing the scaling wall)");
    println!("{}", "=".repeat(80));

    println!("\n{:<12} {:>12} {:>12} {:>12} {:>12}",
             "N", "m", "Recall@10", "Query(ms)", "Memory(MB)");
    println!("{}", "-".repeat(60));

    for &n in &scales {
        let vectors = generate_clustered_data(n, dim, 100, 42);
        let queries = generate_queries(&vectors, n_queries, 123);
        let gt = brute_force_batch_search(&vectors, &queries, k);

        let (recall, query_time, memory) = benchmark_fixed_m(
            &vectors, &queries, &gt, 4096, k, n_candidates
        );

        println!("{:<12} {:>12} {:>11.1}% {:>12.2} {:>12.1}",
                 n, 4096, recall * 100.0, query_time, memory);
    }

    println!("\n{}", "=".repeat(80));
    println!("PART 2: Adaptive m ∝ log(N) (the fix)");
    println!("{}", "=".repeat(80));

    println!("\nAdaptive m values:");
    for &n in &scales {
        let m = compute_adaptive_m(n);
        println!("  N={}: m={}", n, m);
    }

    println!("\n{:<12} {:>12} {:>12} {:>12} {:>12}",
             "N", "m (adaptive)", "Recall@10", "Query(ms)", "Memory(MB)");
    println!("{}", "-".repeat(60));

    for &n in &scales {
        let vectors = generate_clustered_data(n, dim, 100, 42);
        let queries = generate_queries(&vectors, n_queries, 123);
        let gt = brute_force_batch_search(&vectors, &queries, k);

        let m = compute_adaptive_m(n);
        let (recall, query_time, memory) = benchmark_fixed_m(
            &vectors, &queries, &gt, m, k, n_candidates
        );

        println!("{:<12} {:>12} {:>11.1}% {:>12.2} {:>12.1}",
                 n, m, recall * 100.0, query_time, memory);
    }

    println!("\n{}", "=".repeat(80));
    println!("PART 3: Optimal m from REWA bound");
    println!("{}", "=".repeat(80));

    println!("\nOptimal m values (L=64, K=4, gap_ratio=0.3, safety=0.2):");
    for &n in &scales {
        let m = compute_optimal_m(n, 64, 4, 0.3, 0.2);
        println!("  N={}: m={}", n, m);
    }

    println!("\n{:<12} {:>12} {:>12} {:>12} {:>12}",
             "N", "m (optimal)", "Recall@10", "Query(ms)", "Memory(MB)");
    println!("{}", "-".repeat(60));

    for &n in &scales {
        let vectors = generate_clustered_data(n, dim, 100, 42);
        let queries = generate_queries(&vectors, n_queries, 123);
        let gt = brute_force_batch_search(&vectors, &queries, k);

        let m = compute_optimal_m(n, 64, 4, 0.3, 0.2);
        let (recall, query_time, memory) = benchmark_fixed_m(
            &vectors, &queries, &gt, m, k, n_candidates
        );

        println!("{:<12} {:>12} {:>11.1}% {:>12.2} {:>12.1}",
                 n, m, recall * 100.0, query_time, memory);
    }

    println!("\n{}", "=".repeat(80));
    println!("PART 4: Scaling m more aggressively");
    println!("{}", "=".repeat(80));

    // More aggressive scaling: double m every 2x N
    let aggressive_m = |n: usize| -> usize {
        match n {
            _ if n <= 50_000 => 2048,
            _ if n <= 100_000 => 4096,
            _ if n <= 200_000 => 8192,
            _ => 16384,
        }
    };

    println!("\n{:<12} {:>12} {:>12} {:>12} {:>12}",
             "N", "m (aggressive)", "Recall@10", "Query(ms)", "Memory(MB)");
    println!("{}", "-".repeat(60));

    for &n in &scales {
        let vectors = generate_clustered_data(n, dim, 100, 42);
        let queries = generate_queries(&vectors, n_queries, 123);
        let gt = brute_force_batch_search(&vectors, &queries, k);

        let m = aggressive_m(n);
        let (recall, query_time, memory) = benchmark_fixed_m(
            &vectors, &queries, &gt, m, k, n_candidates
        );

        println!("{:<12} {:>12} {:>11.1}% {:>12.2} {:>12.1}",
                 n, m, recall * 100.0, query_time, memory);
    }

    // Compute brute force baseline times
    println!("\n{}", "=".repeat(80));
    println!("BRUTE FORCE BASELINE");
    println!("{}", "=".repeat(80));

    println!("\n{:<12} {:>15}", "N", "Query time(ms)");
    println!("{}", "-".repeat(30));

    for &n in &[50_000usize, 100_000, 200_000] {
        let vectors = generate_clustered_data(n, dim, 100, 42);
        let queries = generate_queries(&vectors, n_queries, 123);

        let start = Instant::now();
        let _gt = brute_force_batch_search(&vectors, &queries, k);
        let elapsed = start.elapsed();

        println!("{:<12} {:>15.2}", n, elapsed.as_secs_f64() * 1000.0 / n_queries as f64);
    }

    println!("\n{}", "=".repeat(80));
    println!("SUMMARY: Speedup with Adaptive m");
    println!("{}", "=".repeat(80));
}
