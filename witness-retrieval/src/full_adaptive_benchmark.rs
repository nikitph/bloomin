//! Full Adaptive Benchmark: Scale BOTH m AND n_candidates
//!
//! The key insight: as N grows, we need:
//! 1. Larger m (to avoid hash collisions)
//! 2. More candidates (to ensure true neighbors are in candidate set)

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::time::Instant;
use std::collections::HashSet;

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

/// Adaptive config that scales both m and n_candidates
fn adaptive_config(n: usize) -> (usize, usize, usize, usize) {
    // Returns (m, num_hashes, num_witnesses, n_candidates)
    match n {
        _ if n <= 50_000 => (2048, 4, 64, 500),
        _ if n <= 100_000 => (4096, 4, 64, 750),
        _ if n <= 200_000 => (8192, 4, 64, 1000),
        _ if n <= 500_000 => (16384, 4, 64, 1500),
        _ => (32768, 4, 64, 2000),
    }
}

fn main() {
    println!("{}", "=".repeat(80));
    println!("FULL ADAPTIVE BENCHMARK: Scale m AND n_candidates");
    println!("{}", "=".repeat(80));

    let dim = 768;
    let n_queries = 100;
    let k = 10;

    let scales = vec![50_000, 100_000, 200_000, 500_000];

    // First, show the scaling wall with fixed params
    println!("\n{}", "=".repeat(80));
    println!("FIXED CONFIG: m=4096, n_candidates=500 (showing scaling wall)");
    println!("{}", "=".repeat(80));

    println!("\n{:<12} {:>10} {:>12} {:>12} {:>10} {:>12}",
             "N", "m", "candidates", "Recall@10", "Speedup", "Memory(MB)");
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
        index.add(vectors.clone(), true);

        let search_start = Instant::now();
        let results = index.batch_search(&queries, k, 500);
        let search_time = search_start.elapsed().as_secs_f64() * 1000.0 / n_queries as f64;

        let recall: f64 = results.iter().enumerate()
            .map(|(i, r)| compute_recall(r, &gt[i], k))
            .sum::<f64>() / n_queries as f64;

        let speedup = brute_time / search_time;
        let memory = index.memory_bytes() as f64 / 1024.0 / 1024.0;

        println!("{:<12} {:>10} {:>12} {:>11.1}% {:>10.1}x {:>12.1}",
                 n, 4096, 500, recall * 100.0, speedup, memory);
    }

    // Now show the fix with adaptive params
    println!("\n{}", "=".repeat(80));
    println!("ADAPTIVE CONFIG: Scale m and n_candidates with N");
    println!("{}", "=".repeat(80));

    println!("\nAdaptive configurations:");
    for &n in &scales {
        let (m, k_hash, w, cand) = adaptive_config(n);
        println!("  N={}: m={}, K={}, witnesses={}, candidates={}", n, m, k_hash, w, cand);
    }

    println!("\n{:<12} {:>10} {:>12} {:>12} {:>10} {:>12}",
             "N", "m", "candidates", "Recall@10", "Speedup", "Memory(MB)");
    println!("{}", "-".repeat(70));

    for (i, &n) in scales.iter().enumerate() {
        let vectors = generate_clustered_data(n, dim, 100, 42);
        let queries = generate_queries(&vectors, n_queries, 123);
        let gt = brute_force_batch_search(&vectors, &queries, k);

        let (m, num_hashes, num_witnesses, n_candidates) = adaptive_config(n);

        let mut index = WitnessLDPCIndex::new(dim, m, num_hashes, num_witnesses, true);
        index.add(vectors.clone(), true);

        let search_start = Instant::now();
        let results = index.batch_search(&queries, k, n_candidates);
        let search_time = search_start.elapsed().as_secs_f64() * 1000.0 / n_queries as f64;

        let recall: f64 = results.iter().enumerate()
            .map(|(i, r)| compute_recall(r, &gt[i], k))
            .sum::<f64>() / n_queries as f64;

        let speedup = brute_times[i].1 / search_time;
        let memory = index.memory_bytes() as f64 / 1024.0 / 1024.0;

        println!("{:<12} {:>10} {:>12} {:>11.1}% {:>10.1}x {:>12.1}",
                 n, m, n_candidates, recall * 100.0, speedup, memory);
    }

    // Try even more aggressive scaling
    println!("\n{}", "=".repeat(80));
    println!("AGGRESSIVE CONFIG: Even larger m and more candidates");
    println!("{}", "=".repeat(80));

    let aggressive_config = |n: usize| -> (usize, usize) {
        match n {
            _ if n <= 50_000 => (4096, 500),
            _ if n <= 100_000 => (8192, 1000),
            _ if n <= 200_000 => (16384, 2000),
            _ if n <= 500_000 => (32768, 3000),
            _ => (65536, 5000),
        }
    };

    println!("\n{:<12} {:>10} {:>12} {:>12} {:>10} {:>12}",
             "N", "m", "candidates", "Recall@10", "Speedup", "Memory(MB)");
    println!("{}", "-".repeat(70));

    for (i, &n) in scales.iter().enumerate() {
        let vectors = generate_clustered_data(n, dim, 100, 42);
        let queries = generate_queries(&vectors, n_queries, 123);
        let gt = brute_force_batch_search(&vectors, &queries, k);

        let (m, n_candidates) = aggressive_config(n);

        let mut index = WitnessLDPCIndex::new(dim, m, 4, 64, true);
        index.add(vectors.clone(), true);

        let search_start = Instant::now();
        let results = index.batch_search(&queries, k, n_candidates);
        let search_time = search_start.elapsed().as_secs_f64() * 1000.0 / n_queries as f64;

        let recall: f64 = results.iter().enumerate()
            .map(|(i, r)| compute_recall(r, &gt[i], k))
            .sum::<f64>() / n_queries as f64;

        let speedup = brute_times[i].1 / search_time;
        let memory = index.memory_bytes() as f64 / 1024.0 / 1024.0;

        println!("{:<12} {:>10} {:>12} {:>11.1}% {:>10.1}x {:>12.1}",
                 n, m, n_candidates, recall * 100.0, speedup, memory);
    }

    // Also try more witnesses
    println!("\n{}", "=".repeat(80));
    println!("MORE WITNESSES: m=8192, witnesses=96, K=6");
    println!("{}", "=".repeat(80));

    println!("\n{:<12} {:>10} {:>12} {:>12} {:>10} {:>12}",
             "N", "witnesses", "candidates", "Recall@10", "Speedup", "Memory(MB)");
    println!("{}", "-".repeat(70));

    for (i, &n) in scales.iter().enumerate() {
        let vectors = generate_clustered_data(n, dim, 100, 42);
        let queries = generate_queries(&vectors, n_queries, 123);
        let gt = brute_force_batch_search(&vectors, &queries, k);

        let m = 8192;
        let num_witnesses = 96;
        let num_hashes = 6;
        let n_candidates = (n / 50).max(500).min(3000); // 2% of dataset, capped

        let mut index = WitnessLDPCIndex::new(dim, m, num_hashes, num_witnesses, true);
        index.add(vectors.clone(), true);

        let search_start = Instant::now();
        let results = index.batch_search(&queries, k, n_candidates);
        let search_time = search_start.elapsed().as_secs_f64() * 1000.0 / n_queries as f64;

        let recall: f64 = results.iter().enumerate()
            .map(|(i, r)| compute_recall(r, &gt[i], k))
            .sum::<f64>() / n_queries as f64;

        let speedup = brute_times[i].1 / search_time;
        let memory = index.memory_bytes() as f64 / 1024.0 / 1024.0;

        println!("{:<12} {:>10} {:>12} {:>11.1}% {:>10.1}x {:>12.1}",
                 n, num_witnesses, n_candidates, recall * 100.0, speedup, memory);
    }

    println!("\n{}", "=".repeat(80));
    println!("CONCLUSION");
    println!("{}", "=".repeat(80));
    println!("\nBrute force baseline times:");
    for (n, t) in &brute_times {
        println!("  N={}: {:.2}ms/query", n, t);
    }
}
