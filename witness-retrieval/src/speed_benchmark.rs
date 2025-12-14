//! Speed Benchmark: Comparing Hamming-Only vs Exact Re-ranking
//!
//! Tests whether we can get massive speedup with Hamming-only search
//! while maintaining acceptable recall through proper m scaling.

use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;
use std::time::Instant;

use witness_retrieval::fast_index::{FastWitnessIndex, brute_force_batch};

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
    println!("SPEED BENCHMARK: Hamming-Only vs Exact Re-ranking");
    println!("{}", "=".repeat(80));

    let dim = 768;
    let n_queries = 100;
    let k = 10;

    let scales = vec![50_000, 100_000, 200_000, 500_000];

    println!("\n{}", "=".repeat(80));
    println!("PART 1: Brute Force Baseline");
    println!("{}", "=".repeat(80));

    let mut brute_times = vec![];
    for &n in &scales {
        let vectors = generate_clustered_data(n, dim, 100, 42);
        let queries = generate_queries(&vectors, n_queries, 123);

        let start = Instant::now();
        let _gt = brute_force_batch(&vectors, &queries, k);
        let elapsed = start.elapsed().as_secs_f64();
        let query_time = elapsed * 1000.0 / n_queries as f64;
        let qps = n_queries as f64 / elapsed;

        println!("N={}: {:.2}ms/query, {:.0} QPS", n, query_time, qps);
        brute_times.push((n, query_time));
    }

    // Test different m values with Hamming-only search
    println!("\n{}", "=".repeat(80));
    println!("PART 2: Hamming-Only Search (NO vector re-ranking)");
    println!("{}", "=".repeat(80));

    println!("\n{:<10} {:>8} {:>8} {:>6} {:>12} {:>10} {:>10} {:>12}",
             "N", "m", "w", "K", "candidates", "Recall", "Speedup", "QPS");
    println!("{}", "-".repeat(85));

    for &n in &scales {
        let vectors = generate_clustered_data(n, dim, 100, 42);
        let queries = generate_queries(&vectors, n_queries, 123);
        let gt = brute_force_batch(&vectors, &queries, k);
        let brute_time = brute_times.iter().find(|(x, _)| *x == n).unwrap().1;

        // Test configurations: (m, witnesses, hashes, candidates)
        let configs = vec![
            (8192, 64, 4, 100),
            (16384, 64, 4, 100),
            (16384, 96, 6, 100),
            (32768, 64, 4, 100),
            (32768, 96, 6, 100),
            (32768, 128, 6, 100),
        ];

        for (m, w, num_k, cand) in configs {
            let mut index = FastWitnessIndex::new(dim, m, num_k, w);
            index.add(vectors.clone(), false); // NO vector storage

            let start = Instant::now();
            let results = index.batch_search_hamming(&queries, k, cand);
            let elapsed = start.elapsed().as_secs_f64();
            let query_time = elapsed * 1000.0 / n_queries as f64;

            let recall: f64 = results.iter().enumerate()
                .map(|(i, r)| compute_recall(r, &gt[i], k))
                .sum::<f64>() / n_queries as f64;

            let speedup = brute_time / query_time;
            let qps = n_queries as f64 / elapsed;

            if recall >= 0.90 {
                println!("{:<10} {:>8} {:>8} {:>6} {:>12} {:>9.1}%* {:>9.0}x {:>12.0}",
                         n, m, w, num_k, cand, recall * 100.0, speedup, qps);
            } else if recall >= 0.70 {
                println!("{:<10} {:>8} {:>8} {:>6} {:>12} {:>9.1}%  {:>9.0}x {:>12.0}",
                         n, m, w, num_k, cand, recall * 100.0, speedup, qps);
            }
        }
        println!();
    }

    // Test with exact re-ranking but fewer candidates
    println!("\n{}", "=".repeat(80));
    println!("PART 3: Exact Re-ranking with Minimal Candidates");
    println!("{}", "=".repeat(80));

    println!("\n{:<10} {:>8} {:>8} {:>6} {:>12} {:>10} {:>10} {:>12}",
             "N", "m", "w", "K", "candidates", "Recall", "Speedup", "QPS");
    println!("{}", "-".repeat(85));

    for &n in &scales {
        let vectors = generate_clustered_data(n, dim, 100, 42);
        let queries = generate_queries(&vectors, n_queries, 123);
        let gt = brute_force_batch(&vectors, &queries, k);
        let brute_time = brute_times.iter().find(|(x, _)| *x == n).unwrap().1;

        // Test with fewer candidates + exact re-ranking
        let configs = vec![
            (16384, 96, 6, 50),
            (16384, 96, 6, 100),
            (16384, 96, 6, 200),
            (32768, 96, 6, 50),
            (32768, 96, 6, 100),
            (32768, 128, 6, 100),
        ];

        for (m, w, num_k, cand) in configs {
            let mut index = FastWitnessIndex::new(dim, m, num_k, w);
            index.add(vectors.clone(), true); // WITH vector storage

            let start = Instant::now();
            let results = index.batch_search_exact(&queries, k, cand);
            let elapsed = start.elapsed().as_secs_f64();
            let query_time = elapsed * 1000.0 / n_queries as f64;

            let recall: f64 = results.iter().enumerate()
                .map(|(i, r)| compute_recall(r, &gt[i], k))
                .sum::<f64>() / n_queries as f64;

            let speedup = brute_time / query_time;
            let qps = n_queries as f64 / elapsed;

            if recall >= 0.95 {
                println!("{:<10} {:>8} {:>8} {:>6} {:>12} {:>9.1}%** {:>8.0}x {:>12.0}",
                         n, m, w, num_k, cand, recall * 100.0, speedup, qps);
            } else if recall >= 0.90 {
                println!("{:<10} {:>8} {:>8} {:>6} {:>12} {:>9.1}%* {:>9.0}x {:>12.0}",
                         n, m, w, num_k, cand, recall * 100.0, speedup, qps);
            } else if recall >= 0.80 {
                println!("{:<10} {:>8} {:>8} {:>6} {:>12} {:>9.1}%  {:>9.0}x {:>12.0}",
                         n, m, w, num_k, cand, recall * 100.0, speedup, qps);
            }
        }
        println!();
    }

    // Best configs comparison
    println!("\n{}", "=".repeat(80));
    println!("PART 4: Best Configuration Comparison");
    println!("{}", "=".repeat(80));

    println!("\n{:<10} {:>20} {:>20} {:>20}",
             "N", "Hamming-Only", "Exact(100 cand)", "Exact(200 cand)");
    println!("{:<10} {:>20} {:>20} {:>20}",
             "", "(R/Speedup/QPS)", "(R/Speedup/QPS)", "(R/Speedup/QPS)");
    println!("{}", "-".repeat(75));

    for &n in &scales {
        let vectors = generate_clustered_data(n, dim, 100, 42);
        let queries = generate_queries(&vectors, n_queries, 123);
        let gt = brute_force_batch(&vectors, &queries, k);
        let brute_time = brute_times.iter().find(|(x, _)| *x == n).unwrap().1;

        let m = 32768;
        let w = 96;
        let num_k = 6;

        // Hamming-only
        let mut idx_h = FastWitnessIndex::new(dim, m, num_k, w);
        idx_h.add(vectors.clone(), false);
        let start = Instant::now();
        let res_h = idx_h.batch_search_hamming(&queries, k, 100);
        let time_h = start.elapsed().as_secs_f64();
        let recall_h: f64 = res_h.iter().enumerate().map(|(i, r)| compute_recall(r, &gt[i], k)).sum::<f64>() / n_queries as f64;
        let speedup_h = brute_time / (time_h * 1000.0 / n_queries as f64);
        let qps_h = n_queries as f64 / time_h;

        // Exact 100
        let mut idx_e1 = FastWitnessIndex::new(dim, m, num_k, w);
        idx_e1.add(vectors.clone(), true);
        let start = Instant::now();
        let res_e1 = idx_e1.batch_search_exact(&queries, k, 100);
        let time_e1 = start.elapsed().as_secs_f64();
        let recall_e1: f64 = res_e1.iter().enumerate().map(|(i, r)| compute_recall(r, &gt[i], k)).sum::<f64>() / n_queries as f64;
        let speedup_e1 = brute_time / (time_e1 * 1000.0 / n_queries as f64);
        let qps_e1 = n_queries as f64 / time_e1;

        // Exact 200
        let start = Instant::now();
        let res_e2 = idx_e1.batch_search_exact(&queries, k, 200);
        let time_e2 = start.elapsed().as_secs_f64();
        let recall_e2: f64 = res_e2.iter().enumerate().map(|(i, r)| compute_recall(r, &gt[i], k)).sum::<f64>() / n_queries as f64;
        let speedup_e2 = brute_time / (time_e2 * 1000.0 / n_queries as f64);
        let qps_e2 = n_queries as f64 / time_e2;

        println!("{:<10} {:>5.0}%/{:>4.0}x/{:>5.0} {:>5.0}%/{:>4.0}x/{:>5.0} {:>5.0}%/{:>4.0}x/{:>5.0}",
                 n,
                 recall_h * 100.0, speedup_h, qps_h,
                 recall_e1 * 100.0, speedup_e1, qps_e1,
                 recall_e2 * 100.0, speedup_e2, qps_e2);
    }

    println!("\n{}", "=".repeat(80));
    println!("CONCLUSION");
    println!("{}", "=".repeat(80));
    println!("\nKey findings:");
    println!("  * = 90%+ recall");
    println!("  ** = 95%+ recall");
    println!("\nHamming-only: Maximum speed, lower recall");
    println!("Exact re-ranking: Higher recall, moderate speed");
    println!("Optimal: Large m (32768+), 100-200 candidates for exact");
}
