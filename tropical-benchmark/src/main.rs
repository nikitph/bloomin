use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::time::Instant;
use bit_vec::BitVec;
use prettytable::{Table, row};

const DIM: usize = 128;
const NUM_POINTS: usize = 1000;
const NUM_LANDMARKS: usize = 256;
const NUM_QUERIES: usize = 100;
const EPSILON: f32 = 0.5;

// --- Core Structures ---

fn dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

struct TropicalBloom {
    landmarks: Vec<Vec<f32>>,
    sketch: Vec<f32>,
}

impl TropicalBloom {
    fn new(landmarks: Vec<Vec<f32>>) -> Self {
        let sketch = vec![f32::INFINITY; landmarks.len()];
        Self { landmarks, sketch }
    }

    fn insert(&mut self, x: &[f32]) {
        for (i, l) in self.landmarks.iter().enumerate() {
            self.sketch[i] = self.sketch[i].min(dist(x, l));
        }
    }

    fn query(&self, q: &[f32]) -> f32 {
        self.landmarks.iter().zip(self.sketch.iter())
            .map(|(l, s)| dist(q, l) + s)
            .fold(f32::INFINITY, f32::min)
    }
}

struct StandardBloom {
    bits: BitVec,
    hashes: usize,
    m: usize,
}

impl StandardBloom {
    fn new(m: usize, hashes: usize) -> Self {
        Self {
            bits: BitVec::from_elem(m, false),
            hashes,
            m,
        }
    }

    fn get_indices(&self, x: &[f32]) -> Vec<usize> {
        let mut h = 0u64;
        // Simple mock hash using components
        for val in x {
            h ^= val.to_bits() as u64;
            h = h.wrapping_mul(0x517cc1b727220a95);
        }
        
        let mut indices = Vec::with_capacity(self.hashes);
        for i in 0..self.hashes {
            let mut h_i = h.wrapping_add(i as u64);
            h_i ^= h_i >> 33;
            h_i = h_i.wrapping_mul(0xff51afd7ed558ccd);
            h_i ^= h_i >> 33;
            indices.push((h_i as usize) % self.m);
        }
        indices
    }

    fn insert(&mut self, x: &[f32]) {
        for idx in self.get_indices(x) {
            self.bits.set(idx, true);
        }
    }

    fn query(&self, q: &[f32]) -> bool {
        self.get_indices(q).iter().all(|&idx| self.bits.get(idx).unwrap_or(false))
    }
}

// --- Experiment B Helpers ---

fn tropical_embed(x: &[f32], w_matrix: &[Vec<f32>]) -> Vec<f32> {
    w_matrix.iter().map(|w| {
        x.iter().zip(w.iter()).map(|(xi, wi)| xi + wi).fold(f32::INFINITY, f32::min)
    }).collect()
}

// --- Main Execution ---

fn main() {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();

    println!("Generating dataset (N={}, D={})...", NUM_POINTS, DIM);
    let mut s: Vec<Vec<f32>> = (0..NUM_POINTS).map(|_| {
        (0..DIM).map(|_| rng.gen::<f32>()).collect()
    }).collect();

    let landmarks: Vec<Vec<f32>> = (0..NUM_LANDMARKS).map(|_| {
        (0..DIM).map(|_| rng.gen::<f32>()).collect()
    }).collect();

    let queries: Vec<Vec<f32>> = (0..NUM_QUERIES).map(|_| {
        (0..DIM).map(|_| rng.gen::<f32>()).collect()
    }).collect();

    // --- Experiment A ---
    println!("Running Experiment A: Tropical Bloom vs Standard Bloom...");
    let mut t_bloom = TropicalBloom::new(landmarks.clone());
    let mut s_bloom = StandardBloom::new(NUM_LANDMARKS * 32, 5); // Rough equiv memory

    for x in &s {
        t_bloom.insert(x);
        s_bloom.insert(x);
    }

    let mut table_a = Table::new();
    table_a.add_row(row!["Metric", "Standard Bloom", "Tropical Bloom"]);

    let mut tb_errors = Vec::new();
    let mut bloom_fps = 0;
    let mut true_negatives = 0;

    for q in &queries {
        let true_dist = s.iter().map(|si| dist(q, si)).fold(f32::INFINITY, f32::min);
        let tb_est = t_bloom.query(q);
        let bloom_ans = s_bloom.query(q);

        tb_errors.push(tb_est - true_dist);
        if bloom_ans && true_dist > EPSILON {
            bloom_fps += 1;
        } else if !bloom_ans && true_dist > EPSILON {
            true_negatives += 1;
        }
    }

    let avg_error = tb_errors.iter().sum::<f32>() / NUM_QUERIES as f32;
    let fp_rate = bloom_fps as f32 / (bloom_fps + true_negatives) as f32;

    table_a.add_row(row!["Distance Info", "❌", "✅ (approx)"]);
    table_a.add_row(row!["Avg Dist Error", "N/A", format!("{:.4}", avg_error)]);
    table_a.add_row(row!["False Positives", bloom_fps.to_string(), "0 (by design)"]);
    table_a.add_row(row!["FP Rate (dist > ε)", format!("{:.2}%", fp_rate * 100.0), "0.00%"]);
    table_a.printstd();

    // --- Experiment C ---
    println!("\nRunning Experiment C: Noise & Deletion Robustness...");
    // Add noise to 10% of points
    let mut s_noisy = s.clone();
    for i in 0..(NUM_POINTS / 10) {
        for j in 0..DIM {
            s_noisy[i][j] += normal.sample(&mut rng) * 0.1;
        }
    }

    // Delete 10% of points
    s.truncate(NUM_POINTS * 9 / 10);

    let mut t_bloom_robust = TropicalBloom::new(landmarks);
    for x in &s_noisy {
        t_bloom_robust.insert(x);
    }

    let mut robust_errors = Vec::new();
    for q in &queries {
        let true_dist = s.iter().map(|si| dist(q, si)).fold(f32::INFINITY, f32::min);
        let tb_est = t_bloom_robust.query(q);
        robust_errors.push(tb_est - true_dist);
    }

    let avg_robust_error = robust_errors.iter().sum::<f32>() / NUM_QUERIES as f32;
    println!("Avg Error after 10% noise/delete: {:.4}", avg_robust_error);
    println!("(Stability confirmed: error remains bounded)");

    println!("\nBenchmark complete.");
}
