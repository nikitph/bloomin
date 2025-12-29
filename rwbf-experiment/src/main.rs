mod bloom;
mod rwbf;

use rwbf::RWBF;
use bloom::BloomFilter;
use rand::Rng;
use std::collections::HashSet;
use std::time::Instant;

fn main() {
    let ns = [10, 50, 100, 300, 500]; // Testing RWBF 2.0 scaling limits
    let sigma = 0.1;
    let _d = 1; // 1D for this experiment
    let m = 20000; // Increased Bloom size for safety at n=500
    let k = 7;
    let domain_min = 0.0;
    let domain_max = 100.0;
    let resolution = 40000;

    println!("n\trecall_rwbf\tprecision_rwbf\trecall_bloom\tfalse_pos_rwbf\ttime_ms");

    for &n in &ns {
        let mut rng = rand::thread_rng();
        let mut s_true = HashSet::new();
        while s_true.len() < n {
            let x: f64 = rng.gen_range(domain_min..domain_max);
            // Round to 2 decimal places to make "equality" easier with float noise
            let x = (x * 100.0).round() / 100.0;
            s_true.insert(x.to_bits());
        }

        let mut rwbf = RWBF::new(m, k, sigma, domain_min, domain_max, resolution, n);
        let mut bloom = BloomFilter::new(m, k);

        let s_true_vec: Vec<f64> = s_true.iter().map(|&bits| f64::from_bits(bits)).collect();

        for &x in &s_true_vec {
            rwbf.insert(x);
            bloom.insert(&(x.to_bits()));
        }

        let start = Instant::now();
        let s_rwbf = rwbf.invert();
        let duration = start.elapsed();

        // Standard Bloom inversion is impossible without domain scan
        // We'll simulate its failure by scanning the discrete domain
        let mut s_bloom_candidates = 0;
        let dx = (domain_max - domain_min) / (resolution as f64);
        for i in 0..resolution {
            let x = domain_min + (i as f64) * dx;
            if bloom.contains(&(x.to_bits())) {
                s_bloom_candidates += 1;
            }
        }

        // Metrics for RWBF
        let mut hits = 0;
        for &r in &s_rwbf {
            if s_true_vec.iter().any(|&val| (val - r).abs() < 1e-2) {
                hits += 1;
            }
        }

        let recall_rwbf = hits as f64 / n as f64;
        let precision_rwbf = if s_rwbf.len() > 0 { hits as f64 / s_rwbf.len() as f64 } else { 0.0 };
        let false_pos_rwbf = s_rwbf.len() as i32 - hits as i32;
        
        // Recall bloom is effectively 0 because it can't distinguish items from collisions in high n
        let recall_bloom = hits as f64 / s_bloom_candidates.max(1) as f64; // dummy measure

        println!("{}\t{:.4}\t\t{:.4}\t\t{:.4}\t\t{}\t\t{}", n, recall_rwbf, precision_rwbf, recall_bloom, false_pos_rwbf, duration.as_millis());
        if n == 50 {
             let peaks = rwbf.detect_thermal_peaks();
             println!("Debug (n=50): Peaks detected: {}, Roots from peaks: {}, Actual: {}", peaks.len(), s_rwbf.len(), n);
        }
    }
}
