use ultra_sparse_bloom::{UltraSparseBloom, StandardBloomFilter};
use std::time::Instant;
use rand::{Rng, thread_rng};
use std::collections::HashSet;

fn main() {
    let n = 10_000_000;
    let q = 1_000_000;
    let m = 50_000_000;
    let k = 5;
    let tau = 4.5; // Adjusted for sparsity

    println!("Experiment: N={}, Q={}, m={}, k={}, tau={}", n, q, m, k, tau);

    // 1. Generate Datasets
    println!("Generating datasets...");
    let mut rng = thread_rng();
    let mut insert_data = Vec::with_capacity(n);
    for _ in 0..n {
        insert_data.push(rng.r#gen::<u64>());
    }

    let mut queries = Vec::with_capacity(q);
    for i in 0..q {
        if i % 2 == 0 {
            // Existing element
            queries.push(insert_data[rng.gen_range(0..n)]);
        } else {
            // Random element
            queries.push(rng.r#gen::<u64>());
        }
    }

    // 2. Insertions
    println!("Starting insertions...");
    
    // Standard Bloom
    let start = Instant::now();
    let mut sbf = StandardBloomFilter::new(m, k);
    for &x in &insert_data {
        sbf.insert(&x);
    }
    let sbf_insert_time = start.elapsed();
    println!("Standard Bloom insertion: {:?}", sbf_insert_time);

    // Ultra-Sparse Bloom
    let start = Instant::now();
    let mut usb = UltraSparseBloom::new(m, k, tau);
    for &x in &insert_data {
        usb.insert(x);
    }
    let usb_insert_time = start.elapsed();
    println!("Ultra-Sparse Bloom insertion: {:?}", usb_insert_time);

    // 3. Queries
    println!("Starting queries...");

    // Standard Bloom
    let mut sbf_hits = 0;
    let mut sbf_false_positives = 0;
    let reservoir: HashSet<u64> = insert_data.iter().cloned().collect();
    
    let start = Instant::now();
    for &query in &queries {
        let result = sbf.query(&query);
        if result {
            if !reservoir.contains(&query) {
                sbf_false_positives += 1;
            }
            sbf_hits += 1;
        }
    }
    let sbf_query_time = start.elapsed();
    let sbf_qps = (q as f64) / sbf_query_time.as_secs_f64();

    // Ultra-Sparse Bloom
    let mut usb_hits = 0;
    let mut usb_false_positives = 0;
    let mut usb_false_negatives = 0;
    
    let start = Instant::now();
    for &query in &queries {
        let result = usb.query(&query, 0.9, 0.0);
        let actual = reservoir.contains(&query);
        if result != actual {
            if result && !actual {
                usb_false_positives += 1;
            } else if !result && actual {
                usb_false_negatives += 1;
            }
        }
        if result {
            usb_hits += 1;
        }
    }
    let usb_query_time = start.elapsed();
    let _ = (sbf_hits, usb_hits); // Suppress unused
    let usb_qps = (q as f64) / usb_query_time.as_secs_f64();

    // 4. Results
    println!("\n--- RESULTS ---");
    println!("Metric                 | Standard Bloom       | Ultra-Sparse Bloom");
    println!("-----------------------|----------------------|-------------------");
    println!("Memory (Estimated)     | {:>10.2} MB    | {:>10.2} MB", 
             (sbf.memory_usage() as f64) / 1_048_576.0, 
             (usb.memory_usage() as f64) / 1_048_576.0);
    println!("Active Buckets         | {:>10}         | {:>10}", m, usb.field.len());
    println!("Query QPS              | {:>10.2}         | {:>10.2}", sbf_qps, usb_qps);
    println!("Query Latency (Avg)    | {:>10.4} ms      | {:>10.4} ms", 
             sbf_query_time.as_secs_f64() * 1000.0 / (q as f64), 
             usb_query_time.as_secs_f64() * 1000.0 / (q as f64));
    println!("False Positives        | {:>10}         | {:>10}", sbf_false_positives, usb_false_positives);
    println!("False Negatives        | {:>10}         | {:>10}", 0, usb_false_negatives);
    println!("Exact Lookups          | {:>10}         | {:>10}", 0, usb.stats.exact_lookups);
    println!("Exact Lookup Rate      | {:>10.2}%        | {:>10.2}%", 0.0, (usb.stats.exact_lookups as f64) / (q as f64) * 100.0);
}
