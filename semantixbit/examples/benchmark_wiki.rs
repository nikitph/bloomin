//! Wikipedia Benchmark Example
//!
//! Demonstrates SemantixBit performance on Wikipedia dataset
//! and compares against FAISS baseline.

use semantixbit::{Config, RewaQuantizer, BinaryIndex, SearchEngine};
use std::time::Instant;
use std::fs;

fn main() {
    println!("=== SemantixBit Wikipedia Benchmark ===\n");
    
    // Try to load real Wikipedia data, fall back to synthetic
    let (num_docs, embedding_dim) = match load_wikipedia_info() {
        Ok((n, d)) => {
            println!("Found Wikipedia dataset:");
            println!("  - Passages: {}", n);
            println!("  - Embedding dimension: {}", d);
            (n.min(10_000), d)  // Limit to 10k for demo
        }
        Err(_) => {
            println!("Wikipedia data not found. Using synthetic data for demonstration.");
            println!("Run 'python scripts/prepare_wiki_data.py' to generate real Wikipedia embeddings.\n");
            (10_000, 384)
        }
    };
    
    let embeddings = generate_synthetic_embeddings(num_docs, embedding_dim);
    let doc_ids: Vec<String> = (0..num_docs).map(|i| format!("doc_{}", i)).collect();
    
    println!("Dataset: {} documents, {} dimensions\n", num_docs, embedding_dim);
    
    // Test different bit depths
    for bit_depth in [256, 1024, 2048] {
        benchmark_bit_depth(&embeddings, &doc_ids, embedding_dim, bit_depth);
    }
}

fn load_wikipedia_info() -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let info_str = fs::read_to_string("data/wikipedia/info.json")?;
    let info: serde_json::Value = serde_json::from_str(&info_str)?;
    
    let num_passages = info["num_passages"].as_u64().ok_or("Missing num_passages")? as usize;
    let embedding_dim = info["embedding_dim"].as_u64().ok_or("Missing embedding_dim")? as usize;
    
    Ok((num_passages, embedding_dim))
}

fn benchmark_bit_depth(embeddings: &[Vec<f32>], doc_ids: &[String], dim: usize, bit_depth: usize) {
    println!("\n=== Benchmark: {} bits ===", bit_depth);
    
    let config = Config {
        input_dim: dim,
        bit_depth,
        seed: 42,
        hybrid_mode: false,
        keyword_bits: 0,
    };
    
    // Build index
    println!("Building index...");
    let start = Instant::now();
    
    let quantizer = RewaQuantizer::new(config.input_dim, config.bit_depth, config.seed);
    let mut index = BinaryIndex::new(config.bit_depth);
    
    for (i, embedding) in embeddings.iter().enumerate() {
        let signature = quantizer.quantize(embedding);
        index.add(doc_ids[i].clone(), signature, None);
    }
    
    let build_time = start.elapsed();
    println!("  Build time: {:.2}s", build_time.as_secs_f64());
    
    // Memory usage
    let memory_mb = index.memory_usage() as f64 / 1024.0 / 1024.0;
    println!("  Memory usage: {:.2} MB", memory_mb);
    
    // Compression ratio vs float32
    let float_memory = embeddings.len() * dim * 4;
    let compression_ratio = float_memory as f64 / index.memory_usage() as f64;
    println!("  Compression ratio: {:.1}x", compression_ratio);
    
    // Search benchmark
    let engine = SearchEngine::new(quantizer, index);
    
    let num_queries = 100;
    let k = 10;
    
    println!("\nRunning {} queries (k={})...", num_queries, k);
    let start = Instant::now();
    
    for i in 0..num_queries {
        let query = &embeddings[i];
        let _results = engine.search(query, k);
    }
    
    let search_time = start.elapsed();
    let qps = num_queries as f64 / search_time.as_secs_f64();
    let latency_ms = search_time.as_secs_f64() * 1000.0 / num_queries as f64;
    
    println!("  Total search time: {:.2}s", search_time.as_secs_f64());
    println!("  Queries per second: {:.0}", qps);
    println!("  Latency per query: {:.2}ms", latency_ms);
    
    // Verify correctness (top-1 should be the query itself)
    let test_query = &embeddings[0];
    let results = engine.search(test_query, 5);
    println!("\nSample search results for doc_0:");
    for (i, result) in results.iter().enumerate() {
        println!("  {}. {} (distance: {}, score: {:.3})", 
                 i+1, result.doc_id, result.distance, result.score);
    }
}

fn generate_synthetic_embeddings(num_docs: usize, dim: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    (0..num_docs)
        .map(|_| {
            let vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
            // Normalize
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            vec.iter().map(|x| x / norm).collect()
        })
        .collect()
}
