//! Recall Evaluation: Measure accuracy vs. exact search
//!
//! Computes Recall@K by comparing SemantixBit results against
//! exact cosine similarity ground truth.

use semantixbit::{Config, RewaQuantizer, BinaryIndex, SearchEngine};
use std::collections::HashSet;
use std::time::Instant;

fn main() {
    println!("=== SemantixBit Recall Evaluation ===\n");
    
    // Generate test dataset
    let num_docs = 10_000;
    let dim = 384;
    let num_queries = 100;
    
    println!("Dataset: {} documents, {} dimensions", num_docs, dim);
    println!("Queries: {}\n", num_queries);
    
    let embeddings = generate_synthetic_embeddings(num_docs, dim);
    let doc_ids: Vec<String> = (0..num_docs).map(|i| format!("doc_{}", i)).collect();
    
    // Use first N as queries
    let query_embeddings: Vec<Vec<f32>> = embeddings[..num_queries].to_vec();
    
    // Compute ground truth using exact cosine similarity
    println!("Computing ground truth (exact cosine similarity)...");
    let start = Instant::now();
    let ground_truth = compute_ground_truth(&embeddings, &query_embeddings, &doc_ids, 100);
    println!("  Time: {:.2}s\n", start.elapsed().as_secs_f64());
    
    // Test different bit depths
    for bit_depth in [256, 512, 1024, 2048, 4096] {
        evaluate_recall(&embeddings, &doc_ids, &query_embeddings, &ground_truth, dim, bit_depth);
    }
}

fn evaluate_recall(
    embeddings: &[Vec<f32>],
    doc_ids: &[String],
    queries: &[Vec<f32>],
    ground_truth: &[Vec<String>],
    dim: usize,
    bit_depth: usize,
) {
    println!("=== Bit Depth: {} ===", bit_depth);
    
    let config = Config {
        input_dim: dim,
        bit_depth,
        seed: 42,
        hybrid_mode: false,
        keyword_bits: 0,
    };
    
    // Build index
    let quantizer = RewaQuantizer::new(config.input_dim, config.bit_depth, config.seed);
    let mut index = BinaryIndex::new(config.bit_depth);
    
    for (i, embedding) in embeddings.iter().enumerate() {
        let signature = quantizer.quantize(embedding);
        index.add(doc_ids[i].clone(), signature, None);
    }
    
    let engine = SearchEngine::new(quantizer, index);
    
    // Compute recall@k for different k values
    let k_values = [1, 5, 10, 20, 50, 100];
    
    for &k in &k_values {
        let mut total_recall = 0.0;
        
        for (i, query) in queries.iter().enumerate() {
            let results = engine.search(query, k);
            let retrieved: HashSet<String> = results.iter()
                .map(|r| r.doc_id.clone())
                .collect();
            
            let relevant: HashSet<String> = ground_truth[i][..k.min(ground_truth[i].len())]
                .iter()
                .cloned()
                .collect();
            
            let intersection = retrieved.intersection(&relevant).count();
            let recall = intersection as f64 / relevant.len() as f64;
            total_recall += recall;
        }
        
        let avg_recall = total_recall / queries.len() as f64;
        println!("  Recall@{:3}: {:.4} ({:.1}%)", k, avg_recall, avg_recall * 100.0);
    }
    
    // Memory and speed
    let memory_mb = engine.index().memory_usage() as f64 / 1024.0 / 1024.0;
    println!("  Memory: {:.2} MB", memory_mb);
    
    let start = Instant::now();
    for query in queries.iter().take(100) {
        let _ = engine.search(query, 10);
    }
    let qps = 100.0 / start.elapsed().as_secs_f64();
    println!("  Speed: {:.0} QPS", qps);
    println!();
}

/// Compute ground truth using exact cosine similarity
fn compute_ground_truth(
    embeddings: &[Vec<f32>],
    queries: &[Vec<f32>],
    doc_ids: &[String],
    k: usize,
) -> Vec<Vec<String>> {
    queries.iter()
        .map(|query| {
            let mut similarities: Vec<(String, f32)> = embeddings.iter()
                .zip(doc_ids.iter())
                .map(|(doc, id)| {
                    let sim = cosine_similarity(query, doc);
                    (id.clone(), sim)
                })
                .collect();
            
            // Sort by similarity (descending)
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            // Return top-k doc IDs
            similarities.into_iter()
                .take(k)
                .map(|(id, _)| id)
                .collect()
        })
        .collect()
}

fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    // For normalized vectors, dot product = cosine similarity
    v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum()
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
