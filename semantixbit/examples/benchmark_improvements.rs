//! Recall Improvement Benchmark
//!
//! Tests different strategies to improve recall within 2ms latency budget.

use semantixbit::{Config, RewaQuantizer, BinaryIndex, SearchEngine};
use std::collections::HashSet;
use std::time::Instant;

fn main() {
    println!("=== Recall Improvement Benchmark ===");
    println!("Target: <2ms latency, maximize recall\n");
    
    // Generate test dataset
    let num_docs = 10_000;
    let dim = 384;
    let num_queries = 100;
    
    let embeddings = generate_synthetic_embeddings(num_docs, dim);
    let doc_ids: Vec<String> = (0..num_docs).map(|i| format!("doc_{}", i)).collect();
    let query_embeddings: Vec<Vec<f32>> = embeddings[..num_queries].to_vec();
    
    // Compute ground truth
    println!("Computing ground truth...");
    let ground_truth = compute_ground_truth(&embeddings, &query_embeddings, &doc_ids, 100);
    println!();
    
    // Test Strategy 1: Two-Stage Retrieval (different candidate sizes)
    println!("=== Strategy 1: Two-Stage Retrieval ===\n");
    
    let bit_depth = 2048;
    let quantizer = RewaQuantizer::new(dim, bit_depth, 42);
    let mut index = BinaryIndex::with_reranking(bit_depth);
    
    for (i, embedding) in embeddings.iter().enumerate() {
        let signature = quantizer.quantize(embedding);
        index.add(doc_ids[i].clone(), signature, Some(embedding.clone()));
    }
    
    let engine = SearchEngine::new(quantizer, index);
    
    for &candidate_size in &[50, 100, 200, 500, 1000] {
        test_two_stage(&engine, &query_embeddings, &ground_truth, candidate_size);
    }
    
    // Test Strategy 2: Multi-Probe LSH
    println!("\n=== Strategy 2: Multi-Probe LSH ===\n");
    
    let quantizer2 = RewaQuantizer::new(dim, bit_depth, 42);
    let mut index2 = BinaryIndex::new(bit_depth);
    
    for (i, embedding) in embeddings.iter().enumerate() {
        let signature = quantizer2.quantize(embedding);
        index2.add(doc_ids[i].clone(), signature, None);
    }
    
    let engine2 = SearchEngine::new(quantizer2, index2);
    
    for &num_probes in &[1, 2, 3, 5] {
        test_multiprobe(&engine2, &query_embeddings, &ground_truth, num_probes);
    }
}

fn test_two_stage(
    engine: &SearchEngine,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<String>],
    candidate_size: usize,
) {
    let k = 10;
    let mut total_recall = 0.0;
    let mut total_time = 0.0;
    
    for (i, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let results = engine.search_with_reranking(query, k, Some(candidate_size));
        total_time += start.elapsed().as_secs_f64();
        
        let retrieved: HashSet<String> = results.iter()
            .map(|r| r.doc_id.clone())
            .collect();
        
        let relevant: HashSet<String> = ground_truth[i][..k]
            .iter()
            .cloned()
            .collect();
        
        let intersection = retrieved.intersection(&relevant).count();
        total_recall += intersection as f64 / relevant.len() as f64;
    }
    
    let avg_recall = total_recall / queries.len() as f64;
    let avg_latency_ms = (total_time / queries.len() as f64) * 1000.0;
    
    let status = if avg_latency_ms < 2.0 { "✅" } else { "⚠️" };
    
    println!("Candidates: {:4} | Recall@10: {:.1}% | Latency: {:.2}ms {}", 
             candidate_size, avg_recall * 100.0, avg_latency_ms, status);
}

fn test_multiprobe(
    engine: &SearchEngine,
    queries: &[Vec<f32>],
    ground_truth: &[Vec<String>],
    num_probes: usize,
) {
    let k = 10;
    let mut total_recall = 0.0;
    let mut total_time = 0.0;
    
    for (i, query) in queries.iter().enumerate() {
        let start = Instant::now();
        let results = engine.search_multiprobe(query, k, num_probes);
        total_time += start.elapsed().as_secs_f64();
        
        let retrieved: HashSet<String> = results.iter()
            .map(|r| r.doc_id.clone())
            .collect();
        
        let relevant: HashSet<String> = ground_truth[i][..k]
            .iter()
            .cloned()
            .collect();
        
        let intersection = retrieved.intersection(&relevant).count();
        total_recall += intersection as f64 / relevant.len() as f64;
    }
    
    let avg_recall = total_recall / queries.len() as f64;
    let avg_latency_ms = (total_time / queries.len() as f64) * 1000.0;
    
    let status = if avg_latency_ms < 2.0 { "✅" } else { "⚠️" };
    
    println!("Probes: {} | Recall@10: {:.1}% | Latency: {:.2}ms {}", 
             num_probes, avg_recall * 100.0, avg_latency_ms, status);
}

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
            
            similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
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
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            vec.iter().map(|x| x / norm).collect()
        })
        .collect()
}
