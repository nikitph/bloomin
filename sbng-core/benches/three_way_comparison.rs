//! Three-way comparison: SBNG vs BM25 vs Vector Embeddings.

mod bm25_baseline;
mod eval_metrics;

use std::sync::Arc;
use std::collections::HashMap;
use std::time::Instant;
use serde::{Deserialize, Serialize};

use sbng_core::{
    SbngConfig,
    corpus::{JsonlCorpus, WhitespaceTokenizer, InterningConceptExtractor, ConceptInterner, ConceptExtractor},
    pipeline::{GraphBuildPipeline, SignatureGenerationPipeline},
    search::{DocIndex, QueryEngine},
    BloomFingerprint,
};

use bm25_baseline::BM25Index;
use eval_metrics::{recall_at_k, ndcg_at_k, mrr};

#[derive(Debug, Deserialize, Serialize)]
struct BenchmarkQuery {
    query: String,
    relevant_docs: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct VectorResult {
    query: String,
    relevant_docs: Vec<String>,
    retrieved: Vec<VectorDoc>,
    latency_ms: f64,
}

#[derive(Debug, Deserialize)]
struct VectorDoc {
    doc_id: String,
    score: f64,
}

fn main() -> anyhow::Result<()> {
    println!("=== SBNG vs BM25 vs Vector Embeddings ===\n");

    // Load benchmark queries
    let queries_json = std::fs::read_to_string("benches/eval_queries.json")?;
    let queries: Vec<BenchmarkQuery> = serde_json::from_str(&queries_json)?;

    // Load vector results
    let vector_json = std::fs::read_to_string("scripts/vector_results.json")?;
    let vector_results: Vec<VectorResult> = serde_json::from_str(&vector_json)?;

    let corpus_path = "data/sample.jsonl";

    // === Build SBNG Index ===
    println!("Building SBNG index...");
    let sbng_start = Instant::now();
    
    let mut config = SbngConfig::default();
    config.cooccur_min = 1;
    config.pmi_min = 0.0;
    config.min_degree = 1;

    let interner = Arc::new(ConceptInterner::with_hasher(
        sbng_core::corpus::interner::SerializableHasher::default()
    ));

    let tokenizer = WhitespaceTokenizer;
    let stopwords = ["the", "is", "a", "an", "of", "and", "or", "for", "to", "in", "at", "by"];
    let extractor = InterningConceptExtractor::new(tokenizer, interner.clone(), &stopwords);

    let corpus = JsonlCorpus::new(corpus_path.to_string());
    let graph_pipeline = GraphBuildPipeline::new(config.clone(), &extractor, interner.clone());
    let graph = graph_pipeline.build_from_jsonl(&corpus)?;

    let sig_pipeline = SignatureGenerationPipeline::new(config.clone());
    let concept_fps_vec = sig_pipeline.generate(&graph)?;
    let concept_fps: HashMap<_, _> = concept_fps_vec.into_iter().collect();

    // Build doc index
    let mut doc_index = DocIndex::new();
    for doc_res in corpus.iter()? {
        let doc = doc_res?;
        let concepts = extractor.extract_concepts(&doc.text);
        let mut fp = BloomFingerprint::new(config.bloom_bits, config.bloom_hashes);
        for c in concepts {
            if let Some(cfp) = concept_fps.get(&c.concept_id) {
                fp.merge(cfp);
            } else {
                fp.insert_concept(c.concept_id);
            }
        }
        doc_index.add(doc.id, fp);
    }

    let sbng_engine = QueryEngine::new(
        Arc::new(extractor),
        &concept_fps,
        &doc_index,
        config.bloom_bits,
        config.bloom_hashes,
    );

    let sbng_build_time = sbng_start.elapsed();
    println!("SBNG index built in {:?}\n", sbng_build_time);

    // === Build BM25 Index ===
    println!("Building BM25 index...");
    let bm25_start = Instant::now();
    let bm25_index = BM25Index::build_from_jsonl(corpus_path)?;
    let bm25_build_time = bm25_start.elapsed();
    println!("BM25 index built in {:?}\n", bm25_build_time);

    // === Evaluate ===
    println!("Running evaluation (k=3 and k=5)...\n");
    
    for k in [3, 5] {
        println!("=== Results @ K={} ===", k);
        println!("{:<30} {:<15} {:<15} {:<15} {:<15} {:<15}", 
            "Query", "System", "Recall", "NDCG", "MRR", "Latency (µs)");
        println!("{}", "=".repeat(105));

        let mut sbng_metrics = MetricsAccumulator::new();
        let mut bm25_metrics = MetricsAccumulator::new();
        let mut vector_metrics = MetricsAccumulator::new();

        for (i, query_item) in queries.iter().enumerate() {
            let query = &query_item.query;
            let relevant = &query_item.relevant_docs;

            // SBNG results
            let sbng_start = Instant::now();
            let sbng_results = sbng_engine.search(query, k);
            let sbng_latency_us = sbng_start.elapsed().as_micros();
            
            let sbng_results_f64: Vec<_> = sbng_results.iter()
                .map(|(id, score)| (id.clone(), *score as f64))
                .collect();

            let sbng_recall = recall_at_k(&sbng_results_f64, relevant, k);
            let sbng_ndcg = ndcg_at_k(&sbng_results_f64, relevant, k);
            let sbng_mrr = mrr(&sbng_results_f64, relevant);

            sbng_metrics.add(sbng_recall, sbng_ndcg, sbng_mrr, sbng_latency_us as f64);

            // BM25 results
            let bm25_start = Instant::now();
            let bm25_results = bm25_index.search(query, k)?;
            let bm25_latency_us = bm25_start.elapsed().as_micros();
            
            let bm25_recall = recall_at_k(&bm25_results, relevant, k);
            let bm25_ndcg = ndcg_at_k(&bm25_results, relevant, k);
            let bm25_mrr = mrr(&bm25_results, relevant);

            bm25_metrics.add(bm25_recall, bm25_ndcg, bm25_mrr, bm25_latency_us as f64);

            // Vector results
            let vector_result = &vector_results[i];
            let vector_retrieved: Vec<_> = vector_result.retrieved.iter()
                .map(|d| (d.doc_id.clone(), d.score))
                .collect();
            
            let vector_recall = recall_at_k(&vector_retrieved, relevant, k);
            let vector_ndcg = ndcg_at_k(&vector_retrieved, relevant, k);
            let vector_mrr = mrr(&vector_retrieved, relevant);
            let vector_latency_us = vector_result.latency_ms * 1000.0; // ms to µs

            vector_metrics.add(vector_recall, vector_ndcg, vector_mrr, vector_latency_us);

            // Print per-query results
            println!("{:<30} {:<15} {:<15.3} {:<15.3} {:<15.3} {:<15.1}", 
                query, "SBNG", sbng_recall, sbng_ndcg, sbng_mrr, sbng_latency_us);
            println!("{:<30} {:<15} {:<15.3} {:<15.3} {:<15.3} {:<15.1}", 
                "", "BM25", bm25_recall, bm25_ndcg, bm25_mrr, bm25_latency_us);
            println!("{:<30} {:<15} {:<15.3} {:<15.3} {:<15.3} {:<15.1}", 
                "", "Vector", vector_recall, vector_ndcg, vector_mrr, vector_latency_us);
            println!();
        }

        // Print average metrics
        println!("{}", "=".repeat(105));
        println!("{:<30} {:<15} {:<15.3} {:<15.3} {:<15.3} {:<15.1}", 
            "AVERAGE", "SBNG", 
            sbng_metrics.avg_recall(), 
            sbng_metrics.avg_ndcg(), 
            sbng_metrics.avg_mrr(),
            sbng_metrics.avg_latency()
        );
        println!("{:<30} {:<15} {:<15.3} {:<15.3} {:<15.3} {:<15.1}", 
            "", "BM25", 
            bm25_metrics.avg_recall(), 
            bm25_metrics.avg_ndcg(), 
            bm25_metrics.avg_mrr(),
            bm25_metrics.avg_latency()
        );
        println!("{:<30} {:<15} {:<15.3} {:<15.3} {:<15.3} {:<15.1}", 
            "", "Vector", 
            vector_metrics.avg_recall(), 
            vector_metrics.avg_ndcg(), 
            vector_metrics.avg_mrr(),
            vector_metrics.avg_latency()
        );
        println!("\n");
    }

    // Print build time comparison
    println!("=== Index Build Time ===");
    println!("SBNG:   {:?}", sbng_build_time);
    println!("BM25:   {:?}", bm25_build_time);
    println!("Vector: (Python script - not measured here)");

    Ok(())
}

struct MetricsAccumulator {
    recall_sum: f64,
    ndcg_sum: f64,
    mrr_sum: f64,
    latency_sum: f64,
    count: usize,
}

impl MetricsAccumulator {
    fn new() -> Self {
        Self {
            recall_sum: 0.0,
            ndcg_sum: 0.0,
            mrr_sum: 0.0,
            latency_sum: 0.0,
            count: 0,
        }
    }

    fn add(&mut self, recall: f64, ndcg: f64, mrr: f64, latency: f64) {
        self.recall_sum += recall;
        self.ndcg_sum += ndcg;
        self.mrr_sum += mrr;
        self.latency_sum += latency;
        self.count += 1;
    }

    fn avg_recall(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.recall_sum / self.count as f64 }
    }

    fn avg_ndcg(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.ndcg_sum / self.count as f64 }
    }

    fn avg_mrr(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.mrr_sum / self.count as f64 }
    }

    fn avg_latency(&self) -> f64 {
        if self.count == 0 { 0.0 } else { self.latency_sum / self.count as f64 }
    }
}
