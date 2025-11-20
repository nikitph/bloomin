//! Comparison benchmark: SBNG vs BM25.

mod bm25_baseline;
mod eval_metrics;

use std::sync::Arc;
use std::collections::HashMap;
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

fn main() -> anyhow::Result<()> {
    println!("=== SBNG vs BM25 Evaluation ===\n");

    // Load benchmark queries
    let queries_json = std::fs::read_to_string("benches/eval_queries.json")?;
    let queries: Vec<BenchmarkQuery> = serde_json::from_str(&queries_json)?;

    let corpus_path = "data/sample.jsonl";
    let k = 3; // Top-K results

    // === Build SBNG Index ===
    println!("Building SBNG index...");
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

    println!("SBNG index built: {} docs, {} concepts\n", doc_index.doc_ids.len(), concept_fps.len());

    // === Build BM25 Index ===
    println!("Building BM25 index...");
    let bm25_index = BM25Index::build_from_jsonl(corpus_path)?;
    println!("BM25 index built\n");

    // === Evaluate ===
    println!("Running evaluation...\n");
    println!("{:<30} {:<15} {:<15} {:<15} {:<15}", "Query", "System", "Recall@3", "NDCG@3", "MRR");
    println!("{}", "=".repeat(90));

    let mut sbng_metrics = MetricsAccumulator::new();
    let mut bm25_metrics = MetricsAccumulator::new();

    for query_item in &queries {
        let query = &query_item.query;
        let relevant = &query_item.relevant_docs;

        // SBNG results
        let sbng_results = sbng_engine.search(query, k);
        let sbng_results_f64: Vec<_> = sbng_results.iter()
            .map(|(id, score)| (id.clone(), *score as f64))
            .collect();

        let sbng_recall = recall_at_k(&sbng_results_f64, relevant, k);
        let sbng_ndcg = ndcg_at_k(&sbng_results_f64, relevant, k);
        let sbng_mrr = mrr(&sbng_results_f64, relevant);

        sbng_metrics.add(sbng_recall, sbng_ndcg, sbng_mrr);

        // BM25 results
        let bm25_results = bm25_index.search(query, k)?;
        let bm25_recall = recall_at_k(&bm25_results, relevant, k);
        let bm25_ndcg = ndcg_at_k(&bm25_results, relevant, k);
        let bm25_mrr = mrr(&bm25_results, relevant);

        bm25_metrics.add(bm25_recall, bm25_ndcg, bm25_mrr);

        // Print per-query results
        println!("{:<30} {:<15} {:<15.3} {:<15.3} {:<15.3}", query, "SBNG", sbng_recall, sbng_ndcg, sbng_mrr);
        println!("{:<30} {:<15} {:<15.3} {:<15.3} {:<15.3}", "", "BM25", bm25_recall, bm25_ndcg, bm25_mrr);
        println!();
    }

    // Print average metrics
    println!("{}", "=".repeat(90));
    println!("{:<30} {:<15} {:<15.3} {:<15.3} {:<15.3}", 
        "AVERAGE", "SBNG", 
        sbng_metrics.avg_recall(), 
        sbng_metrics.avg_ndcg(), 
        sbng_metrics.avg_mrr()
    );
    println!("{:<30} {:<15} {:<15.3} {:<15.3} {:<15.3}", 
        "", "BM25", 
        bm25_metrics.avg_recall(), 
        bm25_metrics.avg_ndcg(), 
        bm25_metrics.avg_mrr()
    );

    Ok(())
}

struct MetricsAccumulator {
    recall_sum: f64,
    ndcg_sum: f64,
    mrr_sum: f64,
    count: usize,
}

impl MetricsAccumulator {
    fn new() -> Self {
        Self {
            recall_sum: 0.0,
            ndcg_sum: 0.0,
            mrr_sum: 0.0,
            count: 0,
        }
    }

    fn add(&mut self, recall: f64, ndcg: f64, mrr: f64) {
        self.recall_sum += recall;
        self.ndcg_sum += ndcg;
        self.mrr_sum += mrr;
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
}
