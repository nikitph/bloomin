use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use sbng_core::{
    corpus::{InterningConceptExtractor, WhitespaceTokenizer, ConceptInterner, JsonlCorpus},
    persistence,
    search::QueryEngine,
};

/// Test query with expected relevant keywords
struct TestQuery {
    query: &'static str,
    keywords: &'static [&'static str],
}

const TEST_QUERIES: &[TestQuery] = &[
    TestQuery {
        query: "quantum mechanics physics",
        keywords: &["quantum", "physics", "mechanics", "particle"],
    },
    TestQuery {
        query: "artificial intelligence machine learning",
        keywords: &["artificial intelligence", "machine learning", "neural", "algorithm"],
    },
    TestQuery {
        query: "climate change global warming",
        keywords: &["climate", "warming", "greenhouse", "carbon"],
    },
    TestQuery {
        query: "DNA genetics heredity",
        keywords: &["dna", "gene", "genetic", "heredity"],
    },
    TestQuery {
        query: "computer programming software",
        keywords: &["computer", "programming", "software", "code"],
    },
    TestQuery {
        query: "world war two hitler",
        keywords: &["world war", "hitler", "nazi", "1939"],
    },
    TestQuery {
        query: "roman empire caesar",
        keywords: &["roman", "empire", "caesar", "rome"],
    },
    TestQuery {
        query: "egyptian pyramids pharaoh",
        keywords: &["egypt", "pyramid", "pharaoh", "ancient"],
    },
    TestQuery {
        query: "christopher columbus america",
        keywords: &["columbus", "america", "voyage", "discovery"],
    },
    TestQuery {
        query: "leonardo da vinci painting",
        keywords: &["leonardo", "vinci", "painting", "renaissance"],
    },
];

/// Check if document is relevant based on keyword matching
fn is_relevant(text: &str, keywords: &[&str]) -> bool {
    let text_lower = text.to_lowercase();
    let matches = keywords.iter()
        .filter(|kw| text_lower.contains(&kw.to_lowercase()))
        .count();
    matches >= 2
}

/// Calculate Recall@K
fn recall_at_k(results: &[(String, f32)], relevant_ids: &[String], k: usize) -> f64 {
    if relevant_ids.is_empty() {
        return 0.0;
    }
    let retrieved_relevant = results.iter()
        .take(k)
        .filter(|(id, _)| relevant_ids.contains(id))
        .count();
    retrieved_relevant as f64 / relevant_ids.len() as f64
}

/// Calculate Precision@K
fn precision_at_k(results: &[(String, f32)], relevant_ids: &[String], k: usize) -> f64 {
    if results.is_empty() || k == 0 {
        return 0.0;
    }
    let retrieved_relevant = results.iter()
        .take(k)
        .filter(|(id, _)| relevant_ids.contains(id))
        .count();
    retrieved_relevant as f64 / k.min(results.len()) as f64
}

/// Calculate MRR (Mean Reciprocal Rank)
fn mrr(results: &[(String, f32)], relevant_ids: &[String]) -> f64 {
    for (i, (id, _)) in results.iter().enumerate() {
        if relevant_ids.contains(id) {
            return 1.0 / (i + 1) as f64;
        }
    }
    0.0
}

/// Calculate NDCG@K
fn ndcg_at_k(results: &[(String, f32)], relevant_ids: &[String], k: usize) -> f64 {
    if relevant_ids.is_empty() {
        return 0.0;
    }

    // DCG
    let dcg: f64 = results.iter()
        .take(k)
        .enumerate()
        .map(|(i, (id, _))| {
            let rel = if relevant_ids.contains(id) { 1.0 } else { 0.0 };
            rel / ((i + 2) as f64).log2()
        })
        .sum();

    // IDCG (ideal DCG)
    let ideal_k = k.min(relevant_ids.len());
    let idcg: f64 = (0..ideal_k)
        .map(|i| 1.0 / ((i + 2) as f64).log2())
        .sum();

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

#[derive(Debug)]
struct QueryMetrics {
    query: String,
    num_relevant: usize,
    latency_no_rerank: f64,
    latency_rerank: f64,
    recall_1_no: f64,
    recall_1_yes: f64,
    recall_10_no: f64,
    recall_10_yes: f64,
    precision_1_no: f64,
    precision_1_yes: f64,
    precision_10_no: f64,
    precision_10_yes: f64,
    ndcg_1_no: f64,
    ndcg_1_yes: f64,
    ndcg_10_no: f64,
    ndcg_10_yes: f64,
    mrr_no: f64,
    mrr_yes: f64,
    order_changed: bool,
    top1_changed: bool,
}

#[test]
fn test_reranker_effectiveness() {
    // Setup paths
    let index_dir = std::path::PathBuf::from("index_wikipedia_10k");
    let corpus_path = std::path::PathBuf::from("data/wikipedia_10k.jsonl");
    let model_path = std::path::PathBuf::from("model_quantized/model.onnx");
    let tokenizer_path = std::path::PathBuf::from("model_quantized/tokenizer.json");

    // Check if paths exist
    if !index_dir.exists() {
        eprintln!("Index directory not found: {:?}", index_dir);
        eprintln!("Run this test from the sbng-core directory");
        panic!("Index not found");
    }

    if !corpus_path.exists() {
        eprintln!("Corpus not found: {:?}", corpus_path);
        panic!("Corpus not found");
    }

    if !model_path.exists() {
        eprintln!("Model not found: {:?}", model_path);
        eprintln!("Skipping reranker test");
        return;
    }

    println!("\n{}", "=".repeat(80));
    println!("RERANKER EFFECTIVENESS TEST");
    println!("{}\n", "=".repeat(80));

    // Load corpus
    println!("Loading corpus from {:?}...", corpus_path);
    let corpus = JsonlCorpus::new(corpus_path.to_string_lossy().to_string());
    let mut corpus_map: HashMap<String, String> = HashMap::new();

    for doc_res in corpus.iter().unwrap() {
        let doc = doc_res.unwrap();
        corpus_map.insert(doc.id, doc.text);
    }
    println!("Loaded {} documents\n", corpus_map.len());

    // Load index
    println!("Loading index from {:?}...", index_dir);
    let (metadata, interner, concept_fps, doc_index) =
        persistence::load_index(&index_dir).unwrap();
    println!("Index loaded: {} docs, {} concepts\n", doc_index.doc_ids.len(), concept_fps.len());

    // Create extractor
    let tokenizer = WhitespaceTokenizer;
    let stopwords = ["the", "is", "a", "an", "of", "and", "or", "for", "to", "in", "at", "by"];
    let extractor = Arc::new(InterningConceptExtractor::new(
        tokenizer,
        interner.clone(),
        &stopwords,
    ));

    // Load reranker
    println!("Loading reranker from {:?}...", model_path);
    let reranker = Arc::new(
        sbng_core::search::reranker::Reranker::new(&model_path, &tokenizer_path).unwrap()
    );
    println!("Reranker loaded\n");

    // Create query engine with reranker
    let query_engine = QueryEngine::new(
        extractor,
        &concept_fps,
        &doc_index,
        metadata.config.doc_bloom.bloom_bits,
        metadata.config.doc_bloom.bloom_hashes,
        Some(reranker.clone()),
    );

    // Run evaluation
    println!("Evaluating on {} queries...\n", TEST_QUERIES.len());
    let mut all_metrics = Vec::new();

    for (i, test_query) in TEST_QUERIES.iter().enumerate() {
        print!("[{}/{}] Evaluating: '{}'", i + 1, TEST_QUERIES.len(), test_query.query);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        // Find relevant documents
        let relevant_ids: Vec<String> = corpus_map.iter()
            .filter(|(_, text)| is_relevant(text, &test_query.keywords))
            .map(|(id, _)| id.clone())
            .collect();

        if relevant_ids.is_empty() {
            println!(" - No relevant docs found, skipping");
            continue;
        }

        // Query WITHOUT reranking
        let start = Instant::now();
        let results_no = query_engine.search_with_rerank(
            test_query.query,
            50,
            false,
            &|doc_id: &str| corpus_map.get(doc_id).cloned(),
        ).unwrap();
        let latency_no = start.elapsed().as_secs_f64() * 1000.0;

        // Query WITH reranking
        let start = Instant::now();
        let results_yes = query_engine.search_with_rerank(
            test_query.query,
            10,
            true,
            &|doc_id: &str| corpus_map.get(doc_id).cloned(),
        ).unwrap();
        let latency_yes = start.elapsed().as_secs_f64() * 1000.0;

        // Calculate metrics
        let metrics = QueryMetrics {
            query: test_query.query.to_string(),
            num_relevant: relevant_ids.len(),
            latency_no_rerank: latency_no,
            latency_rerank: latency_yes,

            recall_1_no: recall_at_k(&results_no, &relevant_ids, 1),
            recall_1_yes: recall_at_k(&results_yes, &relevant_ids, 1),
            recall_10_no: recall_at_k(&results_no, &relevant_ids, 10),
            recall_10_yes: recall_at_k(&results_yes, &relevant_ids, 10),

            precision_1_no: precision_at_k(&results_no, &relevant_ids, 1),
            precision_1_yes: precision_at_k(&results_yes, &relevant_ids, 1),
            precision_10_no: precision_at_k(&results_no, &relevant_ids, 10),
            precision_10_yes: precision_at_k(&results_yes, &relevant_ids, 10),

            ndcg_1_no: ndcg_at_k(&results_no, &relevant_ids, 1),
            ndcg_1_yes: ndcg_at_k(&results_yes, &relevant_ids, 1),
            ndcg_10_no: ndcg_at_k(&results_no, &relevant_ids, 10),
            ndcg_10_yes: ndcg_at_k(&results_yes, &relevant_ids, 10),

            mrr_no: mrr(&results_no, &relevant_ids),
            mrr_yes: mrr(&results_yes, &relevant_ids),

            order_changed: {
                let ids_no: Vec<_> = results_no.iter().take(10).map(|(id, _)| id).collect();
                let ids_yes: Vec<_> = results_yes.iter().take(10).map(|(id, _)| id).collect();
                ids_no != ids_yes
            },
            top1_changed: !results_no.is_empty() && !results_yes.is_empty()
                && results_no[0].0 != results_yes[0].0,
        };

        println!(" ✓ (relevant: {}, top1_changed: {})",
                 metrics.num_relevant, metrics.top1_changed);
        all_metrics.push(metrics);
    }

    println!("\n✓ Successfully evaluated {} queries\n", all_metrics.len());

    // Aggregate metrics
    let n = all_metrics.len() as f64;

    let avg_latency_no = all_metrics.iter().map(|m| m.latency_no_rerank).sum::<f64>() / n;
    let avg_latency_yes = all_metrics.iter().map(|m| m.latency_rerank).sum::<f64>() / n;

    let avg_recall_1_no = all_metrics.iter().map(|m| m.recall_1_no).sum::<f64>() / n;
    let avg_recall_1_yes = all_metrics.iter().map(|m| m.recall_1_yes).sum::<f64>() / n;
    let avg_recall_10_no = all_metrics.iter().map(|m| m.recall_10_no).sum::<f64>() / n;
    let avg_recall_10_yes = all_metrics.iter().map(|m| m.recall_10_yes).sum::<f64>() / n;

    let avg_precision_1_no = all_metrics.iter().map(|m| m.precision_1_no).sum::<f64>() / n;
    let avg_precision_1_yes = all_metrics.iter().map(|m| m.precision_1_yes).sum::<f64>() / n;
    let avg_precision_10_no = all_metrics.iter().map(|m| m.precision_10_no).sum::<f64>() / n;
    let avg_precision_10_yes = all_metrics.iter().map(|m| m.precision_10_yes).sum::<f64>() / n;

    let avg_ndcg_1_no = all_metrics.iter().map(|m| m.ndcg_1_no).sum::<f64>() / n;
    let avg_ndcg_1_yes = all_metrics.iter().map(|m| m.ndcg_1_yes).sum::<f64>() / n;
    let avg_ndcg_10_no = all_metrics.iter().map(|m| m.ndcg_10_no).sum::<f64>() / n;
    let avg_ndcg_10_yes = all_metrics.iter().map(|m| m.ndcg_10_yes).sum::<f64>() / n;

    let avg_mrr_no = all_metrics.iter().map(|m| m.mrr_no).sum::<f64>() / n;
    let avg_mrr_yes = all_metrics.iter().map(|m| m.mrr_yes).sum::<f64>() / n;

    let order_changed_pct = all_metrics.iter().filter(|m| m.order_changed).count() as f64 / n * 100.0;
    let top1_changed_pct = all_metrics.iter().filter(|m| m.top1_changed).count() as f64 / n * 100.0;

    // Warmup analysis
    let first_latency = all_metrics[0].latency_rerank;
    let rest_latencies: Vec<_> = all_metrics.iter().skip(1).map(|m| m.latency_rerank).collect();
    let avg_rest = rest_latencies.iter().sum::<f64>() / rest_latencies.len() as f64;

    // Print results
    println!("{}", "=".repeat(80));
    println!("RESULTS");
    println!("{}", "=".repeat(80));

    println!("\n📊 RETRIEVAL METRICS");
    println!("{}", "-".repeat(80));
    println!("{:<20} {:<15} {:<15} {:<15}", "Metric", "No Rerank", "With Rerank", "Improvement");
    println!("{}", "-".repeat(80));

    println!("{:<20} {:<15.4} {:<15.4} {:>+13.2}%",
             "recall@1", avg_recall_1_no, avg_recall_1_yes,
             ((avg_recall_1_yes - avg_recall_1_no) / avg_recall_1_no * 100.0));
    println!("{:<20} {:<15.4} {:<15.4} {:>+13.2}%",
             "recall@10", avg_recall_10_no, avg_recall_10_yes,
             ((avg_recall_10_yes - avg_recall_10_no) / avg_recall_10_no * 100.0));
    println!("{:<20} {:<15.4} {:<15.4} {:>+13.2}%",
             "precision@1", avg_precision_1_no, avg_precision_1_yes,
             ((avg_precision_1_yes - avg_precision_1_no) / avg_precision_1_no * 100.0));
    println!("{:<20} {:<15.4} {:<15.4} {:>+13.2}%",
             "precision@10", avg_precision_10_no, avg_precision_10_yes,
             ((avg_precision_10_yes - avg_precision_10_no) / avg_precision_10_no * 100.0));
    println!("{:<20} {:<15.4} {:<15.4} {:>+13.2}%",
             "ndcg@1", avg_ndcg_1_no, avg_ndcg_1_yes,
             ((avg_ndcg_1_yes - avg_ndcg_1_no) / avg_ndcg_1_no * 100.0));
    println!("{:<20} {:<15.4} {:<15.4} {:>+13.2}%",
             "ndcg@10", avg_ndcg_10_no, avg_ndcg_10_yes,
             ((avg_ndcg_10_yes - avg_ndcg_10_no) / avg_ndcg_10_no * 100.0));
    println!("{:<20} {:<15.4} {:<15.4} {:>+13.2}%",
             "MRR", avg_mrr_no, avg_mrr_yes,
             ((avg_mrr_yes - avg_mrr_no) / avg_mrr_no * 100.0));

    println!("\n⚡ LATENCY METRICS");
    println!("{}", "-".repeat(80));
    println!("No Rerank (mean):     {:>8.2} ms", avg_latency_no);
    println!("With Rerank (mean):   {:>8.2} ms", avg_latency_yes);
    println!("Rerank Overhead:      {:>8.2} ms", avg_latency_yes - avg_latency_no);

    println!("\n🔥 WARMUP ANALYSIS");
    println!("{}", "-".repeat(80));
    println!("First query latency:  {:>8.2} ms", first_latency);
    println!("Avg rest queries:     {:>8.2} ms", avg_rest);
    println!("Warmup overhead:      {:>8.2} ms ({:.1}%)",
             first_latency - avg_rest,
             (first_latency - avg_rest) / avg_rest * 100.0);

    println!("\n🔄 RANKING CHANGES");
    println!("{}", "-".repeat(80));
    println!("Order changed (top-10): {:.1}%", order_changed_pct);
    println!("Top-1 result changed:   {:.1}%", top1_changed_pct);

    println!("\n💡 SUMMARY");
    println!("{}", "-".repeat(80));
    let avg_ndcg_improvement = ((avg_ndcg_1_yes - avg_ndcg_1_no) / avg_ndcg_1_no
                                 + (avg_ndcg_10_yes - avg_ndcg_10_no) / avg_ndcg_10_no) / 2.0 * 100.0;
    println!("Average NDCG Improvement: {:>+8.2}%", avg_ndcg_improvement);
    println!("MRR Improvement:          {:>+8.2}%", (avg_mrr_yes - avg_mrr_no) / avg_mrr_no * 100.0);

    if avg_ndcg_improvement > 30.0 {
        println!("\n✅ CONCLUSION: Re-ranker provides SIGNIFICANT quality improvements!");
    } else if avg_ndcg_improvement > 15.0 {
        println!("\n✅ CONCLUSION: Re-ranker provides MODERATE quality improvements.");
    } else {
        println!("\n⚠️  CONCLUSION: Re-ranker provides MINOR quality improvements.");
    }

    println!("{}\n", "=".repeat(80));

    // Assert that reranker improves quality
    assert!(avg_ndcg_10_yes > avg_ndcg_10_no,
            "Reranker should improve NDCG@10: {} vs {}",
            avg_ndcg_10_yes, avg_ndcg_10_no);
    assert!(avg_precision_10_yes > avg_precision_10_no,
            "Reranker should improve Precision@10: {} vs {}",
            avg_precision_10_yes, avg_precision_10_no);
}
