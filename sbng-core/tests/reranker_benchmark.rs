/// Benchmark different reranker configurations to find optimal settings
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use sbng_core::{
    corpus::{InterningConceptExtractor, WhitespaceTokenizer, JsonlCorpus},
    persistence,
    search::QueryEngine,
};

struct BenchmarkConfig {
    name: &'static str,
    candidates_multiplier: usize,
    candidates_min: usize,
}

const CONFIGS: &[BenchmarkConfig] = &[
    BenchmarkConfig { name: "Current (5x, min 50)", candidates_multiplier: 5, candidates_min: 50 },
    BenchmarkConfig { name: "Aggressive (3x, min 30)", candidates_multiplier: 3, candidates_min: 30 },
    BenchmarkConfig { name: "Minimal (2x, min 20)", candidates_multiplier: 2, candidates_min: 20 },
    BenchmarkConfig { name: "Ultra-Fast (1.5x, min 15)", candidates_multiplier: 15, candidates_min: 10 }, // 1.5x = 15/10
];

const TEST_QUERIES: &[&str] = &[
    "quantum mechanics physics",
    "artificial intelligence machine learning",
    "climate change global warming",
    "DNA genetics heredity",
    "computer programming software",
    "world war two hitler",
    "roman empire caesar",
    "egyptian pyramids pharaoh",
    "christopher columbus america",
    "leonardo da vinci painting",
];

#[test]
fn benchmark_reranker_configs() {
    // Setup paths
    let index_dir = std::path::PathBuf::from("index_wikipedia_10k");
    let corpus_path = std::path::PathBuf::from("data/wikipedia_10k.jsonl");
    let model_path = std::path::PathBuf::from("model_quantized/model.onnx");
    let tokenizer_path = std::path::PathBuf::from("model_quantized/tokenizer.json");

    if !index_dir.exists() || !corpus_path.exists() || !model_path.exists() {
        eprintln!("Required files not found, skipping benchmark");
        return;
    }

    println!("\n{}", "=".repeat(100));
    println!("RERANKER CONFIGURATION BENCHMARK");
    println!("{}\n", "=".repeat(100));

    // Load corpus
    println!("Loading corpus...");
    let corpus = JsonlCorpus::new(corpus_path.to_string_lossy().to_string());
    let mut corpus_map: HashMap<String, String> = HashMap::new();
    for doc_res in corpus.iter().unwrap() {
        let doc = doc_res.unwrap();
        corpus_map.insert(doc.id, doc.text);
    }
    let corpus_map = Arc::new(corpus_map);
    println!("Loaded {} documents\n", corpus_map.len());

    // Load index
    println!("Loading index...");
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
    println!("Loading reranker...");
    let reranker = Arc::new(
        sbng_core::search::reranker::Reranker::new(&model_path, &tokenizer_path).unwrap()
    );
    println!("Reranker loaded\n");

    // Create query engine
    let query_engine = QueryEngine::new(
        extractor,
        &concept_fps,
        &doc_index,
        metadata.config.doc_bloom.bloom_bits,
        metadata.config.doc_bloom.bloom_hashes,
        Some(reranker.clone()),
    );

    println!("{}", "=".repeat(100));
    println!("BENCHMARKING DIFFERENT CANDIDATE COUNTS");
    println!("{}\n", "=".repeat(100));

    for config in CONFIGS {
        println!("Testing: {}", config.name);
        println!("{}", "-".repeat(100));

        let mut total_latency = 0.0;
        let mut total_candidates = 0;
        let mut first_query_latency = 0.0;

        for (i, query) in TEST_QUERIES.iter().enumerate() {
            let top_k = 10;
            let candidates_k = std::cmp::max(
                (top_k * config.candidates_multiplier) / 10,
                config.candidates_min
            );

            // Retrieve candidates
            let initial_results = query_engine.search(query, candidates_k);

            // Prepare candidates for reranking
            let mut candidates = Vec::new();
            for (doc_id, _) in &initial_results {
                if let Some(text) = corpus_map.get(doc_id) {
                    candidates.push((doc_id.clone(), text.clone()));
                }
            }

            // Time the reranking
            let start = Instant::now();
            let _reranked = reranker.rerank(query, candidates.clone()).unwrap();
            let latency = start.elapsed().as_secs_f64() * 1000.0;

            if i == 0 {
                first_query_latency = latency;
            }
            total_latency += latency;
            total_candidates += candidates.len();

            print!(".");
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        }
        println!();

        let avg_latency = total_latency / TEST_QUERIES.len() as f64;
        let avg_candidates = total_candidates as f64 / TEST_QUERIES.len() as f64;
        let steady_state_latency = (total_latency - first_query_latency) / (TEST_QUERIES.len() - 1) as f64;

        println!("\nResults:");
        println!("  Avg candidates reranked:  {:.1}", avg_candidates);
        println!("  First query latency:      {:.2} ms", first_query_latency);
        println!("  Avg latency (all):        {:.2} ms", avg_latency);
        println!("  Avg latency (steady):     {:.2} ms", steady_state_latency);
        println!("  Latency per candidate:    {:.2} ms\n", steady_state_latency / avg_candidates);
    }

    println!("{}", "=".repeat(100));
    println!("\n💡 DETAILED BREAKDOWN");
    println!("{}", "-".repeat(100));

    // Detailed single-query breakdown
    let query = TEST_QUERIES[0];
    println!("\nAnalyzing query: '{}'", query);

    for config in &[
        BenchmarkConfig { name: "50 candidates", candidates_multiplier: 0, candidates_min: 50 },
        BenchmarkConfig { name: "30 candidates", candidates_multiplier: 0, candidates_min: 30 },
        BenchmarkConfig { name: "20 candidates", candidates_multiplier: 0, candidates_min: 20 },
        BenchmarkConfig { name: "15 candidates", candidates_multiplier: 0, candidates_min: 15 },
        BenchmarkConfig { name: "10 candidates", candidates_multiplier: 0, candidates_min: 10 },
    ] {
        let candidates_k = config.candidates_min;

        // Get candidates
        let initial_results = query_engine.search(query, candidates_k);
        let mut candidates = Vec::new();
        for (doc_id, _) in &initial_results {
            if let Some(text) = corpus_map.get(doc_id) {
                candidates.push((doc_id.clone(), text.clone()));
            }
        }

        // Warmup
        let _ = reranker.rerank(query, candidates.clone()).unwrap();

        // Measure 3 times
        let mut latencies = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            let _ = reranker.rerank(query, candidates.clone()).unwrap();
            latencies.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        let avg = latencies.iter().sum::<f64>() / latencies.len() as f64;
        let per_candidate = avg / candidates.len() as f64;

        println!("  {:<20} → {:.2} ms ({:.2} ms/candidate)",
                 config.name, avg, per_candidate);
    }

    println!("\n{}", "=".repeat(100));
}
