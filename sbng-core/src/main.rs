use std::path::PathBuf;
use std::sync::Arc;
use std::collections::HashMap;
use std::io::{self, Write};

use clap::{Parser, Subcommand};
use sbng_core::{
    SbngConfig,
    corpus::{JsonlCorpus, WhitespaceTokenizer, InterningConceptExtractor, ConceptInterner, ConceptExtractor},
    pipeline::{GraphBuildPipeline, SignatureGenerationPipeline},
    search::{DocIndex, QueryEngine},
    persistence,
    types::ConceptId,
};

#[derive(Parser, Debug)]
#[command(name = "sbng", about = "Sparse Binary Neural Graph CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Build an index from a corpus and persist it.
    Index {
        /// Path to the input corpus (JSONL)
        #[arg(long)]
        corpus: PathBuf,
        /// Output directory for the index
        #[arg(long)]
        index_dir: PathBuf,
        /// Path to config file (JSON)
        #[arg(long)]
        config: Option<PathBuf>,
    },

    /// Run a single query against an existing index.
    Query {
        /// Path to the index directory
        #[arg(long)]
        index_dir: PathBuf,
        /// Query string
        #[arg(long)]
        q: String,
        /// Number of results to return
        #[arg(long, default_value_t = 10)]
        top_k: usize,
    },

    /// Interactive query REPL against an index.
    Repl {
        /// Path to the index directory
        #[arg(long)]
        index_dir: PathBuf,
    },

    /// Start HTTP API server.
    Serve {
        /// Path to the index directory
        #[arg(long)]
        index_dir: PathBuf,
        /// Port to listen on
        #[arg(long, default_value_t = 3000)]
        port: u16,
    },

    /// Diagnose index health (graph stats, Bloom fill rates, hubs).
    Diagnose {
        /// Path to the index directory
        #[arg(long)]
        index_dir: PathBuf,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Index { corpus, index_dir, config } => {
            cmd_index(corpus, index_dir, config)?;
        }
        Commands::Query { index_dir, q, top_k } => {
            cmd_query(index_dir, &q, top_k)?;
        }
        Commands::Repl { index_dir } => {
            cmd_repl(index_dir)?;
        }
        Commands::Serve { index_dir, port } => {
            sbng_core::server::start_server(index_dir, port).await?;
        }
        Commands::Diagnose { index_dir } => {
            cmd_diagnose(index_dir)?;
        }
    }

    Ok(())
}

fn cmd_index(corpus_path: PathBuf, index_dir: PathBuf, config_path: Option<PathBuf>) -> anyhow::Result<()> {
    // 1) Load config
    let mut config = if let Some(path) = config_path {
        let s = std::fs::read_to_string(path)?;
        serde_json::from_str(&s)?
    } else {
        SbngConfig::default()
    };

    // Adjust for small sample automatically if "sample" is in path (convenience)
    if corpus_path.to_string_lossy().contains("sample") {
        println!("Detected sample corpus, adjusting config for small data...");
        config.cooccur_min = 1;
        config.pmi_min = 0.0;
        config.min_degree = 1;
    }

    println!("Using config: {:?}", config);

    // 2) Shared interner
    let interner = Arc::new(ConceptInterner::with_hasher(sbng_core::corpus::interner::SerializableHasher::default()));

    // 3) Tokenizer + extractor
    let tokenizer = WhitespaceTokenizer;
    let stopwords = ["the", "is", "a", "an", "of", "and", "or", "for", "to", "in", "at", "by"];
    let extractor = InterningConceptExtractor::new(
        tokenizer,
        interner.clone(),
        &stopwords,
    );

    // 4) Build graph
    let corpus_str = corpus_path.to_string_lossy().into_owned();
    let corpus = JsonlCorpus::new(corpus_str.clone());
    
    println!("Building graph from {}...", corpus_path.display());
    let graph_pipeline = GraphBuildPipeline::new(config.clone(), &extractor, interner.clone());
    let mut graph = graph_pipeline.build_from_jsonl(&corpus)?;

    println!(
        "Graph built: {} nodes, {} edges",
        graph.inner().node_count(),
        graph.inner().edge_count()
    );

    // 4b) Recompute degrees after pruning
    println!("Computing node degrees...");
    graph.recompute_degrees();

    // 5) Concept fingerprints
    println!("Generating concept signatures...");
    let sig_pipeline = SignatureGenerationPipeline::new(config.clone());
    let concept_fps_vec = sig_pipeline.generate(&graph)?;
    let concept_fps: HashMap<_, _> = concept_fps_vec.into_iter().collect();

    // 6) Document fingerprints (DocIndex) - OPTIMIZED: TF-IDF × degree penalty with precomputed stats
    println!("Building Document Index...");
    
    // 6a) Collect docs map and doc frequencies
    let mut docs_map: HashMap<String, Vec<ConceptId>> = HashMap::new();
    let mut doc_freqs: HashMap<ConceptId, u32> = HashMap::new();
    let mut total_docs = 0usize;

    for doc_res in corpus.iter()? {
        let doc = doc_res?;
        total_docs += 1;

        let concepts = extractor.extract_concepts(&doc.text);
        let ids: Vec<_> = concepts.iter().map(|c| c.concept_id).collect();

        // Track document frequency
        let mut seen = std::collections::HashSet::new();
        for &cid in &ids {
            if seen.insert(cid) {
                *doc_freqs.entry(cid).or_insert(0) += 1;
            }
        }

        docs_map.insert(doc.id.clone(), ids);
    }

    // 6b) Precompute concept stats (IDF, degree_penalty)
    println!("   -> Precomputing concept statistics...");
    let concept_stats = sbng_core::search::concept_stats::compute_concept_stats(
        total_docs,
        &doc_freqs,
        &graph,
    );

    // 6c) Build doc fingerprints in parallel
    let doc_fps_map = sbng_core::search::doc_fingerprints::build_doc_fingerprints_parallel(
        &docs_map,
        &config.doc_bloom,
        &concept_stats,
    );

    // 6d) Convert to DocIndex
    let mut doc_ids = Vec::with_capacity(doc_fps_map.len());
    let mut doc_fps = Vec::with_capacity(doc_fps_map.len());

    for (doc_id, fp) in doc_fps_map {
        doc_ids.push(doc_id);
        doc_fps.push(fp);
    }

    let doc_index = DocIndex {
        doc_ids,
        fingerprints: doc_fps,
    };
    println!("Indexed {} documents.", doc_index.doc_ids.len());

    // 7) Metadata
    let metadata = persistence::IndexMetadata::new(&config);

    // 8) Save to index_dir
    println!("Saving index to {}...", index_dir.display());
    persistence::save_index(
        &index_dir,
        &metadata,
        &interner,
        &concept_fps,
        &doc_index,
    )?;

    println!("Index built and saved successfully.");
    Ok(())
}

fn cmd_query(index_dir: PathBuf, q: &str, top_k: usize) -> anyhow::Result<()> {
    // 1) Load index via persistence
    let (metadata, interner, concept_fps, doc_index) =
        persistence::load_index(&index_dir)?;

    // 2) Recreate extractor (same stopwords & tokenizer)
    let tokenizer = WhitespaceTokenizer;
    let stopwords = ["the", "is", "a", "an", "of", "and", "or", "for", "to", "in", "at", "by"];

    let extractor = Arc::new(InterningConceptExtractor::new(
        tokenizer,
        interner.clone(),
        &stopwords,
    ));

    // 3) Query engine
    let qe = QueryEngine::new(
        extractor,
        &concept_fps,
        &doc_index,
        metadata.config.concept_bloom.bloom_bits,
        metadata.config.concept_bloom.bloom_hashes,
    );

    // 4) Run query
    let results = qe.search(q, top_k);

    println!("Query: '{}'", q);
    if results.is_empty() {
        println!("  No results found.");
    } else {
        for (doc_id, score) in results {
            println!("  DocID: {}, Score: {}", doc_id, score);
        }
    }

    Ok(())
}

fn cmd_repl(index_dir: PathBuf) -> anyhow::Result<()> {
    // 1) Load index via persistence
    println!("Loading index from {}...", index_dir.display());
    let (metadata, interner, concept_fps, doc_index) =
        persistence::load_index(&index_dir)?;
    println!("Index loaded. {} docs, {} concepts.", doc_index.doc_ids.len(), concept_fps.len());

    // 2) Recreate extractor
    let tokenizer = WhitespaceTokenizer;
    let stopwords = ["the", "is", "a", "an", "of", "and", "or", "for", "to", "in", "at", "by"];
    let extractor = Arc::new(InterningConceptExtractor::new(
        tokenizer,
        interner.clone(),
        &stopwords,
    ));

    // 3) Query engine
    let qe = QueryEngine::new(
        extractor,
        &concept_fps,
        &doc_index,
        metadata.config.concept_bloom.bloom_bits,
        metadata.config.concept_bloom.bloom_hashes,
    );

    println!("SBNG REPL. Type a query, or just press Enter to exit.");

    let stdin = io::stdin();
    loop {
        print!("sbng> ");
        io::stdout().flush()?;

        let mut buf = String::new();
        let n = stdin.read_line(&mut buf)?;
        if n == 0 {
            break;
        }
        let q = buf.trim();
        if q.is_empty() {
            break;
        }

        let results = qe.search(q, 10);
        if results.is_empty() {
            println!("  No results found.");
        } else {
            for (doc_id, score) in results {
                println!("  DocID: {}, Score: {}", doc_id, score);
            }
        }
    }

    Ok(())
}

/// Diagnose index health: graph statistics, Bloom fill rates, hub analysis.
fn cmd_diagnose(index_dir: PathBuf) -> anyhow::Result<()> {
    println!("=== SBNG Index Diagnostics ===\n");
    println!("Loading index from {}...", index_dir.display());

    // Load index
    let (metadata, interner, concept_fps, doc_index) = persistence::load_index(&index_dir)?;
    
    println!("Index loaded successfully.\n");
    println!("=== Configuration ===");
    println!("  Concept Bloom bits: {}", metadata.config.concept_bloom.bloom_bits);
    println!("  Concept Bloom hashes: {}", metadata.config.concept_bloom.bloom_hashes);
    println!("  Concept max neighborhood: {}", metadata.config.concept_bloom.max_neighborhood);
    println!("  Doc Bloom bits: {}", metadata.config.doc_bloom.bloom_bits);
    println!("  Doc Bloom hashes: {}", metadata.config.doc_bloom.bloom_hashes);
    println!("  Doc Top-K: {}", metadata.config.doc_bloom.max_neighborhood);
    println!("  PMI min: {}", metadata.config.pmi_min);
    println!("  Cooccur min: {}", metadata.config.cooccur_min);
    println!("  Max degree: {}", metadata.config.max_degree);
    println!();

    // === Graph Statistics ===
    println!("=== Graph Statistics ===");
    println!("  Total concepts (nodes): {}", concept_fps.len());
    println!("  Total documents: {}", doc_index.doc_ids.len());
    
    // === Bloom Fill Rate Analysis ===
    println!("\n=== Bloom Filter Analysis ===");
    
    let mut fill_rates: Vec<f64> = concept_fps.values()
        .map(|fp| fp.fill_ratio() as f64)
        .collect();
    
    if fill_rates.is_empty() {
        println!("  No concept fingerprints found.");
        return Ok(());
    }
    
    fill_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let avg_fill = fill_rates.iter().sum::<f64>() / fill_rates.len() as f64;
    let min_fill = fill_rates.first().copied().unwrap_or(0.0);
    let max_fill = fill_rates.last().copied().unwrap_or(0.0);
    let median_fill = fill_rates[fill_rates.len() / 2];
    
    println!("  Average fill rate: {:.1}%", avg_fill * 100.0);
    println!("  Median fill rate: {:.1}%", median_fill * 100.0);
    println!("  Min fill rate: {:.1}%", min_fill * 100.0);
    println!("  Max fill rate: {:.1}%", max_fill * 100.0);
    
    // Histogram
    println!("\n  Fill Rate Distribution:");
    let buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)];
    for (low, high) in buckets {
        let count = fill_rates.iter()
            .filter(|&&rate| rate >= low && rate < high)
            .count();
        let pct = (count as f64 / fill_rates.len() as f64) * 100.0;
        let bar = "█".repeat((pct / 2.0) as usize);
        println!("    {:.0}%-{:.0}%: {:>5} ({:>5.1}%) {}", 
            low * 100.0, high * 100.0, count, pct, bar);
    }
    
    // === Document Fingerprint Analysis ===
    println!("\n=== Document Fingerprints ===");
    let doc_fill_rates: Vec<f64> = doc_index.fingerprints.iter()
        .map(|fp| fp.fill_ratio() as f64)
        .collect();
    
    if !doc_fill_rates.is_empty() {
        let avg_doc_fill = doc_fill_rates.iter().sum::<f64>() / doc_fill_rates.len() as f64;
        println!("  Average doc fill rate: {:.1}%", avg_doc_fill * 100.0);
    }
    
    // === Top Concepts by Fingerprint Size ===
    println!("\n=== Top 10 Concepts by Neighborhood Size ===");
    let mut concept_sizes: Vec<(sbng_core::types::ConceptId, usize)> = concept_fps.iter()
        .map(|(id, fp)| (*id, fp.count_set_bits()))
        .collect();
    concept_sizes.sort_by_key(|(_, size)| std::cmp::Reverse(*size));
    
    for (i, (concept_id, size)) in concept_sizes.iter().take(10).enumerate() {
        let concept_str = interner.resolve(&concept_id.0);
        let fill = concept_fps.get(concept_id).map(|fp| fp.fill_ratio()).unwrap_or(0.0);
        println!("  {:2}. {:20} - {} bits set ({:.1}% fill)", 
            i + 1, concept_str, size, fill * 100.0);
    }
    
    // === Health Recommendations ===
    println!("\n=== Health Assessment ===");
    
    if avg_fill > 0.7 {
        println!("  ⚠️  WARNING: Average fill rate is HIGH ({:.1}%)", avg_fill * 100.0);
        println!("     → Neighborhoods may be too large or hubs are leaking in");
        println!("     → Recommendations:");
        println!("       - Increase pmi_min (current: {})", metadata.config.pmi_min);
        println!("       - Increase cooccur_min (current: {})", metadata.config.cooccur_min);
        println!("       - Decrease concept max_neighborhood (current: {})", metadata.config.concept_bloom.max_neighborhood);
    } else if avg_fill > 0.6 {
        println!("  ⚠️  CAUTION: Average fill rate is moderately high ({:.1}%)", avg_fill * 100.0);
        println!("     → Consider tightening pruning parameters");
    } else if avg_fill < 0.1 {
        println!("  ⚠️  WARNING: Average fill rate is LOW ({:.1}%)", avg_fill * 100.0);
        println!("     → Graph may be too sparse, neighborhoods too small");
        println!("     → Recommendations:");
        println!("       - Decrease pmi_min (current: {})", metadata.config.pmi_min);
        println!("       - Decrease cooccur_min (current: {})", metadata.config.cooccur_min);
        println!("       - Increase concept max_neighborhood (current: {})", metadata.config.concept_bloom.max_neighborhood);
    } else {
        println!("  ✓ Fill rate is in healthy range ({:.1}%)", avg_fill * 100.0);
        println!("    Optimal range: 30-60%");
    }
    
    if max_fill > 0.8 {
        println!("\n  ⚠️  WARNING: Some concepts have very high fill rates (max: {:.1}%)", max_fill * 100.0);
        println!("     → These may be hubs that should be suppressed");
        println!("     → Check max_degree setting (current: {})", metadata.config.max_degree);
    }
    
    println!("\n=== Diagnostics Complete ===");
    
    Ok(())
}
