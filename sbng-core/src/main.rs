use std::path::PathBuf;
use std::sync::Arc;
use std::collections::HashMap;
use std::io::{self, Write};

use clap::{Parser, Subcommand};
use sbng_core::{
    SbngConfig,
    corpus::{JsonlCorpus, WhitespaceTokenizer, InterningConceptExtractor, ConceptInterner},
    pipeline::{GraphBuildPipeline, SignatureGenerationPipeline},
    search::{DocIndex, QueryEngine},
    persistence,
    BloomFingerprint,
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
    let graph = graph_pipeline.build_from_jsonl(&corpus)?;

    println!(
        "Graph built: {} nodes, {} edges",
        graph.inner().node_count(),
        graph.inner().edge_count()
    );

    // 5) Concept fingerprints
    println!("Generating concept signatures...");
    let sig_pipeline = SignatureGenerationPipeline::new(config.clone());
    let concept_fps_vec = sig_pipeline.generate(&graph)?;
    let concept_fps: HashMap<_, _> = concept_fps_vec.into_iter().collect();

    // 6) Document fingerprints (DocIndex)
    println!("Building Document Index...");
    let (doc_ids, doc_fps) = build_doc_fingerprints(&corpus, &extractor, &concept_fps, &config)?;
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
        metadata.config.bloom_bits,
        metadata.config.bloom_hashes,
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
        metadata.config.bloom_bits,
        metadata.config.bloom_hashes,
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

/// Build per-document fingerprints by merging concept fingerprints.
fn build_doc_fingerprints(
    corpus: &JsonlCorpus,
    extractor: &dyn sbng_core::corpus::ConceptExtractor,
    concept_fps: &HashMap<sbng_core::types::ConceptId, BloomFingerprint>,
    config: &SbngConfig,
) -> anyhow::Result<(Vec<String>, Vec<BloomFingerprint>)> {
    let mut doc_ids = Vec::new();
    let mut doc_fps = Vec::new();

    for doc_res in corpus.iter()? {
        let doc = doc_res?;
        let concepts = extractor.extract_concepts(&doc.text);

        let mut fp = BloomFingerprint::new(config.bloom_bits, config.bloom_hashes);

        for c in concepts {
            if let Some(cfp) = concept_fps.get(&c.concept_id) {
                fp.merge(cfp);
            } else {
                // Fallback: insert concept ID directly if not in graph
                fp.insert_concept(c.concept_id);
            }
        }

        doc_ids.push(doc.id);
        doc_fps.push(fp);
    }

    Ok((doc_ids, doc_fps))
}
