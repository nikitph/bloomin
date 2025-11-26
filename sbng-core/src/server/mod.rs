use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    routing::post,
    Json, Router,
};
use serde::{Deserialize, Serialize};
use tower_http::trace::TraceLayer;

use crate::{
    corpus::{InterningConceptExtractor, WhitespaceTokenizer},
    persistence,
    search::QueryEngine,
};

/// Shared state for the server.
#[derive(Clone)]
struct AppState {
    query_engine: Arc<QueryEngine>,
    corpus_map: Arc<HashMap<String, String>>,
}

/// Request payload for query endpoint.
#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    /// The query string.
    pub q: String,
    /// Number of results to return.
    #[serde(default = "default_k")]
    pub k: usize,
    /// Whether to re-rank results.
    #[serde(default)]
    pub rerank: bool,
}

fn default_k() -> usize {
    10
}

/// Response payload for query endpoint.
#[derive(Debug, Serialize)]
pub struct QueryResponse {
    /// List of matching documents.
    pub results: Vec<QueryResult>,
}

/// A single search result.
#[derive(Debug, Serialize)]
pub struct QueryResult {
    /// Document ID.
    pub doc_id: String,
    /// Similarity score (overlap count).
    pub score: u32,
}

/// Start the HTTP server.
pub async fn start_server(
    index_dir: PathBuf,
    port: u16,
    corpus_path: Option<PathBuf>,
    model_path: Option<PathBuf>,
) -> anyhow::Result<()> {
    // 1. Load index
    tracing::info!("Loading index from {}...", index_dir.display());
    let (metadata, interner, concept_fps, doc_index) = persistence::load_index(&index_dir)?;
    
    tracing::info!("Loaded config: {:?}", metadata.config);
    tracing::info!("Concept Bloom bits: {}", metadata.config.concept_bloom.bloom_bits);
    tracing::info!("Doc Bloom bits: {}", metadata.config.doc_bloom.bloom_bits);
    
    if let Some(first_doc) = doc_index.fingerprints.first() {
        tracing::info!("Actual Doc FP size: {}", first_doc.len());
    }

    tracing::info!(
        "Index loaded. {} docs, {} concepts.",
        doc_index.doc_ids.len(),
        concept_fps.len()
    );

    // 2. Create Tokenizer & Extractor
    let tokenizer = WhitespaceTokenizer;
    let stopwords = ["the", "is", "a", "an", "of", "and", "or", "for", "to", "in", "at", "by"];
    let extractor = Arc::new(InterningConceptExtractor::new(tokenizer, interner.clone(), &stopwords));

    // 3. Load Corpus (if provided)
    let mut corpus_map = HashMap::new();
    if let Some(path) = corpus_path {
        tracing::info!("Loading corpus from {}...", path.display());
        // Simple JSONL loader
        use std::io::BufRead;
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        for line in reader.lines() {
            let line = line?;
            if let Ok(doc) = serde_json::from_str::<crate::corpus::JsonlDoc>(&line) {
                corpus_map.insert(doc.id, doc.text);
            }
        }
        tracing::info!("Loaded {} documents into memory.", corpus_map.len());
    }
    let corpus_map = Arc::new(corpus_map);

    // 4. Load Reranker (if provided)
    let reranker = if let Some(path) = model_path {
        tracing::info!("Loading reranker model from {}...", path.display());
        // Assume tokenizer is in same dir as model or standard path
        // For now, we'll assume tokenizer.json is in the same dir
        let tokenizer_path = path.parent().unwrap().join("tokenizer.json");
        Some(Arc::new(crate::search::reranker::Reranker::new(&path, &tokenizer_path)?))
    } else {
        None
    };

    // 5. Create QueryEngine
    let query_engine = Arc::new(QueryEngine::new(
        extractor,
        &concept_fps,
        &doc_index,
        metadata.config.doc_bloom.bloom_bits,
        metadata.config.doc_bloom.bloom_hashes,
        reranker,
    ));

    // 6. Setup App State
    let state = AppState {
        query_engine,
        corpus_map,
    };

    // 5. Build Router
    let app = Router::new()
        .route("/query", post(post_query))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // 6. Bind and Serve
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    tracing::info!("Listening on {}", addr);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

/// Handler for POST /query
async fn post_query(
    State(state): State<AppState>,
    Json(payload): Json<QueryRequest>,
) -> (StatusCode, Json<QueryResponse>) {
    // Define doc fetcher closure
    let doc_fetcher = |doc_id: &str| -> Option<String> {
        let res = state.corpus_map.get(doc_id).cloned();
        if res.is_none() {
            tracing::warn!("Doc fetcher failed for id: {}", doc_id);
        } else {
            tracing::debug!("Doc fetcher found id: {}", doc_id);
        }
        res
    };

    let results = match state.query_engine.search_with_rerank(
        &payload.q,
        payload.k,
        payload.rerank,
        &doc_fetcher,
    ) {
        Ok(res) => res,
        Err(e) => {
            tracing::error!("Search error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(QueryResponse { results: vec![] }),
            );
        }
    };

    let response_results = results
        .into_iter()
        .map(|(doc_id, score)| {
            // Apply sigmoid to convert logit to probability [0, 1]
            let probability = 1.0 / (1.0 + (-score).exp());
            // Scale to u32 (0-10000)
            let score_u32 = (probability * 10000.0) as u32;
            
            QueryResult {
                doc_id,
                score: score_u32,
            }
        })
        .collect();

    (
        StatusCode::OK,
        Json(QueryResponse {
            results: response_results,
        }),
    )
}
