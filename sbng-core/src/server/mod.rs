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
}

/// Request payload for query endpoint.
#[derive(Debug, Deserialize)]
pub struct QueryRequest {
    /// The query string.
    pub q: String,
    /// Number of results to return.
    #[serde(default = "default_k")]
    pub k: usize,
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
pub async fn start_server(index_dir: PathBuf, port: u16) -> anyhow::Result<()> {
    // 1. Load index
    tracing::info!("Loading index from {}...", index_dir.display());
    let (metadata, interner, concept_fps, doc_index) = persistence::load_index(&index_dir)?;
    tracing::info!(
        "Index loaded. {} docs, {} concepts.",
        doc_index.doc_ids.len(),
        concept_fps.len()
    );

    // 2. Recreate extractor
    let tokenizer = WhitespaceTokenizer;
    let stopwords = ["the", "is", "a", "an", "of", "and", "or", "for", "to", "in", "at", "by"];
    let extractor = Arc::new(InterningConceptExtractor::new(tokenizer, interner.clone(), &stopwords));

    // 3. Create QueryEngine
    let query_engine = Arc::new(QueryEngine::new(
        extractor,
        &concept_fps,
        &doc_index,
        metadata.config.bloom_bits,
        metadata.config.bloom_hashes,
    ));

    // 4. Setup App State
    let state = AppState { query_engine };

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
    let results = state.query_engine.search(&payload.q, payload.k);

    let response_results = results
        .into_iter()
        .map(|(doc_id, score)| QueryResult { doc_id, score })
        .collect();

    (
        StatusCode::OK,
        Json(QueryResponse {
            results: response_results,
        }),
    )
}
