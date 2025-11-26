use std::path::Path;

/// A neural re-ranker that calls a Python GPU service.
pub struct Reranker {
    service_url: String,
}

impl std::fmt::Debug for Reranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reranker")
            .field("service_url", &self.service_url)
            .finish()
    }
}

impl Reranker {
    /// Create a new re-ranker that calls the Python GPU service.
    pub fn new<P: AsRef<Path>>(_model_path: P, _tokenizer_path: P) -> anyhow::Result<Self> {
        Ok(Self {
            service_url: "http://localhost:8001/rerank".to_string(),
        })
    }

    /// Re-rank a list of (doc_id, score, text) candidates against a query.
    ///
    /// Returns a re-ordered list of (doc_id, new_score).
    pub fn rerank(
        &self,
        query: &str,
        candidates: Vec<(String, String)>, // (doc_id, doc_text)
    ) -> anyhow::Result<Vec<(String, f32)>> {
        tracing::info!("Reranking {} candidates for query: '{}'", candidates.len(), query);
        if candidates.is_empty() {
            return Ok(vec![]);
        }

        // Call Python GPU service via HTTP
        let client = reqwest::blocking::Client::new();
        let response = client
            .post(&self.service_url)
            .json(&serde_json::json!({
                "query": query,
                "candidates": candidates,
            }))
            .send()
            .map_err(|e| anyhow::anyhow!("Failed to call reranker service: {}", e))?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Reranker service returned error: {}", response.status()));
        }

        let result: serde_json::Value = response.json()
            .map_err(|e| anyhow::anyhow!("Failed to parse reranker response: {}", e))?;

        let results = result["results"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("Invalid response format"))?
            .iter()
            .filter_map(|item| {
                let arr = item.as_array()?;
                let doc_id = arr.get(0)?.as_str()?.to_string();
                let score = arr.get(1)?.as_f64()? as f32;
                Some((doc_id, score))
            })
            .collect();

        Ok(results)
    }
}
