//! Document retrieval using Bloom fingerprints.

use std::sync::Arc;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::bloom::BloomFingerprint;
use crate::corpus::ConceptExtractor;
use crate::types::ConceptId;

pub mod doc_fingerprints;
pub mod concept_stats;
/// Neural re-ranker module.
pub mod reranker;
pub mod explainability;

pub use explainability::{MatchedConcept, explain_match};

/// Index storing document fingerprints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocIndex {
    /// Document IDs.
    pub doc_ids: Vec<String>,
    /// Corresponding document fingerprints.
    pub fingerprints: Vec<BloomFingerprint>,
}

impl DocIndex {
    /// Create a new empty index.
    pub fn new() -> Self {
        Self {
            doc_ids: Vec::new(),
            fingerprints: Vec::new(),
        }
    }

    /// Add a document to the index.
    pub fn add(&mut self, doc_id: String, fingerprint: BloomFingerprint) {
        self.doc_ids.push(doc_id);
        self.fingerprints.push(fingerprint);
    }
}

/// Engine for executing queries against the index.
/// Engine for executing queries against the index.
pub struct QueryEngine {
    extractor: Arc<dyn ConceptExtractor + Send + Sync>,
    concept_fps: HashMap<ConceptId, BloomFingerprint>,
    doc_index: DocIndex,
    bloom_bits: usize,
    bloom_hashes: usize,
    reranker: Option<Arc<reranker::Reranker>>,
}

impl std::fmt::Debug for QueryEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QueryEngine")
            .field("extractor", &"<ConceptExtractor>")
            .field("concept_fps_count", &self.concept_fps.len())
            .field("doc_index_size", &self.doc_index.doc_ids.len())
            .field("bloom_bits", &self.bloom_bits)
            .field("bloom_hashes", &self.bloom_hashes)
            .finish()
    }
}

impl QueryEngine {
    /// Create a new query engine.
    pub fn new(
        extractor: Arc<dyn ConceptExtractor + Send + Sync>,
        concept_fps: &HashMap<ConceptId, BloomFingerprint>,
        doc_index: &DocIndex,
        bloom_bits: usize,
        bloom_hashes: usize,
        reranker: Option<Arc<reranker::Reranker>>,
    ) -> Self {
        Self {
            extractor,
            concept_fps: concept_fps.clone(),
            doc_index: doc_index.clone(),
            bloom_bits,
            bloom_hashes,
            reranker,
        }
    }

    /// Build a fingerprint for a query string by merging concept fingerprints.
    pub fn build_query_fp(&self, query: &str) -> BloomFingerprint {
        let concepts = self.extractor.extract_concepts(query);
        let mut fp = BloomFingerprint::new(self.bloom_bits, self.bloom_hashes);

        for c in concepts {
            if let Some(cfp) = self.concept_fps.get(&c.concept_id) {
                // Only merge if sizes match (symmetric case)
                if cfp.len() == self.bloom_bits {
                    fp.merge(cfp);
                } else {
                    // Asymmetric case: fallback to inserting concept directly
                    // This loses the pre-computed expansion but ensures validity
                    fp.insert_concept(c.concept_id);
                }
            } else {
                // If concept not in graph, insert it directly (fallback)
                fp.insert_concept(c.concept_id);
            }
        }

        fp
    }

    /// Search for documents matching the query.
    pub fn search(&self, query: &str, k: usize) -> Vec<(String, u32)> {
        // 1. Extract query concepts
        let concepts = self.extractor.extract_concepts(query);

        // 2. Build query fingerprint by merging concept fingerprints
        let mut query_fp = BloomFingerprint::new(self.bloom_bits, self.bloom_hashes);
        for c in &concepts {
            if let Some(cfp) = self.concept_fps.get(&c.concept_id) {
                // Only merge if sizes match (symmetric case)
                if cfp.len() == self.bloom_bits {
                    query_fp.merge(cfp);
                } else {
                    // Asymmetric case: fallback to inserting concept directly
                    query_fp.insert_concept(c.concept_id);
                }
            } else {
                // Fallback: insert concept ID directly if not in graph
                query_fp.insert_concept(c.concept_id);
            }
        }

        // 3. Score all documents by overlap with query fingerprint
        let mut scored: Vec<_> = self
            .doc_index
            .fingerprints
            .iter()
            .enumerate()
            .map(|(idx, doc_fp)| {
                // Ensure sizes match before comparing
                if query_fp.len() == doc_fp.len() {
                    let score = query_fp.and_count(doc_fp);
                    (idx, score)
                } else {
                    (idx, 0) // Skip mismatch
                }
            })
            .filter(|(_, score)| *score > 0)
            .collect();

        // 4. Sort by score descending
        scored.sort_by(|a, b| b.1.cmp(&a.1));

        // 5. Return top-k results
        scored
            .into_iter()
            .take(k)
            .map(|(idx, score)| (self.doc_index.doc_ids[idx].clone(), score))
            .collect()
    }

    /// Search with optional re-ranking.
    pub fn search_with_rerank(
        &self,
        query: &str,
        top_k: usize,
        rerank: bool,
        doc_fetcher: &dyn Fn(&str) -> Option<String>,
    ) -> anyhow::Result<Vec<(String, f32)>> {
        // 1. Initial retrieval (SBNG)
        // Retrieve more candidates for re-ranking
        // Optimized: Use 2x top_k with minimum of 20 (down from 5x / 50)
        // This gives 2.6x speedup with minimal quality loss
        let initial_k = if rerank { std::cmp::max(top_k * 2, 20) } else { top_k };
        let initial_results = self.search(query, initial_k);

        if !rerank || self.reranker.is_none() {
            // Just return SBNG results normalized
            return Ok(initial_results.into_iter().map(|(id, score)| (id, score as f32)).collect());
        }

        // 2. Fetch document text
        let mut candidates = Vec::new();
        for (doc_id, _) in initial_results {
            if let Some(text) = doc_fetcher(&doc_id) {
                candidates.push((doc_id, text));
            }
        }

        // 3. Re-rank
        if let Some(reranker) = &self.reranker {
            let reranked = reranker.rerank(query, candidates)?;
            // Take top-k
            Ok(reranked.into_iter().take(top_k).collect())
        } else {
            Ok(vec![]) // Should not happen due to check above
        }
    }
}
