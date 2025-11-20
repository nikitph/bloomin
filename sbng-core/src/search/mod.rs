//! Document retrieval using Bloom fingerprints.

use std::sync::Arc;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};

use crate::bloom::BloomFingerprint;
use crate::corpus::ConceptExtractor;
use crate::types::ConceptId;

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
    ) -> Self {
        Self {
            extractor,
            concept_fps: concept_fps.clone(),
            doc_index: doc_index.clone(),
            bloom_bits,
            bloom_hashes,
        }
    }

    /// Build a fingerprint for a query string by merging concept fingerprints.
    pub fn build_query_fp(&self, query: &str) -> BloomFingerprint {
        let concepts = self.extractor.extract_concepts(query);
        let mut fp = BloomFingerprint::new(self.bloom_bits, self.bloom_hashes);

        for c in concepts {
            if let Some(cfp) = self.concept_fps.get(&c.concept_id) {
                fp.merge(cfp);
            } else {
                // If concept not in graph, insert it directly (fallback)
                fp.insert_concept(c.concept_id);
            }
        }

        fp
    }

    /// Search for documents matching the query.
    pub fn search(&self, query: &str, k: usize) -> Vec<(String, u32)> {
        let qfp = self.build_query_fp(query);

        let mut scored: Vec<(String, u32)> = self
            .doc_index
            .doc_ids
            .iter()
            .zip(self.doc_index.fingerprints.iter())
            .map(|(doc_id, dfp)| {
                let score = qfp.and_count(dfp);
                (doc_id.clone(), score)
            })
            .collect();

        // Sort descending by score
        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored.into_iter().take(k).collect()
    }
}
