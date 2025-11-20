//! Document fingerprint building with TF-IDF × degree penalty and Top-K selection.

use std::collections::HashMap;
use dashmap::DashMap;
use rayon::prelude::*;

use crate::{
    types::ConceptId,
    bloom::BloomFingerprint,
    config::BloomConfig,
    search::concept_stats::ConceptStats,
};

/// Scored concept for Top-K selection.
#[derive(Debug)]
struct ScoredConcept {
    id: ConceptId,
    score: f32,
}

/// Parallel, optimized doc fingerprint builder using precomputed stats.
///
/// - Per-doc complexity: O(#unique concepts in doc × log(#unique concepts))
/// - Uses precomputed IDF + degree_penalty (ConceptStats)
/// - Top-K per doc (doc_config.max_neighborhood)
/// - Docs are encoded as sparse Bloom filters of ConceptIds
///
/// # Arguments
/// * `docs` - Map of DocID -> Vec<ConceptId> (concepts extracted from each doc)
/// * `doc_config` - Bloom configuration for documents (bits, hashes, max_neighborhood=Top-K)
/// * `stats` - Precomputed IDF and degree penalty per concept
///
/// # Returns
/// Map of DocID -> sparse BloomFingerprint
pub fn build_doc_fingerprints_parallel(
    docs: &HashMap<String, Vec<ConceptId>>,
    doc_config: &BloomConfig,
    stats: &ConceptStats,
) -> HashMap<String, BloomFingerprint> {
    println!(
        "   -> Generating fingerprints for {} docs in parallel...",
        docs.len()
    );

    let doc_fingerprints = DashMap::with_capacity(docs.len());

    docs.par_iter().for_each(|(doc_id, concepts)| {
        // 1. Local TF
        let mut tf_counts: HashMap<ConceptId, u32> = HashMap::new();
        for &cid in concepts {
            *tf_counts.entry(cid).or_insert(0) += 1;
        }

        // 2. Score concepts: TF × IDF × degree_penalty
        let mut scored: Vec<ScoredConcept> = tf_counts
            .into_iter()
            .filter_map(|(cid, tf)| {
                let idf = *stats.idf.get(&cid).unwrap_or(&0.0);
                let deg_penalty = *stats.degree_penalty.get(&cid).unwrap_or(&1.0);
                let score = (tf as f32) * idf * deg_penalty;

                Some(ScoredConcept { id: cid, score })
            })
            .collect();

        // 3. Top-K selection by score (descending)
        scored.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let keep_count = std::cmp::min(scored.len(), doc_config.max_neighborhood);

        // 4. Build sparse doc fingerprint from raw ConceptIds
        let mut fp = BloomFingerprint::new(doc_config.bloom_bits, doc_config.bloom_hashes);
        for item in scored.iter().take(keep_count) {
            fp.insert_concept(item.id);
        }

        doc_fingerprints.insert(doc_id.clone(), fp);
    });

    // Convert DashMap -> HashMap
    doc_fingerprints.into_iter().collect()
}
