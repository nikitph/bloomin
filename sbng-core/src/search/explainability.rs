//! Explainability for SBNG search results.
//!
//! Provides concept-level explanations showing which query concepts
//! matched which document concepts, enabling transparent semantic search.

use std::collections::HashMap;
use serde::{Serialize, Deserialize};

use crate::{
    types::ConceptId,
    bloom::BloomFingerprint,
    corpus::interner::ConceptInterner,
};

/// A matched concept showing semantic connection between query and document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchedConcept {
    /// Concept from the query.
    pub query_concept: String,
    /// Related concept(s) in the document.
    pub doc_concepts: Vec<String>,
    /// Overlap score (0.0-1.0).
    pub overlap_score: f32,
}

/// Explain why a document matched a query by identifying overlapping concepts.
///
/// # Arguments
/// * `query_concepts` - Concept IDs extracted from the query
/// * `doc_fp` - Document's Bloom fingerprint
/// * `concept_fps` - Map of concept ID to concept fingerprint
/// * `interner` - Concept interner for resolving concept names
/// * `top_k` - Number of top matched concepts to return
///
/// # Returns
/// Vector of matched concepts, sorted by overlap score (descending)
pub fn explain_match(
    query_concepts: &[ConceptId],
    doc_fp: &BloomFingerprint,
    concept_fps: &HashMap<ConceptId, BloomFingerprint>,
    interner: &ConceptInterner,
    top_k: usize,
) -> Vec<MatchedConcept> {
    let mut matches = Vec::new();

    for &qc in query_concepts {
        if let Some(qfp) = concept_fps.get(&qc) {
            // Calculate overlap between query concept fingerprint and doc fingerprint
            let overlap_bits = qfp.and_count(doc_fp);
            let qfp_bits = qfp.count_set_bits();
            
            if qfp_bits == 0 {
                continue;
            }

            let overlap_score = overlap_bits as f32 / qfp_bits as f32;

            // Only include if there's meaningful overlap
            if overlap_score > 0.1 {
                let query_concept_name = interner.resolve(&qc.0).to_string();
                
                // For now, we show the query concept matched the doc
                // In future, could extract specific doc concepts that overlapped
                matches.push(MatchedConcept {
                    query_concept: query_concept_name,
                    doc_concepts: vec!["document".to_string()],  // Placeholder
                    overlap_score,
                });
            }
        }
    }

    // Sort by overlap score (descending)
    matches.sort_by(|a, b| {
        b.overlap_score
            .partial_cmp(&a.overlap_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Return top-K matches
    matches.into_iter().take(top_k).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bloom::BloomFingerprint;

    #[test]
    fn test_explain_match_basic() {
        // This is a placeholder test - would need proper setup
        let query_concepts = vec![];
        let doc_fp = BloomFingerprint::new(2048, 5);
        let concept_fps = HashMap::new();
        let interner = ConceptInterner::default();

        let matches = explain_match(&query_concepts, &doc_fp, &concept_fps, &interner, 5);
        assert_eq!(matches.len(), 0);
    }
}
