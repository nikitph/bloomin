//! Concept extraction and canonicalization interface.

use crate::types::ConceptId;

/// A concept extracted from text, before/after canonicalization.
#[derive(Debug, Clone)]
pub struct ExtractedConcept {
    /// Canonical name of the concept.
    pub canonical_name: String,
    /// Unique concept identifier.
    pub concept_id: ConceptId,
}

/// Trait responsible for mapping raw text to concept IDs.
///
/// This is where you plug your LLM-based synonym folding and canonicalization.
pub trait ConceptExtractor: Send + Sync {
    /// Extract canonical concepts from a document.
    fn extract_concepts(&self, text: &str) -> Vec<ExtractedConcept>;
}
