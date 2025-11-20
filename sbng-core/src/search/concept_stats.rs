//! Precomputed concept statistics for efficient document encoding.

use std::collections::HashMap;
use crate::{
    graph::ConceptGraph,
    types::ConceptId,
};

/// Precomputed per-concept statistics for TF-IDF scoring.
#[derive(Debug)]
pub struct ConceptStats {
    /// Inverse document frequency per concept.
    pub idf: HashMap<ConceptId, f32>,
    /// Degree penalty per concept (1 / log(degree + 2)).
    pub degree_penalty: HashMap<ConceptId, f32>,
}

/// Compute concept statistics once for all concepts.
///
/// This avoids repeated ln() calls and graph lookups during doc fingerprint building.
pub fn compute_concept_stats(
    total_docs: usize,
    doc_freqs: &HashMap<ConceptId, u32>,
    graph: &ConceptGraph,
) -> ConceptStats {
    let log_n = (total_docs as f32).ln();
    let mut idf = HashMap::with_capacity(doc_freqs.len());
    let mut degree_penalty = HashMap::with_capacity(doc_freqs.len());

    for (&cid, &df) in doc_freqs {
        let df_f = df.max(1) as f32;
        let idf_val = log_n - df_f.ln();
        idf.insert(cid, idf_val);

        let deg = graph.degree(cid).unwrap_or(0);
        let deg_pen = 1.0 / ((deg as f32 + 2.0).ln());
        degree_penalty.insert(cid, deg_pen);
    }

    ConceptStats { idf, degree_penalty }
}
