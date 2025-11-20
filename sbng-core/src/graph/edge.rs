//! Concept edge definition (PMI-weighted).

use serde::{Deserialize, Serialize};

/// Edge between two concepts with PMI-based weight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptEdge {
    /// PMI-based weight or transformed score.
    pub weight: f32,
    /// Number of triadic paths between the endpoints via a common neighbor.
    pub triadic_paths: u16,
}

impl ConceptEdge {
    /// Create a new edge with the given weight.
    pub fn new(weight: f32) -> Self {
        Self {
            weight,
            triadic_paths: 0,
        }
    }
}
