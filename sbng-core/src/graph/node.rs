//! Concept node definition.

use serde::{Deserialize, Serialize};

use crate::types::{ConceptId, NodeType};

/// A node in the semantic graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptNode {
    /// Unique concept identifier.
    pub id: ConceptId,
    /// Canonical name of the concept.
    pub canonical_name: String,
    /// Type of the node (Entity, Concept, Attribute).
    pub node_type: NodeType,
    /// Raw corpus frequency.
    pub frequency: u32,
    /// Whether this node is considered a "hub" and should be excluded from
    /// Bloom neighborhood expansions.
    pub is_hub: bool,
}

impl ConceptNode {
    /// Create a new concept node.
    pub fn new(id: ConceptId, canonical_name: String, node_type: NodeType) -> Self {
        Self {
            id,
            canonical_name,
            node_type,
            frequency: 0,
            is_hub: false,
        }
    }
}
