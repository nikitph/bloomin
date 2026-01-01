use crate::{ItemId, Query};
use crate::storage::item_store::ItemStore;
use crate::evidence::operator::{EvidenceOperator, EvidenceResult};

pub struct AspectRatioOp {
    pub id: usize,
}

impl EvidenceOperator for AspectRatioOp {
    fn id(&self) -> usize { self.id }
    fn cost(&self) -> f64 { 0.001 } // Extremely cheap O(1)

    fn apply(&self, query: &Query, item_id: ItemId, store: &ItemStore) -> EvidenceResult {
        let meta = &store.metadata[item_id];
        
        // If query has aspect hint
        if let Some(target_aspect) = query.aspect_hint {
            // Check tolerance. e.g. within 0.2
            if (meta.aspect_ratio - target_aspect).abs() > 0.2 {
                // Incompatible! Heavy penalty.
                return EvidenceResult {
                    belief_delta: -100.0, // Pruned
                    uncertainty_delta: 0.1,
                };
            }
        }
        
        // Compatible or no hint
        EvidenceResult {
            belief_delta: 0.0,
            uncertainty_delta: 0.1, 
        }
    }
}

pub struct ColorOp {
    pub id: usize,
}

impl EvidenceOperator for ColorOp {
    fn id(&self) -> usize { self.id }
    fn cost(&self) -> f64 { 0.001 } // Extremely cheap O(1)

    fn apply(&self, query: &Query, item_id: ItemId, store: &ItemStore) -> EvidenceResult {
        let meta = &store.metadata[item_id];
        
        if let Some(target_color) = query.color_hint {
            // Strict match for simplicity, or "distance" in color space
            if meta.color != target_color {
                return EvidenceResult {
                    belief_delta: -100.0, // Pruned
                    uncertainty_delta: 0.1,
                };
            }
        }
        
        EvidenceResult {
            belief_delta: 0.0,
            uncertainty_delta: 0.1,
        }
    }
}

pub struct CoarseClipOp {
    pub id: usize,
}

impl EvidenceOperator for CoarseClipOp {
    fn id(&self) -> usize { self.id }
    fn cost(&self) -> f64 { 0.05 } // Cheap vector op (32 dim)

    fn apply(&self, query: &Query, item_id: ItemId, store: &ItemStore) -> EvidenceResult {
        let meta = &store.metadata[item_id];
        // Assume query.vector has the full embedding.
        // We need a cheap way to compare. 
        // In reality, we'd project query to coarse space.
        // For simulation, let's assume query.vector[0..32] corresponds to coarse.
        
        let dim = meta.coarse_emb.len();
        let query_slice = &query.vector[0..dim]; // Panic if query too short
        
        let dist: f32 = meta.coarse_emb.iter().zip(query_slice.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>().sqrt();
            
        EvidenceResult {
            belief_delta: -(dist as f64),
            uncertainty_delta: 0.4,
        }
    }
}

pub struct FullClipOp {
    pub id: usize,
}

impl EvidenceOperator for FullClipOp {
    fn id(&self) -> usize { self.id }
    fn cost(&self) -> f64 { 1.0 } // Expensive

    fn apply(&self, query: &Query, item_id: ItemId, store: &ItemStore) -> EvidenceResult {
        let d = store.distance(query, item_id);
        EvidenceResult {
            belief_delta: -(d as f64),
            uncertainty_delta: 0.99, // Clears uncertainty
        }
    }
}
