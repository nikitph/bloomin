use crate::{ItemId, Query};
use crate::storage::item_store::ItemStore;
use crate::evidence::operator::{EvidenceOperator, EvidenceResult};

pub struct FaissDistanceOp {
    pub id: usize,
}

impl EvidenceOperator for FaissDistanceOp {
    fn id(&self) -> usize { self.id }
    fn cost(&self) -> f64 { 1.0 }

    fn apply(&self, query: &Query, item_id: ItemId, store: &ItemStore) -> EvidenceResult {
        let d = store.distance(query, item_id);
        EvidenceResult {
            belief_delta: -(d as f64),
            uncertainty_delta: 0.99,
        }
    }
}

pub struct MockCoarseOp {
    pub id: usize,
}

impl EvidenceOperator for MockCoarseOp {
    fn id(&self) -> usize { self.id }
    fn cost(&self) -> f64 { 0.1 }

    fn apply(&self, query: &Query, item_id: ItemId, store: &ItemStore) -> EvidenceResult {
        let true_dist = store.distance(query, item_id);
        let estimated_penalty = if true_dist < 1.0 { 
            0.0 
        } else {
            -5.0 
        };

        EvidenceResult {
            belief_delta: estimated_penalty,
            uncertainty_delta: 0.4,
        }
    }
}

pub struct MockDimensionOp {
    pub id: usize,
    pub dim_index: usize,
}

impl EvidenceOperator for MockDimensionOp {
    fn id(&self) -> usize { self.id }
    fn cost(&self) -> f64 { 0.05 } // Very cheap

    fn apply(&self, query: &Query, item_id: ItemId, store: &ItemStore) -> EvidenceResult {
        let v = store.get(item_id);
        let diff = (v[self.dim_index] - query.vector[self.dim_index]).abs();
        
        // If diff is large in this dimension, likely far away.
        let penalty = if diff > 0.5 { -2.0 } else { 0.0 };
        
        EvidenceResult {
            belief_delta: penalty as f64,
            uncertainty_delta: 0.1, // Small reduction
        }
    }
}
