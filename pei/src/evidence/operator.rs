use crate::{ItemId, Query};
use crate::storage::item_store::ItemStore;

pub struct EvidenceResult {
    pub belief_delta: f64,
    pub uncertainty_delta: f64,
}

pub trait EvidenceOperator: Send + Sync {
    fn id(&self) -> usize;
    fn cost(&self) -> f64;

    fn apply(
        &self,
        query: &Query,
        item_id: ItemId,
        store: &ItemStore,
    ) -> EvidenceResult;
}
