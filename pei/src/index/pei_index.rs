use crate::evidence::operator::EvidenceOperator;
use crate::evidence::dag::EvidenceDAG;
use crate::storage::item_store::ItemStore;

pub struct PEIIndex {
    pub evidence_ops: Vec<Box<dyn EvidenceOperator>>,
    pub dag: EvidenceDAG,
    pub store: ItemStore,
}

impl PEIIndex {
    pub fn new(
        evidence_ops: Vec<Box<dyn EvidenceOperator>>, 
        dag: EvidenceDAG, 
        store: ItemStore
    ) -> Self {
        Self { evidence_ops, dag, store }
    }
}
