use std::cmp::Ordering;
use bit_set::BitSet;
use crate::ItemId;

#[derive(Clone, Debug)]
pub struct Candidate {
    pub item_id: ItemId,
    pub belief: f64,
    pub uncertainty: f64,
    pub applied: BitSet,
}

impl Candidate {
    pub fn new(item_id: ItemId, belief: f64, uncertainty: f64) -> Self {
        Self {
            item_id,
            belief,
            uncertainty,
            applied: BitSet::new(),
        }
    }

    // Score for priority queue: belief - uncertainty
    // Higher is better.
    pub fn score(&self) -> f64 {
        self.belief - self.uncertainty
    }
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.item_id == other.item_id
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // We want a Max-Heap based on score.
        // float comparison needs unwrap
        self.score().partial_cmp(&other.score()).unwrap_or(Ordering::Equal)
    }
}
