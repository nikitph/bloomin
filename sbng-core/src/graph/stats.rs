use std::collections::HashMap;

use crate::types::ConceptId;

/// Aggregated statistics for PMI.
#[derive(Debug, Default)]
pub struct StatsAccumulator {
    /// Total number of windows processed.
    pub total_windows: u64,
    /// Frequency count for each concept.
    pub concept_freq: HashMap<ConceptId, u64>,
    /// Co-occurrence count for each pair of concepts.
    pub pair_freq: HashMap<(ConceptId, ConceptId), u64>,
}

impl StatsAccumulator {
    /// Create a new empty accumulator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a sliding window (or sentence) of concept IDs.
    pub fn add_window(&mut self, concepts: &[ConceptId]) {
        use std::cmp::{min, max};

        if concepts.is_empty() {
            return;
        }

        self.total_windows += 1;

        let mut uniq = concepts.to_vec();
        uniq.sort_unstable();
        uniq.dedup();

        for &c in &uniq {
            *self.concept_freq.entry(c).or_insert(0) += 1;
        }

        for i in 0..uniq.len() {
            for j in (i + 1)..uniq.len() {
                let a = uniq[i];
                let b = uniq[j];
                let key = (min(a, b), max(a, b));
                *self.pair_freq.entry(key).or_insert(0) += 1;
            }
        }
    }

    /// Merge another accumulator into this one.
    pub fn merge(&mut self, other: StatsAccumulator) {
        self.total_windows += other.total_windows;

        for (k, v) in other.concept_freq {
            *self.concept_freq.entry(k).or_insert(0) += v;
        }

        for (k, v) in other.pair_freq {
            *self.pair_freq.entry(k).or_insert(0) += v;
        }
    }
}
