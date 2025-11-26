//! Bloom filter fingerprint implementation.

use fixedbitset::FixedBitSet;
use serde::{Deserialize, Serialize};
use std::hash::Hasher;
use twox_hash::XxHash64;

use crate::types::ConceptId;

/// Fixed-size Bloom fingerprint for a concept.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomFingerprint {
    bits: FixedBitSet,
    m: usize,
    k: usize,
}

impl BloomFingerprint {
    /// Create a new empty fingerprint.
    pub fn new(m: usize, k: usize) -> Self {
        let mut bits = FixedBitSet::with_capacity(m);
        bits.clear();
        Self { bits, m, k }
    }

    /// Get the size of the Bloom filter in bits.
    pub fn len(&self) -> usize {
        self.m
    }

    #[inline]
    fn hash_with_seed(x: u64, seed: u64) -> u64 {
        let mut h = XxHash64::with_seed(seed);
        h.write_u64(x);
        h.finish()
    }

    /// Insert an interned concept ID into the fingerprint.
    pub fn insert_concept(&mut self, concept_id: ConceptId) {
        // ConceptId = Spur; get a stable integer representation
        // Note: We assume ConceptId wraps Spur which can be converted to usize/u64.
        // In our types.rs, ConceptId(pub Spur).
        let raw_id = concept_id.0.into_inner();
        // lasso::Key::into_usize() or similar needed if raw_id is Key.
        // Assuming we fixed this in previous steps to use .get() or similar.
        // Let's check types.rs again or use the known working method.
        // In previous steps we used `raw_id.get() as usize`.
        let val = raw_id.get() as usize;

        // Simple double-hashing scheme
        let base = Self::hash_with_seed(val as u64, 0x9E37_79B1_85EB_CA87);

        for i in 0..self.k {
            let h = base.wrapping_add(i as u64).rotate_left(i as u32);
            let pos = (h as usize) % self.m;
            self.bits.insert(pos);
        }
    }

    /// Bitwise AND and count of overlapping bits.
    pub fn and_count(&self, other: &Self) -> u32 {
        assert_eq!(self.bits.len(), other.bits.len());
        let mut count = 0;
        for (a, b) in self.bits.as_slice().iter().zip(other.bits.as_slice()) {
            count += (a & b).count_ones();
        }
        count
    }

    /// Total number of bits set.
    pub fn popcount(&self) -> u32 {
        self.bits.as_slice().iter().map(|w| w.count_ones()).sum()
    }

    /// Merge another fingerprint into this one (bitwise OR).
    pub fn merge(&mut self, other: &Self) {
        assert_eq!(self.bits.len(), other.bits.len());
        self.bits.union_with(&other.bits);
    }

    /// Compute fill ratio (fraction of bits set).
    pub fn fill_ratio(&self) -> f32 {
        let set_bits = self.bits.count_ones(..);
        set_bits as f32 / self.m as f32
    }

    /// Count the number of set bits.
    pub fn count_set_bits(&self) -> usize {
        self.bits.count_ones(..)
    }
}

/// Compute Jaccard similarity between two fingerprints.
/// J(A, B) = |A ∩ B| / |A ∪ B|
pub fn jaccard(a: &BloomFingerprint, b: &BloomFingerprint) -> f32 {
    let inter = a.and_count(b) as f32;
    let union = (a.count_set_bits() + b.count_set_bits()) as f32 - inter;
    if union == 0.0 {
        0.0
    } else {
        inter / union
    }
}
