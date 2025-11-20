//! Global configuration for SBNG graph building and fingerprint generation.

use serde::{Deserialize, Serialize};

/// Configuration for PMI-based graph building and Bloom fingerprints.
/// Configuration for PMI-based graph building and Bloom fingerprints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SbngConfig {
    /// Minimum co-occurrence count to consider an edge.
    pub cooccur_min: u32,
    /// Minimum PMI value to keep an edge.
    pub pmi_min: f32,
    /// Maximum node degree before marking hub.
    pub max_degree: u32,
    /// Minimum node degree to keep (else consider noise).
    pub min_degree: u32,
    /// Sliding window size for stats accumulation.
    pub window_size: usize,
    /// Target Bloom fingerprint bit length.
    pub bloom_bits: usize,
    /// Number of hash functions (k) for Bloom.
    pub bloom_hashes: usize,
    /// Maximum neighborhood size for fingerprints.
    pub max_neighborhood: usize,
}

impl Default for SbngConfig {
    fn default() -> Self {
        Self {
            cooccur_min: 5,
            pmi_min: 2.0,
            max_degree: 100,
            min_degree: 2,
            window_size: 8,
            bloom_bits: 2048,
            bloom_hashes: 5,
            max_neighborhood: 300,
        }
    }
}
