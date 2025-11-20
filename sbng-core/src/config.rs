//! Global configuration for SBNG graph building and fingerprint generation.

use serde::{Deserialize, Serialize};

///! Configuration for SBNG.

/// Bloom filter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BloomConfig {
    /// Number of bits in the Bloom filter.
    pub bloom_bits: usize,
    /// Number of hash functions.
    pub bloom_hashes: usize,
    /// Maximum neighborhood size (for concepts) or Top-K (for docs).
    pub max_neighborhood: usize,
}

/// Main configuration for SBNG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SbngConfig {
    /// Minimum co-occurrence count for edge creation.
    pub cooccur_min: u32,
    /// Minimum PMI threshold for edge retention.
    pub pmi_min: f32,
    /// Maximum degree before marking as hub.
    pub max_degree: u32,
    /// Minimum degree for a node to be kept.
    pub min_degree: u32,
    /// Sliding window size for co-occurrence.
    pub window_size: usize,

    /// Bloom configuration for concept fingerprints (dense, semantic).
    pub concept_bloom: BloomConfig,
    /// Bloom configuration for document fingerprints (sparse, precise).
    pub doc_bloom: BloomConfig,
}

impl Default for SbngConfig {
    fn default() -> Self {
        Self {
            cooccur_min: 8,
            pmi_min: 1.5,
            max_degree: 80,
            min_degree: 2,
            window_size: 8,

            // Concept fingerprints: dense, semantic expansion
            concept_bloom: BloomConfig {
                bloom_bits: 2048,
                bloom_hashes: 5,
                max_neighborhood: 150,
            },

            // Document fingerprints: sparse, Top-K selection
            doc_bloom: BloomConfig {
                bloom_bits: 4096,
                bloom_hashes: 5,
                max_neighborhood: 64,
            },
        }
    }
}
