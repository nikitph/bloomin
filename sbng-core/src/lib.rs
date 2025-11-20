#![forbid(unsafe_code)]
#![deny(
    warnings,
    missing_debug_implementations,
    missing_docs,
    rust_2018_idioms
)]

//! # sbng-core
//!
//! Core library for the Sparse Binary Neural Graph (SBNG) engine:
//! - PMI-based semantic graph construction
//! - Hub suppression and entropy-constrained pruning
//! - Bloom-coded semantic fingerprints (2048-bit)
//!
//! This crate is designed to be deterministic, testable, and embedding-free.


pub mod config;
pub mod corpus;
pub mod errors;
pub mod graph;
/// Bloom filter implementation.
pub mod bloom;
/// High-level pipelines.
pub mod pipeline;
/// Search and retrieval.
pub mod search;
/// Persistence layer.
pub mod persistence;
/// HTTP API server.
pub mod server;
pub mod types;

pub use bloom::BloomFingerprint;
pub use config::SbngConfig;
pub use errors::SbngError;
pub use graph::ConceptGraph;
pub use pipeline::{GraphBuildPipeline, SignatureGenerationPipeline};
pub use types::{ConceptId, NodeType};
