//! High-level pipelines: graph building and signature generation.

pub mod build_graph;
pub mod generate_signatures;

pub use build_graph::GraphBuildPipeline;
pub use generate_signatures::SignatureGenerationPipeline;
