//! Error types for sbng-core.

use thiserror::Error;

/// Top-level error type for SBNG operations.
#[derive(Debug, Error)]
pub enum SbngError {
    /// Configuration-related errors.
    #[error("configuration error: {0}")]
    Config(String),

    /// Corpus ingestion or tokenization errors.
    #[error("corpus error: {0}")]
    Corpus(String),

    /// Graph construction errors.
    #[error("graph error: {0}")]
    Graph(String),

    /// Bloom fingerprint / encoding errors.
    #[error("bloom error: {0}")]
    Bloom(String),

    /// I/O error wrapper.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Serde serialization/deserialization error.
    #[error("serde error: {0}")]
    Serde(#[from] serde_json::Error),
}

/// Result type for SBNG operations.
pub type Result<T> = std::result::Result<T, SbngError>;
