//! Persistence layer for SBNG index.
//! Saves/loads: Metadata, Interner, Concept Fingerprints, DocIndex.

use std::collections::HashMap;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::bloom::BloomFingerprint;
use crate::config::SbngConfig;
use crate::search::DocIndex;
use crate::types::ConceptId;

/// Metadata stored with the index.
#[derive(Debug, Serialize, Deserialize)]
pub struct IndexMetadata {
    /// Version of the SBNG core library.
    pub version: String,
    /// ISO 8601 timestamp of creation.
    pub created_at: String,
    /// Configuration used to build the index.
    pub config: SbngConfig,
}

impl IndexMetadata {
    /// Create new metadata with current version and timestamp.
    pub fn new(config: &SbngConfig) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            config: config.clone(),
        }
    }
}

use lasso::Key;
use crate::corpus::{ConceptInterner, ConceptKey};
use crate::corpus::interner::SerializableHasher;

/// Save the index to the specified directory.
pub fn save_index(
    index_dir: &Path,
    metadata: &IndexMetadata,
    interner: &ConceptInterner,
    concept_fps: &HashMap<ConceptId, BloomFingerprint>,
    doc_index: &DocIndex,
) -> Result<()> {
    std::fs::create_dir_all(index_dir)?;

    // 1. Metadata (JSON)
    let meta_path = index_dir.join("metadata.json");
    let meta_file = File::create(&meta_path).context("Failed to create metadata file")?;
    serde_json::to_writer_pretty(meta_file, metadata)?;

    // 2. Interner (Bincode - Manual Vec<String>)
    // We serialize as a list of strings in ID order to preserve mapping.
    let interner_path = index_dir.join("interner.bin");
    let interner_file = File::create(&interner_path).context("Failed to create interner file")?;
    let mut interner_writer = BufWriter::new(interner_file);
    
    let len = interner.len();
    let mut strings = Vec::with_capacity(len);
    for i in 0..len {
        let key = ConceptKey::try_from_usize(i).ok_or_else(|| anyhow::anyhow!("Invalid key index"))?;
        strings.push(interner.resolve(&key));
    }
    bincode::serialize_into(&mut interner_writer, &strings)?;

    // 3. Concept Fingerprints (Bincode)
    let concepts_path = index_dir.join("concepts.bin");
    let concepts_file = File::create(&concepts_path).context("Failed to create concepts file")?;
    let mut concepts_writer = BufWriter::new(concepts_file);
    bincode::serialize_into(&mut concepts_writer, concept_fps)?;

    // 4. DocIndex (Bincode)
    let docs_path = index_dir.join("docs.bin");
    let docs_file = File::create(&docs_path).context("Failed to create docs file")?;
    let mut docs_writer = BufWriter::new(docs_file);
    bincode::serialize_into(&mut docs_writer, doc_index)?;

    Ok(())
}

/// Load the index from the specified directory.
pub fn load_index(
    index_dir: &Path,
) -> Result<(
    IndexMetadata,
    Arc<ConceptInterner>,
    HashMap<ConceptId, BloomFingerprint>,
    DocIndex,
)> {
    // 1. Metadata
    let meta_path = index_dir.join("metadata.json");
    let meta_file = File::open(&meta_path).context("Failed to open metadata file")?;
    let metadata: IndexMetadata = serde_json::from_reader(meta_file)?;

    // 2. Interner (Manual Vec<String>)
    let interner_path = index_dir.join("interner.bin");
    let interner_bytes = std::fs::read(&interner_path).context("Failed to read interner file")?;
    
    let strings: Vec<String> = bincode::deserialize(&interner_bytes)?;
    let interner = ConceptInterner::with_hasher(SerializableHasher::default());
    for s in strings {
        interner.get_or_intern(s);
    }

    // 3. Concept Fingerprints
    let concepts_path = index_dir.join("concepts.bin");
    let concepts_bytes = std::fs::read(&concepts_path).context("Failed to read concepts file")?;
    let concept_fps: HashMap<ConceptId, BloomFingerprint> = bincode::deserialize(&concepts_bytes)?;

    // 4. DocIndex
    let docs_path = index_dir.join("docs.bin");
    let docs_bytes = std::fs::read(&docs_path).context("Failed to read docs file")?;
    let doc_index: DocIndex = bincode::deserialize(&docs_bytes)?;

    Ok((metadata, Arc::new(interner), concept_fps, doc_index))
}
