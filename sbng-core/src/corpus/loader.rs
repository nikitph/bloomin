use std::fs::File;
use std::io::{BufRead, BufReader};

use serde::Deserialize;

use crate::errors::Result;

/// A single document from the JSONL corpus.
#[derive(Debug, Deserialize)]
pub struct JsonlDoc {
    /// Unique document identifier.
    pub id: String,
    /// Raw text content of the document.
    pub text: String,
}

/// A corpus reader for line-delimited JSON files.
#[derive(Debug)]
pub struct JsonlCorpus {
    path: String,
}

impl JsonlCorpus {
    /// Create a new corpus reader for the given path.
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }

    /// Iterate over documents in the corpus.
    pub fn iter(&self) -> Result<impl Iterator<Item = Result<JsonlDoc>>> {
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);

        Ok(reader.lines().map(|line| {
            let line = line?;
            let doc: JsonlDoc = serde_json::from_str(&line)?;
            Ok(doc)
        }))
    }
}
