use lasso::{Spur, ThreadedRodeo};
use serde::{Deserialize, Serialize};
use std::hash::BuildHasher;
use twox_hash::XxHash64;

/// A serializable BuildHasher that uses XxHash64.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SerializableHasher;

impl BuildHasher for SerializableHasher {
    type Hasher = XxHash64;
    fn build_hasher(&self) -> Self::Hasher {
        XxHash64::default()
    }
}

/// Global semantic string interner.
pub type ConceptInterner = ThreadedRodeo<Spur, SerializableHasher>;
/// Key type used by the interner.
pub type ConceptKey = Spur;
