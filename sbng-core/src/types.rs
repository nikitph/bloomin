//! Common core types used across the SBNG engine.

use lasso::Spur;
use serde::{Deserialize, Serialize};

/// Internal identifier for a concept (interned string).
/// We use a newtype wrapper to ensure clean serialization if needed, 
/// but for now let's stick to type alias and ensure feature is active.
/// Actually, the error might be because `Spur` doesn't implement `Deserialize` 
/// without the feature, and maybe it didn't pick up?
/// Let's use a newtype wrapper to be safe and explicit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ConceptId(pub Spur);

impl Serialize for ConceptId {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        // Serialize as the logical index (usize)
        use lasso::Key;
        self.0.into_usize().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ConceptId {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Deserialize as usize and convert to Spur
        // Note: This assumes the key is valid or we just wrap it. 
        // Spur::try_from_usize is what we want if available, or unsafe/raw construction.
        // lasso::Key trait has `try_from_usize`.
        use lasso::Key;
        let val = usize::deserialize(deserializer)?;
        // Spur::try_from_usize returns Option.
        Spur::try_from_usize(val)
            .map(ConceptId)
            .ok_or_else(|| serde::de::Error::custom("invalid concept id key"))
    }
}

impl From<Spur> for ConceptId {
    fn from(s: Spur) -> Self {
        Self(s)
    }
}

impl From<ConceptId> for Spur {
    fn from(c: ConceptId) -> Self {
        c.0
    }
}

/// Internal numeric identifier for edges, if needed.
pub type EdgeId = u64;

/// Node type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    /// Named entity (e.g., "tesla", "germany").
    Entity,
    /// Abstract concept (e.g., "battery", "sedan").
    Concept,
    /// Attribute / property (e.g., "range", "height").
    Attribute,
}
