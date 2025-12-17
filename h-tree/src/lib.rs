//! # Holographic B-Tree (H-Tree)
//!
//! A novel data structure for high-dimensional vector search that combines:
//! - Witness Field Theory (thermodynamic semantics)
//! - Spectral Bloom Filters (hierarchical witness encoding)
//! - B-Tree structure (disk-friendly, ACID-safe)
//! - Merkle tree integrity (cryptographic proofs)
//!
//! ## Key Properties
//! - O(log N) query time
//! - O(1) vacuum detection (instant rejection of impossible queries)
//! - O(log N) insert with lazy propagation
//! - Disk-friendly (page-aligned nodes)
//! - ACID-safe transactions
//!
//! ## Theoretical Foundation
//! The H-Tree fills a previously empty cell in the Witness Field Theory
//! periodic table of computational primitives:
//! - Row: Atemporal
//! - Column: Conservative
//! - Properties: AC⊕I∞W (Atemporal-Conservative-Commutative-Idempotent-Manifold-Weak)

pub mod spectral;
pub mod node;
pub mod tree;
pub mod merkle;
pub mod vector;

pub use spectral::SpectralBloomFilter;
pub use node::{HNode, NodeId};
pub use tree::HTree;
pub use vector::Vector;
