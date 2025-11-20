//! Pipeline to generate semantic fingerprints from a ConceptGraph.

use std::collections::{HashMap, HashSet};
use rayon::prelude::*;

use petgraph::graph::{Graph, NodeIndex};
use crate::{
    bloom::BloomFingerprint,
    config::SbngConfig,
    errors::Result,
    graph::{ConceptEdge, ConceptGraph, ConceptNode},
    types::ConceptId,
};

/// Pipeline to generate semantic fingerprints from a ConceptGraph.
#[derive(Debug)]
pub struct SignatureGenerationPipeline {
    config: SbngConfig,
}

impl SignatureGenerationPipeline {
    /// Create a new pipeline with the given configuration.
    pub fn new(config: SbngConfig) -> Self {
        Self { config }
    }

    /// Generate per-concept fingerprints.
    ///
    /// Returns (ConceptId, BloomFingerprint) pairs.
    pub fn generate(&self, graph: &ConceptGraph) -> Result<Vec<(ConceptId, BloomFingerprint)>> {
        let inner = graph.inner();
        let node_indices: Vec<_> = inner.node_indices().collect();

        // Parallel fingerprint generation
        let results: Vec<_> = node_indices
            .into_par_iter()
            .map(|node_idx| {
                let node = inner.node_weight(node_idx).expect("node exists");
                let cid = node.id;

                // Build neighborhood for this concept.
                let neighborhood = build_neighborhood(inner, node_idx, &self.config);

                // Create Bloom fingerprint.
                let mut fp = BloomFingerprint::new(self.config.bloom_bits, self.config.bloom_hashes);

                // Always insert the concept itself.
                fp.insert_concept(cid);

                // Insert neighborhood.
                for nb_cid in neighborhood {
                    fp.insert_concept(nb_cid);
                }

                (cid, fp)
            })
            .collect();

        Ok(results)
    }
}

/// Build N(C) for a node: radius-1 + radius-2 with triadic closure and scoring.
fn build_neighborhood(
    graph: &Graph<ConceptNode, ConceptEdge>,
    center: NodeIndex,
    config: &SbngConfig,
) -> Vec<ConceptId> {
    use petgraph::visit::EdgeRef;

    let center_node = graph.node_weight(center).expect("center node");
    let center_id = center_node.id;

    // Collect 1-hop neighbors: N1
    let mut n1 = Vec::new(); // (neighbor_idx, weight)
    let mut neighbor_set = HashSet::new(); // NodeIndex set for quick lookup

    for edge_ref in graph.edges(center) {
        let w = edge_ref.weight().weight;
        if w < config.pmi_min {
            continue;
        }

        let nb_idx = if edge_ref.source() == center {
            edge_ref.target()
        } else {
            edge_ref.source()
        };

        let nb_node = graph.node_weight(nb_idx).expect("neighbor node");

        // Exclude hubs as neighbors.
        if nb_node.is_hub {
            continue;
        }

        // Additional degree-based filter:
        let degree = graph.neighbors(nb_idx).count() as u32;
        if degree > config.max_degree {
            continue;
        }

        n1.push((nb_idx, w));
        neighbor_set.insert(nb_idx);
    }

    // Map for candidate scores / meta:
    #[derive(Debug)]
    struct CandidateInfo {
        weight_sum: f32,
        triadic_paths: u16,
        degree: u32,
        id: ConceptId,
    }

    let mut candidates: HashMap<NodeIndex, CandidateInfo> = HashMap::new();

    // Seed candidates from N1 (radius-1 neighbors).
    for (nb_idx, w) in &n1 {
        let nb_node = graph.node_weight(*nb_idx).unwrap();
        let degree = graph.neighbors(*nb_idx).count() as u32;

        candidates.insert(
            *nb_idx,
            CandidateInfo {
                weight_sum: *w,
                triadic_paths: 1,
                degree,
                id: nb_node.id,
            },
        );
    }

    // Radius-2 candidates via triadic closure:
    //
    // For each neighbor n1 of C, look at its neighbors u.
    // If u != C, not a hub, and is *not already* a direct neighbor,
    // count number of distinct paths C - n1 - u (triadic_paths).
    //
    // We only keep u if triadic_paths >= 2.
    for (nb_idx, _) in &n1 {
        for edge_ref in graph.edges(*nb_idx) {
            let u_idx = if edge_ref.source() == *nb_idx {
                edge_ref.target()
            } else {
                edge_ref.source()
            };

            if u_idx == center {
                continue;
            }
            if neighbor_set.contains(&u_idx) {
                // Already 1-hop neighbor, handled above.
                continue;
            }

            let u_node = match graph.node_weight(u_idx) {
                Some(n) => n,
                None => continue,
            };

            if u_node.is_hub {
                continue;
            }

            let degree_u = graph.neighbors(u_idx).count() as u32;
            if degree_u > config.max_degree {
                continue;
            }

            let w = edge_ref.weight().weight;
            if w < config.pmi_min {
                continue;
            }

            let entry = candidates.entry(u_idx).or_insert_with(|| CandidateInfo {
                weight_sum: 0.0,
                triadic_paths: 0,
                degree: degree_u,
                id: u_node.id,
            });

            entry.weight_sum += w;
            entry.triadic_paths = entry.triadic_paths.saturating_add(1);
        }
    }

    // Now filter radius-2 candidates by triadic_paths >= 2
    candidates.retain(|idx, info| {
        if neighbor_set.contains(idx) {
            true // N1 always kept
        } else {
            info.triadic_paths >= 2
        }
    });

    // Compute scores.
    let mut scored: Vec<(f32, ConceptId)> = candidates
        .into_iter()
        .map(|(_idx, info)| {
            let degree_penalty = 1.0 / (1.0 + (info.degree as f32).ln());
            let triadic_factor = info.triadic_paths.max(1) as f32;
            let score = info.weight_sum * triadic_factor * degree_penalty;
            (score, info.id)
        })
        .collect();

    // Sort by descending score
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Truncate to max_neighborhood
    scored
        .into_iter()
        .take(config.max_neighborhood)
        .map(|(_, cid)| cid)
        // Avoid including center_id itself; Bloom will insert the center separately.
        .filter(|cid| *cid != center_id)
        .collect()
}
