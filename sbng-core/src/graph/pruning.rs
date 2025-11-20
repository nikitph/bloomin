//! Pruning logic: hub suppression and entropy-constrained edge removal.

use petgraph::visit::EdgeRef;
use rayon::prelude::*;

use crate::config::SbngConfig;
use crate::graph::ConceptGraph;

/// Functions related to pruning edges and marking hubs.
#[derive(Debug)]
pub struct GraphPruner;

impl GraphPruner {
    /// Mark nodes whose degree exceeds config.max_degree as hubs.
    ///
    /// Hubs are kept in the graph but will be *excluded from Bloom neighborhoods*.
    pub fn mark_hubs(graph: &mut ConceptGraph, config: &SbngConfig) {
        let inner = &mut graph.inner_mut();
        for node_idx in inner.node_indices() {
            let degree = inner.neighbors(node_idx).count() as u32;
            if let Some(node) = inner.node_weight_mut(node_idx) {
                node.is_hub = degree > config.max_degree;
            }
        }
    }

    /// Apply entropy-constrained pruning:
    ///
    /// For each node C, consider its neighbor edges.
    /// For each edge (C, u), compute Shannon entropy of neighbor weights:
    ///   - H_before: using all neighbors
    ///   - H_after: if this neighbor u is removed
    /// If H_after < H_before - epsilon (entropy decreases), we remove the edge (C, u),
    /// because removing u makes the neighborhood more "focused".
    pub fn entropy_constrained_prune(graph: &mut ConceptGraph) {
        // Phase 1: Read-only parallel analysis
        let mut edges_to_remove: Vec<_> = {
            let inner = graph.inner();
            let node_indices: Vec<_> = inner.node_indices().collect();

            node_indices.into_par_iter()
                .flat_map(|node_idx| {
                    let mut local_removals = Vec::new();
                    let epsilon: f32 = 0.01;

                    // Collect neighbor edges
                    let mut neighbor_edges = Vec::new();
                    for edge_ref in inner.edges(node_idx) {
                        neighbor_edges.push((edge_ref.id(), edge_ref.weight().weight));
                    }

                    if neighbor_edges.len() < 2 {
                        return local_removals;
                    }

                    // Compute entropy with all neighbors
                    let weights: Vec<f32> = neighbor_edges.iter().map(|(_, w)| *w).collect();
                    let h_full = entropy_from_weights(&weights);
                    let sum_full: f32 = weights.iter().sum();

                    for (edge_id, w_remove) in &neighbor_edges {
                        let sum_without = sum_full - w_remove;
                        if sum_without <= 0.0 {
                            continue;
                        }

                        // Recompute probabilities without this edge
                        let mut probs_without = Vec::with_capacity(weights.len() - 1);
                        for (other_edge_id, w_other) in &neighbor_edges {
                            if other_edge_id == edge_id {
                                continue;
                            }
                            probs_without.push(w_other / sum_without);
                        }

                        let h_without = entropy_from_probs(&probs_without);

                        if h_without + epsilon < h_full {
                            local_removals.push(*edge_id);
                        }
                    }
                    local_removals
                })
                .collect()
        };

        // Phase 2: Apply removals
        let inner = graph.inner_mut();
        
        // Deduplicate edges to remove
        edges_to_remove.sort_unstable();
        edges_to_remove.dedup();

        for eid in edges_to_remove {
            let _ = inner.remove_edge(eid);
        }
    }
}

/// Compute Shannon entropy from raw weights (will be normalized inside).
fn entropy_from_weights(weights: &[f32]) -> f32 {
    let sum: f32 = weights.iter().sum();
    if sum <= 0.0 {
        return 0.0;
    }
    let probs: Vec<f32> = weights.iter().map(|w| w / sum).collect();
    entropy_from_probs(&probs)
}

/// Compute Shannon entropy H = -Σ p log2 p from probabilities.
fn entropy_from_probs(probs: &[f32]) -> f32 {
    let mut h = 0.0;
    for &p in probs {
        if p > 0.0 {
            h -= p * p.log2();
        }
    }
    h
}
