//! Graph construction logic: from co-occurrence / PMI into a ConceptGraph.

use std::collections::HashMap;
use std::sync::Arc;


use petgraph::graph::Graph;
use petgraph::prelude::NodeIndex;

use crate::{
    config::SbngConfig,
    corpus::ConceptInterner,
    errors::{Result, SbngError},
    graph::{ConceptEdge, ConceptNode, ConceptGraph, StatsAccumulator},
    types::{ConceptId, NodeType},
};

/// Builder for ConceptGraph from corpus statistics.
#[derive(Debug)]
pub struct ConceptGraphBuilder {
    config: SbngConfig,
    graph: Graph<ConceptNode, ConceptEdge>,
    stats: Option<StatsAccumulator>,
    id_to_node: HashMap<ConceptId, NodeIndex>,
    interner: Arc<ConceptInterner>,
}

impl ConceptGraphBuilder {
    /// Create a new builder with the given configuration and interner.
    pub fn new(config: SbngConfig, interner: Arc<ConceptInterner>) -> Self {
        Self {
            config,
            graph: Graph::new(),
            stats: None,
            id_to_node: HashMap::new(),
            interner,
        }
    }

    /// Ingest pre-calculated statistics.
    pub fn ingest_stats(&mut self, stats: StatsAccumulator) {
        self.stats = Some(stats);
    }

    /// Finalize the graph construction.
    pub fn finalize(mut self) -> Result<ConceptGraph> {
        let stats = self
            .stats
            .take()
            .ok_or_else(|| SbngError::Graph("no stats ingested".into()))?;

        let total_windows = stats.total_windows as f32;

        // 1. Create nodes
        for (&cid, &freq) in &stats.concept_freq {
            if freq < self.config.min_degree as u64 {
                continue;
            }
            
            let name = self.interner.resolve(&cid.0).to_string();
            let mut node = ConceptNode::new(cid, name, NodeType::Concept);
            node.frequency = freq as u32;

            let idx = self.graph.add_node(node);
            self.id_to_node.insert(cid, idx);
        }

        // 2. Create edges with PMI
        for ((a, b), &pair_count) in &stats.pair_freq {
            if pair_count < self.config.cooccur_min as u64 {
                continue;
            }
            
            let idx_a = match self.id_to_node.get(&a) {
                Some(i) => *i,
                None => continue,
            };
            let idx_b = match self.id_to_node.get(&b) {
                Some(i) => *i,
                None => continue,
            };

            let freq_a = stats.concept_freq.get(&a).copied().unwrap_or(1) as f32;
            let freq_b = stats.concept_freq.get(&b).copied().unwrap_or(1) as f32;
            let count_ab = pair_count as f32;

            // PMI calculation
            // P(a,b) = count_ab / N
            // P(a) = freq_a / N
            // P(b) = freq_b / N
            // PMI = log( (count_ab * N) / (freq_a * freq_b) )
            
            let pmi = ((count_ab * total_windows) / (freq_a * freq_b)).ln();

            if pmi < self.config.pmi_min {
                continue;
            }

            let edge = ConceptEdge::new(pmi);
            self.graph.add_edge(idx_a, idx_b, edge);
        }

        Ok(ConceptGraph { inner: self.graph })
    }
}
