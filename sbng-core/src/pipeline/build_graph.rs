//! Orchestrates: corpus -> cooccurrence -> PMI -> ConceptGraph.

use std::sync::Arc;
use rayon::prelude::*;

use crate::{
    config::SbngConfig,
    corpus::{ConceptExtractor, JsonlCorpus},
    errors::Result,
    graph::{ConceptGraph, ConceptGraphBuilder, StatsAccumulator, GraphPruner},
};

use crate::corpus::ConceptInterner;

/// High-level pipeline: corpus -> stats -> graph.
pub struct GraphBuildPipeline<'a> {
    config: SbngConfig,
    extractor: &'a dyn ConceptExtractor,
    interner: Arc<ConceptInterner>,
}

impl<'a> std::fmt::Debug for GraphBuildPipeline<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GraphBuildPipeline")
            .field("config", &self.config)
            .field("extractor", &"<ConceptExtractor>")
            .field("interner", &self.interner)
            .finish()
    }
}

impl<'a> GraphBuildPipeline<'a> {
    /// Create a new pipeline with the given config, extractor, and interner.
    pub fn new(
        config: SbngConfig,
        extractor: &'a dyn ConceptExtractor,
        interner: Arc<ConceptInterner>,
    ) -> Self {
        Self {
            config,
            extractor,
            interner,
        }
    }

    /// Build the graph from a JSONL corpus.
    pub fn build_from_jsonl(&self, corpus: &JsonlCorpus) -> Result<ConceptGraph> {
        let window_size = self.config.window_size;

        // Parallel stats accumulation (Map-Reduce)
        let stats = corpus.iter()?
            .par_bridge()
            .fold(StatsAccumulator::new, |mut acc, doc_res| {
                if let Ok(doc) = doc_res {
                    let concepts = self.extractor.extract_concepts(&doc.text);
                    let ids: Vec<_> = concepts.iter().map(|c| c.concept_id).collect();

                    if !ids.is_empty() {
                        // sliding windows over concepts
                        for win in ids.windows(window_size) {
                            acc.add_window(win);
                        }
                        // Also handle short documents
                        if ids.len() < window_size {
                            acc.add_window(&ids);
                        }
                    }
                }
                acc
            })
            .reduce(StatsAccumulator::new, |mut a, b| {
                a.merge(b);
                a
            });

        let mut builder = ConceptGraphBuilder::new(self.config.clone(), self.interner.clone());
        builder.ingest_stats(stats);
        let mut graph = builder.finalize()?;

        GraphPruner::mark_hubs(&mut graph, &self.config);
        GraphPruner::entropy_constrained_prune(&mut graph);

        Ok(graph)
    }
}
