use std::sync::Arc;
use sbng_core::corpus::{JsonlCorpus, WhitespaceTokenizer, InterningConceptExtractor, ConceptInterner};
use sbng_core::{SbngConfig, GraphBuildPipeline};

#[test]
fn user_snippet_verification() -> anyhow::Result<()> {
    // Interner
    let interner = Arc::new(ConceptInterner::with_hasher(
        sbng_core::corpus::interner::SerializableHasher::default()
    ));

    let tokenizer = WhitespaceTokenizer;
    let stopwords = ["the", "is", "a", "an", "of", "and"];

    // Extractor
    let extractor = InterningConceptExtractor::new(
        tokenizer,
        interner.clone(),
        &stopwords,
    );

    let config = SbngConfig::default();
    let pipeline = GraphBuildPipeline::new(config, &extractor, interner.clone());

    // Ensure the path is correct relative to where cargo test runs
    // Cargo runs tests with CWD as the package root.
    let corpus = JsonlCorpus::new("data/sample.jsonl");
    let graph = pipeline.build_from_jsonl(&corpus)?;

    println!("nodes: {}, edges: {}", 
        graph.inner().node_count(),
        graph.inner().edge_count());

    Ok(())
}
