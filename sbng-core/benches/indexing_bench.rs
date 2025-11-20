//! Benchmark indexing performance.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::sync::Arc;

use sbng_core::{
    SbngConfig,
    corpus::{JsonlCorpus, WhitespaceTokenizer, InterningConceptExtractor, ConceptInterner},
    pipeline::GraphBuildPipeline,
};

fn bench_indexing(c: &mut Criterion) {
    let corpus_path = "data/sample.jsonl";
    
    c.bench_function("index_sample_corpus", |b| {
        b.iter(|| {
            // Setup
            let mut config = SbngConfig::default();
            config.cooccur_min = 1;
            config.pmi_min = 0.0;
            config.min_degree = 1;

            let interner = Arc::new(ConceptInterner::with_hasher(
                sbng_core::corpus::interner::SerializableHasher::default()
            ));

            let tokenizer = WhitespaceTokenizer;
            let stopwords = ["the", "is", "a", "an", "of", "and", "or", "for", "to", "in", "at", "by"];
            let extractor = InterningConceptExtractor::new(tokenizer, interner.clone(), &stopwords);

            let corpus = JsonlCorpus::new(corpus_path.to_string());
            let pipeline = GraphBuildPipeline::new(config, &extractor, interner.clone());
            
            // Benchmark: build graph
            let graph = pipeline.build_from_jsonl(&corpus).unwrap();
            black_box(graph);
        });
    });
}

criterion_group!(benches, bench_indexing);
criterion_main!(benches);
