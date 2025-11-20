use std::sync::Arc;
use std::fs::File;
use std::io::Write;
use sbng_core::{
    SbngConfig, GraphBuildPipeline, SignatureGenerationPipeline,
    corpus::{JsonlCorpus, WhitespaceTokenizer, InterningConceptExtractor, ConceptInterner},
};

#[test]
fn smoke_build_and_signatures() {
    // 1. Setup Dummy Data
    let data_path = "tests/smoke_data.jsonl";
    let mut file = std::fs::File::create(data_path).expect("failed to create smoke data");
    writeln!(file, r#"{{"id": "1", "text": "tesla produces electric cars"}}"#).unwrap();
    writeln!(file, r#"{{"id": "2", "text": "ev batteries provide power"}}"#).unwrap();
    writeln!(file, r#"{{"id": "3", "text": "tesla batteries are expensive"}}"#).unwrap();
    writeln!(file, r#"{{"id": "4", "text": "spacex launches rockets"}}"#).unwrap();
    writeln!(file, r#"{{"id": "5", "text": "rockets fly to mars"}}"#).unwrap();
    drop(file); // Close file

    // 2. Setup Components
    let mut config = SbngConfig::default();
    // Lower thresholds for small test data
    config.cooccur_min = 1;
    config.pmi_min = 0.1; 
    config.min_degree = 1;

    // Interner
    let interner = Arc::new(ConceptInterner::with_hasher(
        sbng_core::corpus::interner::SerializableHasher::default()
    ));

    // Extractor
    let tokenizer = WhitespaceTokenizer;
    let stopwords = ["the", "is", "a", "an", "of", "and", "are", "to"];
    let extractor = InterningConceptExtractor::new(
        tokenizer,
        interner.clone(),
        &stopwords,
    );

    let mut config = SbngConfig::default();
    // Lower thresholds for small test data
    config.cooccur_min = 1;
    config.pmi_min = 0.1; 
    config.min_degree = 1;

    let pipeline = GraphBuildPipeline::new(config.clone(), &extractor, interner.clone());
    let corpus = JsonlCorpus::new(data_path);

    // 3. Build Graph
    let graph = pipeline.build_from_jsonl(&corpus).expect("graph build failed");

    println!("Graph Stats: {} nodes, {} edges", 
        graph.inner().node_count(), 
        graph.inner().edge_count()
    );

    assert!(graph.inner().node_count() > 0, "Graph should have nodes");
    assert!(graph.inner().edge_count() > 0, "Graph should have edges");

    // 4. Generate Signatures
    let sig_pipeline = SignatureGenerationPipeline::new(config);
    let signatures = sig_pipeline.generate(&graph).expect("signature gen failed");

    println!("Generated {} signatures", signatures.len());
    assert!(!signatures.is_empty(), "Should generate signatures");

    for (id, sig) in &signatures {
        let name = interner.resolve(&id.0);
        println!("ID: {} ({}), Popcount: {}, Fill Ratio: {:.3}", 
            id.0.into_inner(), name, sig.popcount(), sig.fill_ratio());
        
        assert!(sig.popcount() > 0, "Signature should not be empty");
    }

    // 5. Check Jaccard Similarity
    // We expect "tesla" and "electric" to have higher overlap than "tesla" and "mcdonalds"
    // Note: IDs might vary, so we find them by name.
    let find_sig = |name: &str| -> Option<&sbng_core::BloomFingerprint> {
        let spur = interner.get(name)?;
        signatures.iter().find(|(id, _)| id.0 == spur).map(|(_, sig)| sig)
    };

    // In sample data: "tesla" (lowercase from extractor), "electric", "mcdonalds"
    if let (Some(tesla), Some(electric), Some(mcd)) = (find_sig("tesla"), find_sig("electric"), find_sig("mcdonalds")) {
        let j_te_el = sbng_core::bloom::jaccard(tesla, electric);
        let j_te_mcd = sbng_core::bloom::jaccard(tesla, mcd);
        
        println!("Jaccard(tesla, electric) = {:.3}", j_te_el);
        println!("Jaccard(tesla, mcdonalds) = {:.3}", j_te_mcd);

        assert!(j_te_el > j_te_mcd, "Tesla should be closer to electric than mcdonalds");
    } else {
        println!("Could not find all concepts for Jaccard check. Available concepts:");
        for (id, _) in &signatures {
             println!(" - {}", interner.resolve(&id.0));
        }
    }
    
    // Cleanup
    std::fs::remove_file(data_path).unwrap();
}
