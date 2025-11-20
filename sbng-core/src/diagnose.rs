use std::path::PathBuf;
use std::sync::Arc;
use std::collections::HashMap;
use std::io::{self, Write};

use clap::{Parser, Subcommand};
use sbng_core::{
    SbngConfig,
    corpus::{JsonlCorpus, WhitespaceTokenizer, InterningConceptExtractor, ConceptInterner},
    pipeline::{GraphBuildPipeline, SignatureGenerationPipeline},
    search::{DocIndex, QueryEngine},
    persistence,
    BloomFingerprint,
    types::ConceptId,
};

// ... existing CLI code ...

/// Diagnose index health: graph statistics, Bloom fill rates, hub analysis.
fn cmd_diagnose(index_dir: PathBuf) -> anyhow::Result<()> {
    println!("=== SBNG Index Diagnostics ===\n");
    println!("Loading index from {}...", index_dir.display());

    // Load index
    let (metadata, interner, concept_fps, doc_index) = persistence::load_index(&index_dir)?;
    
    println!("Index loaded successfully.\n");
    println!("=== Configuration ===");
    println!("  Bloom bits: {}", metadata.config.bloom_bits);
    println!("  Bloom hashes: {}", metadata.config.bloom_hashes);
    println!("  PMI min: {}", metadata.config.pmi_min);
    println!("  Cooccur min: {}", metadata.config.cooccur_min);
    println!("  Max degree: {}", metadata.config.max_degree);
    println!("  Max neighborhood: {}", metadata.config.max_neighborhood);
    println!();

    // === Graph Statistics ===
    println!("=== Graph Statistics ===");
    println!("  Total concepts (nodes): {}", concept_fps.len());
    println!("  Total documents: {}", doc_index.doc_ids.len());
    
    // We don't have the graph loaded, but we can analyze concept fingerprints
    // to understand neighborhood sizes
    
    // === Bloom Fill Rate Analysis ===
    println!("\n=== Bloom Filter Analysis ===");
    
    let mut fill_rates: Vec<f64> = concept_fps.values()
        .map(|fp| fp.fill_ratio())
        .collect();
    
    if fill_rates.is_empty() {
        println!("  No concept fingerprints found.");
        return Ok(());
    }
    
    fill_rates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let avg_fill = fill_rates.iter().sum::<f64>() / fill_rates.len() as f64;
    let min_fill = fill_rates.first().copied().unwrap_or(0.0);
    let max_fill = fill_rates.last().copied().unwrap_or(0.0);
    let median_fill = fill_rates[fill_rates.len() / 2];
    
    println!("  Average fill rate: {:.1}%", avg_fill * 100.0);
    println!("  Median fill rate: {:.1}%", median_fill * 100.0);
    println!("  Min fill rate: {:.1}%", min_fill * 100.0);
    println!("  Max fill rate: {:.1}%", max_fill * 100.0);
    
    // Histogram
    println!("\n  Fill Rate Distribution:");
    let buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)];
    for (low, high) in buckets {
        let count = fill_rates.iter()
            .filter(|&&rate| rate >= low && rate < high)
            .count();
        let pct = (count as f64 / fill_rates.len() as f64) * 100.0;
        let bar = "█".repeat((pct / 2.0) as usize);
        println!("    {:.0}%-{:.0}%: {:>5} ({:>5.1}%) {}", 
            low * 100.0, high * 100.0, count, pct, bar);
    }
    
    // === Document Fingerprint Analysis ===
    println!("\n=== Document Fingerprints ===");
    let doc_fill_rates: Vec<f64> = doc_index.fingerprints.iter()
        .map(|fp| fp.fill_ratio())
        .collect();
    
    if !doc_fill_rates.is_empty() {
        let avg_doc_fill = doc_fill_rates.iter().sum::<f64>() / doc_fill_rates.len() as f64;
        println!("  Average doc fill rate: {:.1}%", avg_doc_fill * 100.0);
    }
    
    // === Top Concepts by Fingerprint Size ===
    println!("\n=== Top 10 Concepts by Neighborhood Size ===");
    let mut concept_sizes: Vec<(ConceptId, usize)> = concept_fps.iter()
        .map(|(id, fp)| (*id, fp.count_ones()))
        .collect();
    concept_sizes.sort_by_key(|(_, size)| std::cmp::Reverse(*size));
    
    for (i, (concept_id, size)) in concept_sizes.iter().take(10).enumerate() {
        let concept_str = interner.resolve(&concept_id.0);
        let fill = concept_fps.get(concept_id).map(|fp| fp.fill_ratio()).unwrap_or(0.0);
        println!("  {:2}. {:20} - {} bits set ({:.1}% fill)", 
            i + 1, concept_str, size, fill * 100.0);
    }
    
    // === Health Recommendations ===
    println!("\n=== Health Assessment ===");
    
    if avg_fill > 0.7 {
        println!("  ⚠️  WARNING: Average fill rate is HIGH ({:.1}%)", avg_fill * 100.0);
        println!("     → Neighborhoods may be too large or hubs are leaking in");
        println!("     → Recommendations:");
        println!("       - Increase pmi_min (current: {})", metadata.config.pmi_min);
        println!("       - Increase cooccur_min (current: {})", metadata.config.cooccur_min);
        println!("       - Decrease max_neighborhood (current: {})", metadata.config.max_neighborhood);
    } else if avg_fill > 0.6 {
        println!("  ⚠️  CAUTION: Average fill rate is moderately high ({:.1}%)", avg_fill * 100.0);
        println!("     → Consider tightening pruning parameters");
    } else if avg_fill < 0.1 {
        println!("  ⚠️  WARNING: Average fill rate is LOW ({:.1}%)", avg_fill * 100.0);
        println!("     → Graph may be too sparse, neighborhoods too small");
        println!("     → Recommendations:");
        println!("       - Decrease pmi_min (current: {})", metadata.config.pmi_min);
        println!("       - Decrease cooccur_min (current: {})", metadata.config.cooccur_min);
        println!("       - Increase max_neighborhood (current: {})", metadata.config.max_neighborhood);
    } else {
        println!("  ✓ Fill rate is in healthy range ({:.1}%)", avg_fill * 100.0);
        println!("    Optimal range: 30-60%");
    }
    
    if max_fill > 0.8 {
        println!("\n  ⚠️  WARNING: Some concepts have very high fill rates (max: {:.1}%)", max_fill * 100.0);
        println!("     → These may be hubs that should be suppressed");
        println!("     → Check max_degree setting (current: {})", metadata.config.max_degree);
    }
    
    println!("\n=== Diagnostics Complete ===");
    
    Ok(())
}
