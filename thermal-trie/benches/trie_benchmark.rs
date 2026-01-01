use criterion::{black_box, criterion_group, criterion_main, Criterion};
use thermal_trie::{ThermalTrie, ThermalConfig};
use std::fs::File;
use std::io::{BufRead, BufReader};

fn load_words() -> Vec<String> {
    let path = "/usr/share/dict/words";
    if let Ok(file) = File::open(path) {
        let reader = BufReader::new(file);
        reader.lines()
            .map(|l| l.unwrap())
            .filter(|w| w.len() > 2 && w.is_ascii())
            .take(5000) 
            .collect()
    } else {
        vec!["apple".to_string(), "banana".to_string()]
    }
}

fn bench_insert(c: &mut Criterion) {
    let words = load_words();
    c.bench_function("trie_insert_5000", |b| {
        b.iter(|| {
            let config = ThermalConfig::default();
            let mut trie = ThermalTrie::new(config);
            for word in &words {
                trie.insert(black_box(word), 1.0);
            }
        })
    });
}

fn bench_variants(c: &mut Criterion) {
    let words = load_words();
    
    // Setup Baseline (Plain)
    let mut trie_base = ThermalTrie::new(ThermalConfig::default());
    for word in &words { trie_base.insert(word, 1.0); }

    // Setup Weighted
    let mut config_weighted = ThermalConfig::default();
    config_weighted.mode = thermal_trie::DiffusionMode::Weighted;
    let mut trie_weighted = ThermalTrie::new(config_weighted);
    for word in &words { trie_weighted.insert(word, 1.0); }

    // Setup Gravity
    let mut config_gravity = ThermalConfig::default();
    config_gravity.mode = thermal_trie::DiffusionMode::Gravity;
    config_gravity.gravity_beta = 0.1;
    let mut trie_gravity = ThermalTrie::new(config_gravity);
    for word in &words { trie_gravity.insert(word, 1.0); }
    trie_gravity.insert("apple", 10.0);

    // Bench Search - Baseline
    c.bench_function("search_baseline", |b| {
        b.iter(|| {
            trie_base.fuzzy_search(black_box("histry"), 5);
        })
    });

    // Bench Search - Weighted
    c.bench_function("search_weighted", |b| {
        b.iter(|| {
            // Typing 'histery' (e->o is close-ish?)
            trie_weighted.fuzzy_search(black_box("histery"), 5);
        })
    });

    // Bench Search - Gravity
    c.bench_function("search_gravity", |b| {
        b.iter(|| {
            trie_gravity.fuzzy_search(black_box("appl"), 5);
        })
    });

    // Bench Search - Semantic (Mock)
    let mut graph_sem = thermal_trie::SemanticThermalGraph::new(ThermalConfig::default());
    // Insert 100 random vectors
    for i in 0..100 {
        let v = vec![i as f32, (i as f32).sin()];
        graph_sem.insert(v, 1.0);
    }
    c.bench_function("search_semantic", |b| {
        b.iter(|| {
            graph_sem.inject_heat_at_vector(&[50.0, 0.0]);
            graph_sem.relax(5);
        })
    });
}

criterion_group!(benches, bench_insert, bench_variants);
criterion_main!(benches);
