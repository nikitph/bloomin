//! H-Tree: Holographic B-Tree Demo
//!
//! This demonstrates the key properties of the H-Tree:
//! 1. O(log N) vector search
//! 2. O(1) vacuum detection
//! 3. Merkle tree integrity
//!
//! For comprehensive benchmarks, run: cargo run --release --bin benchmark

use h_tree::{HTree, Vector};

fn main() {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  H-Tree: Holographic B-Tree for Vector Search              ║");
    println!("║  Based on Witness Field Theory                             ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // Create a new H-Tree
    let mut tree = HTree::default();

    println!("1. Inserting vectors...\n");

    // Insert some sample vectors
    let vectors: Vec<Vector> = (0..1000)
        .map(|i| Vector::random(i, 128))
        .collect();

    for v in &vectors {
        tree.insert(v.clone());
    }

    let stats = tree.stats();
    println!("   Inserted {} vectors", stats.vector_count);
    println!("   Tree height: {}", stats.height);
    println!("   Node count: {}", stats.node_count);
    println!("   Memory: {:.2} KB\n", stats.memory_bytes as f64 / 1024.0);

    // Query for nearest neighbors
    println!("2. Querying for k-nearest neighbors...\n");

    let query = &vectors[500]; // Use an existing vector as query
    let (results, query_stats) = tree.query_with_stats(query, 10);

    println!("   Query for vector #{}", query.id);
    println!("   Top 10 results:");
    for (i, r) in results.iter().enumerate() {
        println!("     {}. ID {} (similarity: {:.4})", i + 1, r.vector_id, r.similarity);
    }

    println!("\n   Query statistics:");
    println!("     Nodes visited: {}", query_stats.nodes_visited);
    println!("     Vacuum pruned: {}", query_stats.vacuum_pruned);
    println!("     Leaves searched: {}", query_stats.leaves_searched);
    println!("     Vectors compared: {}", query_stats.vectors_compared);

    // Demonstrate vacuum detection
    println!("\n3. Vacuum detection (O(1) rejection)...\n");

    // Create a query far from any data
    let vacuum_query = Vector::new(9999, vec![1000.0; 128]);

    let heat = tree.root_heat(&vacuum_query);
    let is_vacuum = tree.is_vacuum(&vacuum_query);

    println!("   Query in distant region:");
    println!("   Root heat: {:.6}", heat);
    println!("   Is vacuum: {}\n", is_vacuum);

    // Verify Merkle integrity
    println!("4. Merkle tree integrity verification...\n");

    let integrity_ok = tree.verify_integrity();
    println!("   Integrity check: {}\n",
             if integrity_ok { "PASSED ✓" } else { "FAILED ✗" });

    // Demonstrate proof generation
    println!("5. Generating existence proof for vector #500...\n");

    if let Some(proof) = tree.generate_proof(500) {
        println!("   Proof generated!");
        println!("   Path length: {} nodes", proof.path.len());
        println!("   Proof size: {} bytes", proof.size());
    }

    println!("\n════════════════════════════════════════════════════════════");
    println!("  For comprehensive benchmarks, run:");
    println!("  cargo run --release --bin benchmark");
    println!("════════════════════════════════════════════════════════════\n");
}
