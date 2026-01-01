use hyperbolic_hnsw::embedding::{HyperbolicEmbedder, TreeNode};
use hyperbolic_hnsw::hnsw::HyperbolicHNSW;
use ndarray::{Array1, Array2};
use rand::Rng;
use std::collections::HashMap;

fn main() {
    println!("============================================================");
    println!("EXAMPLE 1: Random vectors");
    println!("============================================================");
    example_usage();

    println!("\n============================================================");
    println!("EXAMPLE 2: Tree hierarchy");
    println!("============================================================");
    example_hierarchy();
}

fn example_usage() {
    // dim = 20, curvature = 1.0
    let dim = 20;
    let curvature = 1.0;

    let embedder = HyperbolicEmbedder::new(dim, curvature);
    // Parameters: M=16, M_max=16, M_max_0=32, ef_construction=200, ml=1/ln(2)
    let ml = 1.0 / (2.0f64).ln(); // approx 1.44
    let mut index = HyperbolicHNSW::new(dim, curvature, 16, 16, 32, 200, ml);

    println!("Generating random Euclidean vectors...");
    let n_points = 1000;
    let d_euclidean = 20; // Matches dim for simplicity, or could be higher
    let mut rng = rand::thread_rng();
    
    // Create random data
    let mut data_vec = Vec::with_capacity(n_points * d_euclidean);
    for _ in 0..n_points * d_euclidean {
        data_vec.push(rng.gen_range(-1.0..1.0));
    }
    let euclidean_vectors = Array2::from_shape_vec((n_points, d_euclidean), data_vec).unwrap();

    println!("Embedding vectors into hyperbolic space...");
    let hyperbolic_points = embedder.embed_euclidean(&euclidean_vectors);

    println!("Building hyperbolic HNSW index...");
    for (i, point) in hyperbolic_points.iter().enumerate() {
        index.insert(point.clone());
        if (i + 1) % 100 == 0 {
            println!("Inserted {} points...", i + 1);
        }
    }

    println!("Index built: {} points, max layer: {}", index.data.len(), index.max_layer);

    // Query
    let query = &hyperbolic_points[0];
    println!("\nSearching for 10 nearest neighbors of point 0...");
    let results = index.search(query, 10, 50);

    println!("\nResults:");
    for (rank, (idx, dist)) in results.iter().enumerate() {
        println!("{}. Index {}, Distance: {:.4}", rank + 1, idx, dist);
    }
}

fn example_hierarchy() {
    // Create simple tree
    // Depth 3, branching factor 3
    let tree = build_tree(3, 3, "root".to_string());
    
    let dim = 10;
    let curvature = 1.0;
    let embedder = HyperbolicEmbedder::new(dim, curvature);
    
    println!("Embedding tree...");
    let embeddings = embedder.embed_tree(&tree, 0.5);
    println!("Embedded {} nodes", embeddings.len());

    let ml = 1.0 / (2.0f64).ln();
    let mut index = HyperbolicHNSW::new(dim, curvature, 16, 16, 32, 200, ml);
    
    let mut id_to_idx = HashMap::new();
    let mut idx_to_id = HashMap::new();
    
    for (id, coords) in &embeddings {
        let idx = index.data.len();
        id_to_idx.insert(id.clone(), idx);
        idx_to_id.insert(idx, id.clone());
        index.insert(coords.clone());
    }
    
    let query_id = "root.0";
    if let Some(query_coords) = embeddings.get(query_id) {
         println!("\nNearest neighbors of {}:", query_id);
         let results = index.search(query_coords, 5, 50);
         
         for (rank, (idx, dist)) in results.iter().enumerate() {
             if let Some(node_id) = idx_to_id.get(idx) {
                 println!("{}. {}, Distance: {:.4}", rank + 1, node_id, dist);
             }
         }
    } else {
        println!("Query ID {} not found", query_id);
    }
}

fn build_tree(depth: usize, branch_factor: usize, prefix: String) -> TreeNode {
    if depth == 0 {
        return TreeNode { id: prefix, children: Vec::new() };
    }
    
    let mut children = Vec::new();
    for i in 0..branch_factor {
        let child_prefix = if prefix == "root" {
            format!("root.{}", i)
        } else {
            format!("{}.{}", prefix, i)
        };
        children.push(build_tree(depth - 1, branch_factor, child_prefix));
    }
    
    TreeNode { id: prefix, children }
}
