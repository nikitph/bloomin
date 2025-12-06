//! Utility functions for FAISS-Sphere

use ndarray::{Array1, Array2, Axis};

/// Normalize vectors to unit length
pub fn normalize_vectors(vectors: &Array2<f32>) -> Array2<f32> {
    let norms = vectors.map_axis(Axis(1), |row| {
        row.dot(&row).sqrt().max(1e-10) // Avoid division by zero
    });
    
    vectors / &norms.insert_axis(Axis(1))
}

/// Compute recall between two sets of indices
pub fn compute_recall(ground_truth: &Array2<usize>, results: &Array2<usize>) -> f32 {
    let (n_queries, k) = ground_truth.dim();
    
    let mut total_recall = 0.0;
    
    for i in 0..n_queries {
        let gt_row = ground_truth.row(i);
        let result_row = results.row(i);
        
        let gt_set: std::collections::HashSet<_> = gt_row.iter().collect();
        let result_set: std::collections::HashSet<_> = result_row.iter().collect();
        
        let intersection = gt_set.intersection(&result_set).count();
        total_recall += intersection as f32 / k as f32;
    }
    
    total_recall / n_queries as f32
}

/// Generate random normalized vectors for testing
pub fn random_normalized_vectors(n: usize, d: usize) -> Array2<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    
    let data = Array2::from_shape_fn((n, d), |_| rng.gen::<f32>() - 0.5);
    normalize_vectors(&data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let data = random_normalized_vectors(100, 50);
        
        // Check all vectors are unit length
        for row in data.axis_iter(Axis(0)) {
            let norm = row.dot(&row).sqrt();
            assert!((norm - 1.0).abs() < 1e-5);
        }
    }
}
