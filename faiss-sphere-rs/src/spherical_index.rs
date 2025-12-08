//! Spherical index with geodesic distance computations
//!
//! Fast inner product search with optional geodesic distance conversion

use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use std::cmp::Ordering;

/// Spherical index for fast similarity search
///
/// Optimized for normalized vectors on the unit hypersphere (k=1).
/// For normalized vectors, Euclidean distance ranking is equivalent to
/// inner product ranking: ||v-q||² = 2 - 2<v,q>, so we maximize <v,q>.
pub struct SphericalIndex {
    /// Database vectors (N × D) - assumed normalized
    database: Option<Array2<f32>>,
    /// Dimension
    dimension: usize,
}

impl SphericalIndex {
    /// Create new spherical index
    pub fn new(dimension: usize) -> Self {
        Self {
            database: None,
            dimension,
        }
    }

    /// Add vectors to index
    ///
    /// Vectors are assumed to be normalized (||v|| = 1)
    pub fn add(&mut self, vectors: Array2<f32>) -> crate::Result<()> {
        let (_, d) = vectors.dim();
        
        if d != self.dimension {
            return Err(format!(
                "Vector dimension {} doesn't match index dimension {}",
                d, self.dimension
            ).into());
        }

        self.database = Some(vectors);
        Ok(())
    }

    /// Search for k nearest neighbors
    ///
    /// For normalized vectors, ranks by inner product (equivalent to Euclidean distance)
    pub fn search(&self, queries: &Array2<f32>, k: usize) -> crate::Result<(Array2<f32>, Array2<usize>)> {
        let db = self.database.as_ref()
            .ok_or("Index is empty")?;
        
        let (n_queries, _) = queries.dim();
        let (_n_db, _) = db.dim();
        
        let mut distances = Array2::zeros((n_queries, k));
        let mut indices = Array2::zeros((n_queries, k));
        
        // Compute inner products for each query
        for (i, query) in queries.axis_iter(Axis(0)).enumerate() {
            let mut scores: Vec<(f32, usize)> = db.axis_iter(Axis(0))
                .enumerate()
                .map(|(idx, db_vec)| {
                    let ip = query.dot(&db_vec);
                    (ip, idx)
                })
                .collect();
            
            // Sort by inner product (descending = nearest neighbors)
            scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
            
            // Take top k
            for (j, (score, idx)) in scores.iter().take(k).enumerate() {
                distances[[i, j]] = *score;
                indices[[i, j]] = *idx;
            }
        }
        
        Ok((distances, indices))
    }

    /// Search in parallel (for large queries)
    ///
    /// For normalized vectors, ranks by inner product (equivalent to Euclidean distance)
    pub fn search_parallel(&self, queries: &Array2<f32>, k: usize) -> crate::Result<(Array2<f32>, Array2<usize>)> {
        let db = self.database.as_ref()
            .ok_or("Index is empty")?;
        
        let (n_queries, _) = queries.dim();
        
        // Process queries in parallel
        let results: Vec<(Vec<f32>, Vec<usize>)> = (0..n_queries)
            .into_par_iter()
            .map(|i| {
                let query = queries.row(i);
                
                // Compute inner products (fast FMA)
                let mut scores: Vec<(f32, usize)> = (0..db.nrows())
                    .map(|idx| {
                        let db_vec = db.row(idx);
                        let mut ip = 0.0;
                        for j in 0..query.len() {
                            ip += query[j] * db_vec[j];
                        }
                        (ip, idx)
                    })
                    .collect();
                
                scores.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
                
                let top_k: Vec<(f32, usize)> = scores.into_iter().take(k).collect();
                let dists: Vec<f32> = top_k.iter().map(|(d, _)| *d).collect();
                let idxs: Vec<usize> = top_k.iter().map(|(_, i)| *i).collect();
                
                (dists, idxs)
            })
            .collect();
        
        // Assemble results
        let mut distances = Array2::zeros((n_queries, k));
        let mut indices = Array2::zeros((n_queries, k));
        
        for (i, (dists, idxs)) in results.into_iter().enumerate() {
            for (j, (d, idx)) in dists.into_iter().zip(idxs).enumerate() {
                distances[[i, j]] = d;
                indices[[i, j]] = idx;
            }
        }
        
        Ok((distances, indices))
    }

    /// Get number of vectors in index
    pub fn len(&self) -> usize {
        self.database.as_ref().map(|db| db.nrows()).unwrap_or(0)
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_search() {
        let n = 1000;
        let d = 320;
        let k = 10;
        
        // Generate random database
        let db = Array::from_shape_fn((n, d), |(i, j)| {
            ((i * 7 + j * 13) % 100) as f32 / 100.0
        });
        
        // Create index
        let mut index = SphericalIndex::new(d);
        index.add(db.clone()).unwrap();
        
        // Query
        let queries = db.slice(s![..10, ..]).to_owned();
        let (distances, indices) = index.search(&queries, k).unwrap();
        
        assert_eq!(distances.dim(), (10, k));
        assert_eq!(indices.dim(), (10, k));
        
        // First result should be the query itself
        for i in 0..10 {
            assert_eq!(indices[[i, 0]], i);
        }
    }
}
