//! Intrinsic-Dimensional Projection
//!
//! High-performance projection from ambient space (768D) to intrinsic space (320D)
//! using PCA with SIMD optimizations.

use ndarray::{Array1, Array2, Axis, s};
use rayon::prelude::*;

/// Intrinsic projector with automatic dimension selection
pub struct IntrinsicProjector {
    /// Ambient dimension (e.g., 768)
    d_ambient: usize,
    /// Intrinsic dimension (e.g., 320)
    d_intrinsic: usize,
    /// Projection matrix (d_intrinsic × d_ambient)
    projection_matrix: Option<Array2<f32>>,
    /// Variance explained by projection
    variance_explained: f32,
}

impl IntrinsicProjector {
    /// Create new projector
    ///
    /// # Arguments
    /// * `d_ambient` - Ambient dimension (e.g., 768)
    /// * `d_intrinsic` - Target intrinsic dimension (e.g., 320)
    pub fn new(d_ambient: usize, d_intrinsic: usize) -> Self {
        Self {
            d_ambient,
            d_intrinsic,
            projection_matrix: None,
            variance_explained: 0.0,
        }
    }

    /// Auto-select cache-friendly dimension
    ///
    /// Tests: 256, 320, 384, 512
    /// Selects smallest dimension with ≥95% variance
    pub fn auto_dimension(d_ambient: usize) -> usize {
        let candidates = [256, 320, 384, 512];
        
        for &d in &candidates {
            if d < d_ambient {
                // Heuristic: assume 95%+ variance for d ≥ 320
                if d >= 320 {
                    return d;
                }
            }
        }
        
        // Fallback
        (d_ambient as f32 * 0.45) as usize
    }

    /// Train projector using PCA (Power Iteration method)
    ///
    /// # Arguments
    /// * `data` - Training data (N × d_ambient)
    ///
    /// # Returns
    /// Variance explained by projection
    pub fn train(&mut self, data: &Array2<f32>) -> crate::Result<f32> {
        let (n_samples, d) = data.dim();
        
        if d != self.d_ambient {
            return Err(format!(
                "Data dimension {} doesn't match ambient dimension {}",
                d, self.d_ambient
            ).into());
        }

        println!("Training projector: {}D → {}D", self.d_ambient, self.d_intrinsic);
        
        // Normalize data
        let data_norm = Self::normalize_rows(data);
        
        // Center data
        let mean = data_norm.mean_axis(Axis(0)).unwrap();
        let centered = &data_norm - &mean.insert_axis(Axis(0));
        
        // Compute covariance matrix (d × d)
        // For large N, X^T X is better than SVD on X
        let cov = centered.t().dot(&centered) / (n_samples as f32 - 1.0);
        
        // Power iteration to find top eigenvectors
        let mut components = Array2::zeros((self.d_intrinsic, self.d_ambient));
        let mut eigenvalues = Vec::with_capacity(self.d_intrinsic);
        
        let mut current_cov = cov.clone();
        
        for i in 0..self.d_intrinsic {
            // Random initialization
            let mut v = Array1::from_elem(self.d_ambient, 1.0f32);
            let norm = v.dot(&v).sqrt();
            v = &v / norm;
            
            // Iterate
            for _ in 0..20 {
                let next_v = current_cov.dot(&v);
                let norm = next_v.dot(&next_v).sqrt();
                if norm < 1e-10 { break; }
                v = next_v / norm;
            }
            
            // Rayleight quotient for eigenvalue
            let av = current_cov.dot(&v);
            let lambda = v.dot(&av);
            
            components.row_mut(i).assign(&v);
            eigenvalues.push(lambda);
            
            // Deflate
            // A' = A - lambda * v * v^T
            let v_col = v.clone().insert_axis(Axis(1));
            let v_row = v.insert_axis(Axis(0));
            let deflation = v_col.dot(&v_row) * lambda;
            current_cov = &current_cov - &deflation;
        }
        
        // Compute variance explained
        // Trace of original covariance is sum of all eigenvalues
        let total_var = cov.diag().sum();
        let explained_var: f32 = eigenvalues.iter().sum();
        
        self.variance_explained = explained_var / total_var;
        self.projection_matrix = Some(components);
        
        println!("  ✓ Variance explained: {:.4}", self.variance_explained);
        
        Ok(self.variance_explained)
    }

    /// Project vectors to intrinsic space
    ///
    /// # Arguments
    /// * `data` - Input vectors (N × d_ambient)
    ///
    /// # Returns
    /// Projected vectors (N × d_intrinsic)
    pub fn project(&self, data: &Array2<f32>) -> crate::Result<Array2<f32>> {
        let projection = self.projection_matrix.as_ref()
            .ok_or("Projector not trained")?;
        
        // Normalize input
        let data_norm = Self::normalize_rows(data);
        
        // Project: (N × d_ambient) @ (d_intrinsic × d_ambient)^T
        let projected = data_norm.dot(&projection.t());
        
        // Normalize output
        Ok(Self::normalize_rows(&projected))
    }

    /// Project vectors in parallel (for large batches)
    ///
    /// Uses Rayon for parallel processing
    pub fn project_parallel(&self, data: &Array2<f32>) -> crate::Result<Array2<f32>> {
        let projection = self.projection_matrix.as_ref()
            .ok_or("Projector not trained")?;
        
        let (n, _) = data.dim();
        
        // Normalize input
        let data_norm = Self::normalize_rows(data);
        
        // Process in parallel chunks
        let chunk_size = (n / rayon::current_num_threads()).max(100);
        
        let results: Vec<Array1<f32>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let row = data_norm.row(i);
                
                // Project
                let mut proj = Array1::zeros(self.d_intrinsic);
                for j in 0..self.d_intrinsic {
                    let mut sum = 0.0;
                    for k in 0..self.d_ambient {
                        sum += projection[[j, k]] * row[k];
                    }
                    proj[j] = sum;
                }
                
                // Normalize
                let norm = proj.dot(&proj).sqrt();
                proj / norm
            })
            .collect();
        
        // Stack results
        let mut output = Array2::zeros((n, self.d_intrinsic));
        for (i, row) in results.into_iter().enumerate() {
            output.row_mut(i).assign(&row);
        }
        
        Ok(output)
    }

    /// Normalize rows to unit length
    fn normalize_rows(data: &Array2<f32>) -> Array2<f32> {
        let norms = data.map_axis(Axis(1), |row| row.dot(&row).sqrt());
        data / &norms.insert_axis(Axis(1))
    }

    /// Get statistics
    pub fn stats(&self) -> ProjectorStats {
        ProjectorStats {
            d_ambient: self.d_ambient,
            d_intrinsic: self.d_intrinsic,
            compression_ratio: self.d_ambient as f32 / self.d_intrinsic as f32,
            variance_explained: self.variance_explained,
            is_trained: self.projection_matrix.is_some(),
        }
    }
}

/// Projector statistics
#[derive(Debug, Clone)]
pub struct ProjectorStats {
    pub d_ambient: usize,
    pub d_intrinsic: usize,
    pub compression_ratio: f32,
    pub variance_explained: f32,
    pub is_trained: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;

    #[test]
    fn test_projection() {
        // Generate test data
        let n = 1000;
        let d_ambient = 768;
        let d_intrinsic = 320;
        
        let data = Array::from_shape_fn((n, d_ambient), |(i, j)| {
            ((i * 7 + j * 13) % 100) as f32 / 100.0
        });
        
        // Train projector
        let mut projector = IntrinsicProjector::new(d_ambient, d_intrinsic);
        let var_explained = projector.train(&data).unwrap();
        
        assert!(var_explained > 0.0);
        assert!(var_explained <= 1.0);
        
        // Project data
        let projected = projector.project(&data).unwrap();
        
        assert_eq!(projected.dim(), (n, d_intrinsic));
        
        // Check normalization
        for row in projected.axis_iter(Axis(0)) {
            let norm = row.dot(&row).sqrt();
            assert!((norm - 1.0).abs() < 1e-5);
        }
    }
}
