//! Spherical Cap Tree for efficient range queries on the unit sphere
//!
//! Provides O(log N) range queries instead of O(N) linear scans

use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;

/// Spherical Cap Tree node
pub struct SphericalCapTree {
    /// Center of this cap (normalized)
    center: Array1<f32>,
    /// Angular radius of this cap (in radians)
    radius: f32,
    /// Points in this node (if leaf)
    points: Option<Vec<usize>>,
    /// Left child
    left: Option<Box<SphericalCapTree>>,
    /// Right child
    right: Option<Box<SphericalCapTree>>,
}

impl SphericalCapTree {
    /// Build a spherical cap tree from normalized vectors
    ///
    /// # Arguments
    /// * `data` - Normalized vectors (N × D)
    /// * `max_leaf_size` - Maximum points in a leaf node
    pub fn build(data: &Array2<f32>, max_leaf_size: usize) -> Self {
        let indices: Vec<usize> = (0..data.nrows()).collect();
        Self::build_recursive(data, indices, max_leaf_size)
    }

    fn build_recursive(data: &Array2<f32>, indices: Vec<usize>, max_leaf_size: usize) -> Self {
        // Compute centroid on sphere (mean direction)
        let center = Self::compute_spherical_centroid(data, &indices);
        
        // Compute angular radius (max distance from center)
        let radius = indices.iter()
            .map(|&idx| Self::angular_distance(&center, &data.row(idx)))
            .fold(0.0f32, f32::max);
        
        // If small enough, make a leaf
        if indices.len() <= max_leaf_size {
            return Self {
                center,
                radius,
                points: Some(indices),
                left: None,
                right: None,
            };
        }
        
        // Partition by hemisphere
        let (left_indices, right_indices) = Self::partition_by_hemisphere(data, &indices, &center);
        
        // Recursively build children
        let left = if !left_indices.is_empty() {
            Some(Box::new(Self::build_recursive(data, left_indices, max_leaf_size)))
        } else {
            None
        };
        
        let right = if !right_indices.is_empty() {
            Some(Box::new(Self::build_recursive(data, right_indices, max_leaf_size)))
        } else {
            None
        };
        
        Self {
            center,
            radius,
            points: None,
            left,
            right,
        }
    }

    /// Range query: find all points within angular distance θ of query
    ///
    /// # Arguments
    /// * `query` - Query vector (normalized)
    /// * `theta` - Angular threshold in radians
    ///
    /// # Returns
    /// Indices of points within the spherical cap
    pub fn range_query(&self, query: &Array1<f32>, theta: f32) -> Vec<usize> {
        let mut results = Vec::new();
        self.range_query_recursive(query, theta, &mut results);
        results
    }

    fn range_query_recursive(&self, query: &Array1<f32>, theta: f32, results: &mut Vec<usize>) {
        // Check if query cap intersects this cap
        let dist_to_center = Self::angular_distance(query, &self.center);
        
        // Prune if caps don't intersect
        if dist_to_center - theta > self.radius {
            return;
        }
        
        // If leaf, check all points
        if let Some(ref points) = self.points {
            // This is a leaf - we would need access to data here
            // For now, just return the indices (caller will filter)
            results.extend(points.iter().copied());
            return;
        }
        
        // Recursively search children
        if let Some(ref left) = self.left {
            left.range_query_recursive(query, theta, results);
        }
        if let Some(ref right) = self.right {
            right.range_query_recursive(query, theta, results);
        }
    }

    /// Compute spherical centroid (mean direction on sphere)
    fn compute_spherical_centroid(data: &Array2<f32>, indices: &[usize]) -> Array1<f32> {
        let mut sum: Array1<f32> = Array1::zeros(data.ncols());
        
        for &idx in indices {
            sum = sum + data.row(idx);
        }
        
        // Normalize to unit sphere
        let norm = sum.dot(&sum).sqrt();
        if norm > 1e-10 {
            sum / norm
        } else {
            // Degenerate case: return first point
            data.row(indices[0]).to_owned()
        }
    }

    /// Partition points by hemisphere relative to center
    fn partition_by_hemisphere(
        data: &Array2<f32>,
        indices: &[usize],
        center: &Array1<f32>,
    ) -> (Vec<usize>, Vec<usize>) {
        // Find the direction of maximum variance
        let axis = Self::find_split_axis(data, indices, center);
        
        let mut left = Vec::new();
        let mut right = Vec::new();
        
        for &idx in indices {
            let point = data.row(idx);
            if point.dot(&axis) >= 0.0 {
                left.push(idx);
            } else {
                right.push(idx);
            }
        }
        
        (left, right)
    }

    /// Find best axis to split on (direction of maximum variance)
    fn find_split_axis(data: &Array2<f32>, indices: &[usize], center: &Array1<f32>) -> Array1<f32> {
        // Simple heuristic: use the first principal component
        // For now, just use a random orthogonal direction
        let d = data.ncols();
        let mut axis = Array1::zeros(d);
        
        // Create a vector orthogonal to center
        axis[0] = -center[1];
        axis[1] = center[0];
        
        let norm = axis.dot(&axis).sqrt();
        if norm > 1e-10 {
            axis / norm
        } else {
            axis[0] = 1.0;
            axis
        }
    }

    /// Angular distance between two normalized vectors (in radians)
    fn angular_distance<S1, S2>(a: &ndarray::ArrayBase<S1, ndarray::Ix1>, b: &ndarray::ArrayBase<S2, ndarray::Ix1>) -> f32
    where
        S1: ndarray::Data<Elem = f32>,
        S2: ndarray::Data<Elem = f32>,
    {
        let dot = a.dot(b).clamp(-1.0, 1.0);
        dot.acos()
    }

    /// Get tree statistics
    pub fn stats(&self) -> TreeStats {
        let mut stats = TreeStats {
            num_nodes: 1,
            num_leaves: 0,
            max_depth: 0,
            total_points: 0,
        };
        
        self.compute_stats(0, &mut stats);
        stats
    }

    fn compute_stats(&self, depth: usize, stats: &mut TreeStats) {
        stats.max_depth = stats.max_depth.max(depth);
        
        if let Some(ref points) = self.points {
            stats.num_leaves += 1;
            stats.total_points += points.len();
        } else {
            if let Some(ref left) = self.left {
                stats.num_nodes += 1;
                left.compute_stats(depth + 1, stats);
            }
            if let Some(ref right) = self.right {
                stats.num_nodes += 1;
                right.compute_stats(depth + 1, stats);
            }
        }
    }
}

/// Tree statistics
#[derive(Debug, Clone)]
pub struct TreeStats {
    pub num_nodes: usize,
    pub num_leaves: usize,
    pub max_depth: usize,
    pub total_points: usize,
}

/// Index with spherical cap tree for range queries
pub struct SphericalCapIndex {
    data: Array2<f32>,
    tree: SphericalCapTree,
}

impl SphericalCapIndex {
    /// Build index from normalized vectors
    pub fn new(data: Array2<f32>) -> Self {
        // Use larger leaf size to prevent stack overflow on large datasets
        let tree = SphericalCapTree::build(&data, 1000);
        Self { data, tree }
    }

    /// Range query: find all points within angular distance θ
    pub fn range_query(&self, query: &Array1<f32>, theta: f32) -> Vec<usize> {
        // Get candidate indices from tree
        let candidates = self.tree.range_query(query, theta);
        
        // Filter to exact matches
        candidates.into_iter()
            .filter(|&idx| {
                let point = self.data.row(idx);
                let dist = SphericalCapTree::angular_distance(query, &point);
                dist <= theta
            })
            .collect()
    }

    /// Get tree statistics
    pub fn stats(&self) -> TreeStats {
        self.tree.stats()
    }
}
