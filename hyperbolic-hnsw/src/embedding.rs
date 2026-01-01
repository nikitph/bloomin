use crate::geometry::PoincareBall;
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::f64::consts::PI;

pub struct HyperbolicEmbedder {
    pub dim: usize,
    pub space: PoincareBall,
}

pub struct TreeNode {
    pub id: String,
    pub children: Vec<TreeNode>,
}

impl HyperbolicEmbedder {
    pub fn new(dim: usize, c: f64) -> Self {
        Self {
            dim,
            space: PoincareBall::new(dim, c),
        }
    }

    /// Embed Euclidean vectors into hyperbolic space using stereographic projection
    /// π(x) = x / (1 + √(1 + ||x||²))
    pub fn embed_euclidean(&self, vectors: &Array2<f64>) -> Vec<Array1<f64>> {
        let (n, d) = vectors.dim();
        // Ideally we would do PCA here if d > self.dim, but skipping for MVP
        assert!(d <= self.dim, "Input dimension {} > embedding dimension {}. PCA not implemented.", d, self.dim);

        let mut embedded = Vec::with_capacity(n);

        for i in 0..n {
            let vec = vectors.row(i);
            // If d < self.dim, pad with zeros? Or just use first d dims?
            // Let's assume we map to first d coords and rest are 0.
            let mut point = Array1::zeros(self.dim);
            for j in 0..d {
                point[j] = vec[j];
            }

            let norm_sq = point.dot(&point);
            let denominator = 1.0 + (1.0 + norm_sq).sqrt();
            
            let projected = point / denominator;
            
            // Project to ball for numerical stability
            embedded.push(self.space.project_to_ball(projected));
        }
        
        embedded
    }

    /// Embed tree into hyperbolic space (Sarkar's algorithm)
    pub fn embed_tree(&self, root: &TreeNode, tau: f64) -> HashMap<String, Array1<f64>> {
        let mut embeddings = HashMap::new();
        
        // Root at origin
        let root_coords = Array1::zeros(self.dim);
        embeddings.insert(root.id.clone(), root_coords.clone());
        
        self.embed_recursive(root, &root_coords, 1, tau, &mut embeddings);
        
        embeddings
    }

    fn embed_recursive(
        &self, 
        node: &TreeNode, 
        parent_coords: &Array1<f64>, 
        depth: usize, 
        tau: f64, 
        embeddings: &mut HashMap<String, Array1<f64>>
    ) {
        if node.children.is_empty() {
            return;
        }

        let angular_step = 2.0 * PI / (node.children.len() as f64);
        
        // Radial distance: r = (1/√c) * tanh⁻¹(tanh(τ*depth))
        // Wait, pseudocode: r = tanh(tau * depth) / sqrt(c)
        // Checks out approximately.
        let r = (tau * (depth as f64)).tanh() / self.space.c.sqrt();
        
        for (i, child) in node.children.iter().enumerate() {
            let theta = (i as f64) * angular_step;
            
            let mut direction = Array1::zeros(self.dim);
            direction[0] = theta.cos();
            if self.dim > 1 {
                direction[1] = theta.sin();
            }
            
            // Move from parent via exponential map.
            // We need a direction tangent vector.
            // The direction computed above is in the tangent space of the Origin?
            // Sarkar's algorithm usually constructs relative to parent.
            // The pseudocode uses `exp_map(parent_coords, r * direction)`. 
            // This assumes `direction` is in T_parent.
            // `direction` constructed with simple sin/cos is valid tangent vector if normalized.
            // But we need to be careful about orientation.
            // For MVP, randomly rotating or using fixed frame is 'okay' for simple trees, 
            // but effectively we are parallel transporting a frame from root or just defining local frame.
            // Given the pseudocode does `direction[0] = cos...`, it defines a local frame.
            
            let v = &direction * r;
            let child_coords = self.space.exp_map(&parent_coords.view(), &v.view());
            
            embeddings.insert(child.id.clone(), child_coords.clone());
            
            self.embed_recursive(child, &child_coords, depth + 1, tau, embeddings);
        }
    }
}
