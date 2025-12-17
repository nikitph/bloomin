//! Vector representation and operations for high-dimensional embeddings

use serde::{Deserialize, Serialize};
use std::ops::{Add, Sub, Mul};

/// A high-dimensional vector with its unique identifier
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Vector {
    pub id: u64,
    pub data: Vec<f32>,
}

impl Vector {
    /// Create a new vector with the given ID and data
    pub fn new(id: u64, data: Vec<f32>) -> Self {
        Self { id, data }
    }

    /// Create a zero vector of given dimension
    pub fn zeros(id: u64, dim: usize) -> Self {
        Self {
            id,
            data: vec![0.0; dim],
        }
    }

    /// Create a random vector (uniform in [-1, 1])
    pub fn random(id: u64, dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Self { id, data }
    }

    /// Dimension of the vector
    pub fn dim(&self) -> usize {
        self.data.len()
    }

    /// L2 norm (magnitude)
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    /// Normalize to unit length
    pub fn normalize(&self) -> Self {
        let n = self.norm();
        if n > 1e-10 {
            Self {
                id: self.id,
                data: self.data.iter().map(|x| x / n).collect(),
            }
        } else {
            self.clone()
        }
    }

    /// Dot product with another vector
    pub fn dot(&self, other: &Vector) -> f32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Cosine similarity with another vector
    pub fn cosine_similarity(&self, other: &Vector) -> f32 {
        let dot = self.dot(other);
        let norm_self = self.norm();
        let norm_other = other.norm();
        if norm_self > 1e-10 && norm_other > 1e-10 {
            dot / (norm_self * norm_other)
        } else {
            0.0
        }
    }

    /// Euclidean distance to another vector
    pub fn euclidean_distance(&self, other: &Vector) -> f32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Squared Euclidean distance (faster, no sqrt)
    pub fn squared_distance(&self, other: &Vector) -> f32 {
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum()
    }
}

impl Add for &Vector {
    type Output = Vector;

    fn add(self, other: &Vector) -> Vector {
        Vector {
            id: self.id,
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }
}

impl Sub for &Vector {
    type Output = Vector;

    fn sub(self, other: &Vector) -> Vector {
        Vector {
            id: self.id,
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a - b)
                .collect(),
        }
    }
}

impl Mul<f32> for &Vector {
    type Output = Vector;

    fn mul(self, scalar: f32) -> Vector {
        Vector {
            id: self.id,
            data: self.data.iter().map(|x| x * scalar).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let v1 = Vector::new(1, vec![1.0, 0.0, 0.0]);
        let v2 = Vector::new(2, vec![1.0, 0.0, 0.0]);
        assert!((v1.cosine_similarity(&v2) - 1.0).abs() < 1e-6);

        let v3 = Vector::new(3, vec![0.0, 1.0, 0.0]);
        assert!(v1.cosine_similarity(&v3).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let v = Vector::new(1, vec![3.0, 4.0]);
        let n = v.normalize();
        assert!((n.norm() - 1.0).abs() < 1e-6);
    }
}
