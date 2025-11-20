//! Evaluation metrics for information retrieval.

use std::collections::HashSet;

/// Compute Recall@K: fraction of relevant docs in top-K results.
pub fn recall_at_k(retrieved: &[(String, f64)], relevant: &[String], k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }

    let relevant_set: HashSet<_> = relevant.iter().collect();
    let retrieved_k: HashSet<_> = retrieved.iter().take(k).map(|(id, _)| id).collect();
    
    let hits = relevant_set.intersection(&retrieved_k).count();
    hits as f64 / relevant.len() as f64
}

/// Compute NDCG@K: Normalized Discounted Cumulative Gain.
pub fn ndcg_at_k(retrieved: &[(String, f64)], relevant: &[String], k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }

    let relevant_set: HashSet<_> = relevant.iter().collect();
    
    // DCG: sum of relevance / log2(rank + 1)
    let dcg: f64 = retrieved
        .iter()
        .take(k)
        .enumerate()
        .map(|(rank, (doc_id, _))| {
            let relevance = if relevant_set.contains(&doc_id) { 1.0 } else { 0.0 };
            relevance / (rank as f64 + 2.0).log2() // rank+2 because rank is 0-indexed
        })
        .sum();

    // IDCG: ideal DCG (all relevant docs at top)
    let ideal_k = k.min(relevant.len());
    let idcg: f64 = (0..ideal_k)
        .map(|rank| 1.0 / (rank as f64 + 2.0).log2())
        .sum();

    if idcg == 0.0 {
        0.0
    } else {
        dcg / idcg
    }
}

/// Compute MRR: Mean Reciprocal Rank (reciprocal of first relevant doc rank).
pub fn mrr(retrieved: &[(String, f64)], relevant: &[String]) -> f64 {
    let relevant_set: HashSet<_> = relevant.iter().collect();
    
    for (rank, (doc_id, _)) in retrieved.iter().enumerate() {
        if relevant_set.contains(&doc_id) {
            return 1.0 / (rank as f64 + 1.0);
        }
    }
    
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_at_k() {
        let retrieved = vec![
            ("d1".to_string(), 1.0),
            ("d2".to_string(), 0.8),
            ("d3".to_string(), 0.6),
        ];
        let relevant = vec!["d1".to_string(), "d2".to_string()];
        
        assert_eq!(recall_at_k(&retrieved, &relevant, 2), 1.0);
        assert_eq!(recall_at_k(&retrieved, &relevant, 1), 0.5);
    }

    #[test]
    fn test_mrr() {
        let retrieved = vec![
            ("d1".to_string(), 1.0),
            ("d2".to_string(), 0.8),
            ("d3".to_string(), 0.6),
        ];
        let relevant = vec!["d2".to_string()];
        
        assert_eq!(mrr(&retrieved, &relevant), 0.5); // 1/2
    }
}
