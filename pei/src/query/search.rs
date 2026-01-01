use std::collections::BinaryHeap;
use bit_set::BitSet;
use crate::{Query, ItemId};
use crate::index::pei_index::PEIIndex;
use crate::index::candidate::Candidate;

pub fn search(
    index: &PEIIndex,
    query: &Query,
    budget: f64,
    k: usize,
) -> Vec<Candidate> {
    let mut heap = BinaryHeap::new();
    let mut remaining_budget = budget;

    // Initialize candidates
    // In a real system, we might use an inverted index to get C0.
    // For now, we scan all items, creating initial candidates.
    for i in 0..index.store.len() {
        heap.push(Candidate::new(i, 0.0, 1.0));
    }

    let mut results = Vec::new();
    
    // Safety break to prevent infinite loops in mock
    let mut steps = 0;
    let max_steps = 100_000;

    while remaining_budget > 0.0 && !heap.is_empty() && steps < max_steps {
        steps += 1;
        
        let mut c = heap.pop().unwrap();

        // Stopping condition for a candidate: Low uncertainty
        if c.uncertainty < 0.01 {
            results.push(c);
            if results.len() >= k {
                break;
            }
            continue;
        }

        let available_ops = index.dag.next_ops(&c.applied);
        let best_op_idx = select_best_operator(&available_ops, &c, index);

        if let Some(op_idx) = best_op_idx {
            let op = &index.evidence_ops[op_idx];
            
            // Check budget
            if op.cost() > remaining_budget {
                // Can't afford this op, maybe push back or discard?
                // If we can't afford best op, we likely can't refine further 
                // effectively in this greedy path.
                // For simplified logic: emit as result if good enough, else drop?
                // Let's emit.
                results.push(c);
                continue;
            }

            let res = op.apply(query, c.item_id, &index.store);
            remaining_budget -= op.cost();

            c.belief += res.belief_delta;
            // Ensure uncertainty doesn't go below 0
            c.uncertainty = (c.uncertainty - res.uncertainty_delta).max(0.0);
            c.applied.insert(op_idx);

            heap.push(c);
        } else {
            // No more evidence available
            results.push(c);
        }
    }
    
    // Sort results by belief
    results.sort_by(|a, b| b.belief.partial_cmp(&a.belief).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    
    results
}

fn select_best_operator(ops: &[usize], _c: &Candidate, index: &PEIIndex) -> Option<usize> {
    let mut best_score = f64::NEG_INFINITY;
    let mut best_op = None;

    for &op_idx in ops {
        let op = &index.evidence_ops[op_idx];
        // Heuristic: Information Gain / Cost
        // For now, assume gain is proportional to uncertainty reduction potential.
        // Higher cost ops (Distance) reduce uncertainty by 1.0 (approx).
        // Lower cost ops (Centroid) reduce by ~0.4.
        
        // This should theoretically be dynamic based on 'c'.
        // Simplified: 
        let gain = if op.cost() >= 1.0 { 1.0 } else { 0.4 }; 
        
        let score = gain / op.cost();
        
        if score > best_score {
            best_score = score;
            best_op = Some(op_idx);
        }
    }

    best_op
}
