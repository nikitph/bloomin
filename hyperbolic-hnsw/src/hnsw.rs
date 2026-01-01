use crate::geometry::PoincareBall;
use ndarray::{Array1, ArrayView1};
use ordered_float::OrderedFloat;
use rand::Rng;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::cmp::Reverse;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Candidate {
    dist: OrderedFloat<f64>,
    idx: usize,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.cmp(&other.dist)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct HyperbolicHNSW {
    pub space: PoincareBall,
    pub dim: usize,
    pub m: usize,
    pub m_max: usize,
    pub m_max_0: usize,
    pub ef_construction: usize,
    pub ml: f64,
    
    pub data: Vec<Array1<f64>>,
    pub graph: Vec<Vec<Vec<usize>>>, // graph[node_idx][layer_idx] -> neighbors
    
    pub entry_point: Option<usize>,
    pub max_layer: i32,
}

impl HyperbolicHNSW {
    pub fn new(dim: usize, curvature: f64, m: usize, m_max: usize, m_max_0: usize, ef_construction: usize, ml: f64) -> Self {
        Self {
            space: PoincareBall::new(dim, curvature),
            dim,
            m,
            m_max,
            m_max_0,
            ef_construction,
            ml,
            data: Vec::new(),
            graph: Vec::new(),
            entry_point: None,
            max_layer: -1,
        }
    }

    pub fn insert(&mut self, point: Array1<f64>) {
        let point = self.space.project_to_ball(point);
        let level = self.get_random_level();
        let idx = self.data.len();
        
        self.data.push(point.clone());
        // Initialize layers for this node. Layers 0 to level.
        // graph[idx] will allow accessing graph[idx][l].
        let mut node_layers = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            node_layers.push(Vec::new());
        }
        self.graph.push(node_layers);

        if self.entry_point.is_none() {
            self.entry_point = Some(idx);
            self.max_layer = level as i32;
            return;
        }

        let entry_pt = self.entry_point.unwrap();
        let curr_max_layer = self.max_layer as usize;
        
        let mut curr_obj = entry_pt;
        // curr_dist is not strictly needed for the logic as we just want the closest node index

        // Phase 1: Zoom down to layer 'level'
        for l in (level + 1..=curr_max_layer).rev() {
             let res = self.search_layer(&point.view(), &[curr_obj], 1, l);
             if let Some(closest) = res.iter().min() {
                 curr_obj = closest.idx;
             }
        }

        // Phase 2: Insert from 'level' down to 0
        let mut ep_candidates = vec![curr_obj];
        
        for l in (0..=std::cmp::min(level, curr_max_layer)).rev() {
            let limit = if l == 0 { self.m_max_0 } else { self.m_max };
            
            // Search level to find candidates
            let neighbors = self.search_layer(&point.view(), &ep_candidates, self.ef_construction, l);
            
            // Select heuristics
            let selected_neighbors = self.get_neighbors_heuristic(&self.data[idx], neighbors, self.m);
            
            // Connect bidirectionally
            for &neighbor_idx in &selected_neighbors {
                self.connect(idx, neighbor_idx, l);
                self.connect(neighbor_idx, idx, l);
                
                // Prune neighbor if needed
                let neighbor_conn_len = self.graph[neighbor_idx][l].len();
                if neighbor_conn_len > limit {
                    self.prune_connections(neighbor_idx, limit, l);
                }
            }
            
            ep_candidates = selected_neighbors;
        }

        if level as i32 > self.max_layer {
            self.max_layer = level as i32;
            self.entry_point = Some(idx);
        }
    }

    fn search_layer(&self, query: &ArrayView1<f64>, entry_points: &[usize], ef: usize, layer: usize) -> Vec<Candidate> {
        let mut visited = HashSet::new();
        let mut candidates = BinaryHeap::new(); // Min-heap via Reverse
        let mut w = BinaryHeap::new(); // Max-heap: keeps closest ef (stores (dist, idx))

        for &ep in entry_points {
            if visited.insert(ep) {
                let dist = self.space.distance(query, &self.data[ep].view());
                candidates.push(Reverse(Candidate { dist: OrderedFloat(dist), idx: ep }));
                w.push(Candidate { dist: OrderedFloat(dist), idx: ep });
                if w.len() > ef {
                    w.pop();
                }
            }
        }

        while let Some(Reverse(c)) = candidates.pop() {
            let c_dist = c.dist;
            let c_idx = c.idx;
            
            if let Some(furthest) = w.peek() {
                if c_dist > furthest.dist {
                     break;
                }
            }
            
            if layer < self.graph[c_idx].len() {
                for &neighbor_idx in &self.graph[c_idx][layer] {
                    if visited.insert(neighbor_idx) {
                        let dist = self.space.distance(query, &self.data[neighbor_idx].view());
                        
                         let furthest_dist = w.peek().map(|c| c.dist).unwrap_or(OrderedFloat(f64::MAX));

                        if OrderedFloat(dist) < furthest_dist || w.len() < ef {
                            candidates.push(Reverse(Candidate { dist: OrderedFloat(dist), idx: neighbor_idx }));
                            w.push(Candidate { dist: OrderedFloat(dist), idx: neighbor_idx });
                            
                            if w.len() > ef {
                                w.pop();
                            }
                        }
                    }
                }
            }
        }

        w.into_vec()
    }
    
    fn get_neighbors_heuristic(&self, _pt_data: &Array1<f64>, mut candidates: Vec<Candidate>, m: usize) -> Vec<usize> {
        candidates.sort_by(|a, b| a.dist.cmp(&b.dist));
        
        let mut selected = Vec::new();
        
        for cand in &candidates {
            if selected.len() >= m {
                break;
            }
            
            let mut is_diverse = true;
            let cand_point = &self.data[cand.idx];
            let dist_to_query = cand.dist.into_inner();
            
            for &sel_idx in &selected {
                let start_v = cand_point.view();
                let sel_point: &Array1<f64> = &self.data[sel_idx];
                let end_v: ArrayView1<f64> = sel_point.view();
                let dist_to_selected = self.space.distance(&start_v, &end_v);
                if dist_to_selected < dist_to_query {
                    is_diverse = false;
                    break;
                }
            }
            
            if is_diverse {
                selected.push(cand.idx);
            }
        }
        
        // Fill up if needed
        if selected.len() < m {
            for cand in candidates {
                if selected.len() >= m { break; }
                if !selected.contains(&cand.idx) {
                    selected.push(cand.idx);
                }
            }
        }
        
        selected
    }

    fn connect(&mut self, idx: usize, neighbor: usize, layer: usize) {
        // self.graph[idx][layer].push(neighbor) but check duplicates?
        // HNSW implementations usually just push.
        if !self.graph[idx][layer].contains(&neighbor) {
            self.graph[idx][layer].push(neighbor);
        }
    }
    
    fn prune_connections(&mut self, idx: usize, max_conn: usize, layer: usize) {
        let neighbors = self.graph[idx][layer].clone();
        if neighbors.len() <= max_conn {
             return;
        }
        
        let mut candidates = Vec::new();
        let pt = &self.data[idx];
        for &n_idx in &neighbors {
             let dist = self.space.distance(&pt.view(), &self.data[n_idx].view());
             candidates.push(Candidate { dist: OrderedFloat(dist), idx: n_idx });
        }
        
        let selected = self.get_neighbors_heuristic(pt, candidates, max_conn);
        self.graph[idx][layer] = selected;
    }

    fn get_random_level(&self) -> usize {
        let mut rng = rand::thread_rng();
        let r: f64 = rng.gen();
        (-r.ln() * self.ml) as usize
    }

    pub fn search(&self, query: &Array1<f64>, k: usize, ef: usize) -> Vec<(usize, f64)> {
        let query = self.space.project_to_ball(query.clone());
        
        if self.entry_point.is_none() {
            return Vec::new();
        }
        
        let mut curr_obj = self.entry_point.unwrap();
        // let mut curr_dist = self.space.distance(&query.view(), &self.data[curr_obj].view());
        
        let max_l = self.max_layer as usize;
        
        // Zoom down
        for l in (1..=max_l).rev() {
             let res = self.search_layer(&query.view(), &[curr_obj], 1, l);
             if let Some(closest) = res.iter().min() {
                 curr_obj = closest.idx;
             }
        }
        
        // Last layer
        let mut res = self.search_layer(&query.view(), &[curr_obj], ef, 0);
        
        // Sort and take k
        res.sort(); // Sorts by distance ascending
        
        res.into_iter().take(k).map(|c| (c.idx, c.dist.into_inner())).collect()
    }
}
