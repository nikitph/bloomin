use crate::{ItemId, Query};

#[derive(Clone, Debug, serde::Deserialize)]
pub struct ItemMetadata {
    pub aspect_ratio: f32,
    pub color: u8,
    pub coarse_emb: Vec<f32>,
}

#[derive(serde::Deserialize)]
pub struct ItemStore {
    pub vectors: Vec<Vec<f32>>,
    pub metadata: Vec<ItemMetadata>,
}

impl ItemStore {
    pub fn new(vectors: Vec<Vec<f32>>, metadata: Vec<ItemMetadata>) -> Self {
        Self { vectors, metadata }
    }

    pub fn load_from_json(path: &str) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);
        
        // JSON structure from python script:
        // [ { "id": i, "vector": [], "metadata": {} }, ... ]
        #[derive(serde::Deserialize)]
        struct RawItem {
            vector: Vec<f32>,
            metadata: ItemMetadata,
        }
        
        let raw_items: Vec<RawItem> = serde_json::from_reader(reader)?;
        
        // Unzip
        let n = raw_items.len();
        let mut vectors = Vec::with_capacity(n);
        let mut metadata = Vec::with_capacity(n);
        
        for item in raw_items {
            vectors.push(item.vector);
            metadata.push(item.metadata);
        }
        
        Ok(Self::new(vectors, metadata))
    }

    pub fn get(&self, id: ItemId) -> &[f32] {
        &self.vectors[id]
    }
    
    pub fn len(&self) -> usize {
        self.vectors.len()
    }
    
    // Simple Euclidean distance for internal usage if needed, 
    // though evidence operators might implement their own logic.
    pub fn distance(&self, q: &Query, id: ItemId) -> f32 {
        let v = &self.vectors[id];
        v.iter().zip(q.vector.iter())
         .map(|(a, b)| (a - b).powi(2))
         .sum::<f32>()
         .sqrt()
    }
}
