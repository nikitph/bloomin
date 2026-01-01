pub mod index;
pub mod evidence;
pub mod query;
pub mod storage;

pub type ItemId = usize;

#[derive(Clone, Debug)]
pub struct Query {
    pub vector: Vec<f32>,
    pub color_hint: Option<u8>, // 0..255 or enum
    pub aspect_hint: Option<f32>, // e.g. 1.0 = square
}

impl Query {
    pub fn new(vector: Vec<f32>) -> Self {
        Self { vector, color_hint: None, aspect_hint: None }
    }
}
