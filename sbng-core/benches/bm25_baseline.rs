//! BM25 baseline using tantivy.

use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::*;
use tantivy::{doc, Index, IndexWriter, ReloadPolicy, TantivyDocument};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub text: String,
}

pub struct BM25Index {
    index: Index,
    id_field: Field,
    text_field: Field,
}

impl BM25Index {
    /// Create a new BM25 index from a JSONL corpus.
    pub fn build_from_jsonl(corpus_path: &str) -> anyhow::Result<Self> {
        // Define schema
        let mut schema_builder = Schema::builder();
        let id_field = schema_builder.add_text_field("id", STRING | STORED);
        let text_field = schema_builder.add_text_field("text", TEXT | STORED);
        let schema = schema_builder.build();

        // Create index in memory
        let index = Index::create_in_ram(schema);
        let mut index_writer: IndexWriter = index.writer(50_000_000)?;

        // Read and index documents
        let file = std::fs::File::open(corpus_path)?;
        let reader = std::io::BufReader::new(file);
        
        for line in std::io::BufRead::lines(reader) {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let doc: Document = serde_json::from_str(&line)?;
            index_writer.add_document(doc!(
                id_field => doc.id,
                text_field => doc.text,
            ))?;
        }

        index_writer.commit()?;

        Ok(Self {
            index,
            id_field,
            text_field,
        })
    }

    /// Search for documents matching the query.
    /// Returns (doc_id, score) pairs sorted by descending score.
    pub fn search(&self, query: &str, k: usize) -> anyhow::Result<Vec<(String, f64)>> {
        let reader = self.index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        let searcher = reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.text_field]);
        let query = query_parser.parse_query(query)?;

        let top_docs = searcher.search(&query, &TopDocs::with_limit(k))?;

        let mut results = Vec::new();
        for (score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
            let doc_id = retrieved_doc
                .get_first(self.id_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            results.push((doc_id, score as f64));
        }

        Ok(results)
    }
}
