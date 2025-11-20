use std::collections::HashSet;
use std::sync::Arc;

use lasso::Spur;

use crate::corpus::{ConceptExtractor, ConceptInterner, ExtractedConcept, Tokenizer};

/// Extracts concepts by tokenizing and interning them.
#[derive(Debug)]
pub struct InterningConceptExtractor<T: Tokenizer> {
    tokenizer: T,
    interner: Arc<ConceptInterner>,
    stopword_ids: HashSet<Spur>,
}

impl<T: Tokenizer> InterningConceptExtractor<T> {
    /// Create a new extractor.
    pub fn new(
        tokenizer: T,
        interner: Arc<ConceptInterner>,
        raw_stopwords: &[&str],
    ) -> Self {
        let mut stopword_ids = HashSet::new();
        for w in raw_stopwords {
            let id = interner.get_or_intern(w.to_lowercase());
            stopword_ids.insert(id);
        }

        Self {
            tokenizer,
            interner,
            stopword_ids,
        }
    }

    fn is_stopword(&self, spur: Spur) -> bool {
        self.stopword_ids.contains(&spur)
    }
}

impl<T: Tokenizer> ConceptExtractor for InterningConceptExtractor<T> {
    fn extract_concepts(&self, text: &str) -> Vec<ExtractedConcept> {
        let tokens = self.tokenizer.tokenize(text);
        let mut out = Vec::new();

        for tok in tokens {
            let term = tok.text.to_lowercase();
            if term.is_empty() {
                continue;
            }

            // Intern canonical string once
            let spur = self.interner.get_or_intern(&term);

            if self.is_stopword(spur) {
                continue;
            }

            out.push(ExtractedConcept {
                canonical_name: term,      // for debug; optional later
                concept_id: spur.into(),
            });
        }

        out
    }
}
