//! Tokenizer abstraction. You can plug in your own implementation.

/// A simple token representation (you'll likely expand this).
#[derive(Debug, Clone)]
pub struct Token {
    /// The text content of the token.
    pub text: String,
    /// The character position of the token in the source text.
    pub position: usize,
}

/// Trait for tokenization.
pub trait Tokenizer: Send + Sync {
    /// Tokenize a raw text document into a sequence of tokens.
    fn tokenize(&self, text: &str) -> Vec<Token>;
}

/// Very naive whitespace tokenizer for bootstrap.
#[derive(Debug)]
pub struct WhitespaceTokenizer;

impl Tokenizer for WhitespaceTokenizer {
    fn tokenize(&self, text: &str) -> Vec<Token> {
        text.split_whitespace()
            .enumerate()
            .map(|(i, t)| Token {
                text: t.to_string(),
                position: i,
            })
            .collect()
    }
}
