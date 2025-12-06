"""
Model Configurations
====================

Defines all models to test with their metadata.
"""

MODELS_TO_TEST = {
    # ========== HISTORICAL (Pre-Transformer) ==========
    # Skipped - require large downloads from gensim
    # 'word2vec': {
    #     'source': 'gensim',
    #     'model': 'word2vec-google-news-300',
    #     'dim': 300,
    #     'year': 2013,
    #     'arch': 'CBOW',
    #     'n_samples': 100000,
    #     'domain': 'Words',
    #     'corpus_type': 'words'
    # },
    # 
    # 'glove': {
    #     'source': 'gensim',
    #     'model': 'glove-wiki-gigaword-300',
    #     'dim': 300,
    #     'year': 2014,
    #     'arch': 'Matrix',
    #     'n_samples': 100000,
    #     'domain': 'Words',
    #     'corpus_type': 'words'
    # },
    # 
    # 'fasttext': {
    #     'source': 'gensim',
    #     'model': 'fasttext-wiki-news-subwords-300',
    #     'dim': 300,
    #     'year': 2016,
    #     'arch': 'CBOW+char',
    #     'n_samples': 100000,
    #     'domain': 'Words',
    #     'corpus_type': 'words'
    # },
    
    # ========== BERT FAMILY ==========
    'bert-base': {
        'source': 'transformers',
        'model': 'bert-base-uncased',
        'dim': 768,
        'year': 2018,
        'arch': 'Trans-Enc',
        'n_samples': 10000,  # Reduced for faster testing
        'domain': 'Context',
        'corpus_type': 'sentences'
    },
    
    # 'bert-large': {  # Skipped - large download (1.34GB)
    #     'source': 'transformers',
    #     'model': 'bert-large-uncased',
    #     'dim': 1024,
    #     'year': 2018,
    #     'arch': 'Trans-Enc',
    #     'n_samples': 10000,
    #     'domain': 'Context',
    #     'corpus_type': 'sentences'
    # },
    
    'roberta': {
        'source': 'transformers',
        'model': 'roberta-base',
        'dim': 768,
        'year': 2019,
        'arch': 'Trans-Enc',
        'n_samples': 10000,
        'domain': 'Context',
        'corpus_type': 'sentences'
    },
    
    'distilbert': {
        'source': 'transformers',
        'model': 'distilbert-base-uncased',
        'dim': 768,
        'year': 2019,
        'arch': 'Trans-Enc',
        'n_samples': 10000,
        'domain': 'Context',
        'corpus_type': 'sentences'
    },
    
    # ========== SENTENCE ENCODERS ==========
    'sentence-bert': {
        'source': 'sentence-transformers',
        'model': 'all-MiniLM-L6-v2',
        'dim': 384,
        'year': 2019,
        'arch': 'Trans-Enc',
        'n_samples': 10000,
        'domain': 'Sentence',
        'corpus_type': 'sentences'
    },
    
    # ========== GENERATIVE ==========
    'gpt2': {
        'source': 'transformers',
        'model': 'gpt2',
        'dim': 768,
        'year': 2019,
        'arch': 'Trans-Dec',
        'n_samples': 10000,
        'domain': 'Context',
        'corpus_type': 'sentences'
    },
    
    # ========== DOMAIN-SPECIFIC ==========
    'codebert': {
        'source': 'transformers',
        'model': 'microsoft/codebert-base',
        'dim': 768,
        'year': 2020,
        'arch': 'Trans-Enc',
        'n_samples': 5000,
        'domain': 'Code',
        'corpus_type': 'code'
    },
    
    'scibert': {
        'source': 'transformers',
        'model': 'allenai/scibert_scivocab_uncased',
        'dim': 768,
        'year': 2019,
        'arch': 'Trans-Enc',
        'n_samples': 5000,
        'domain': 'Science',
        'corpus_type': 'science'
    },
}


# Quick test subset (for development)
QUICK_TEST_MODELS = {
    'sentence-bert': MODELS_TO_TEST['sentence-bert'],
    'bert-base': MODELS_TO_TEST['bert-base'],
}
