"""
Corpus Loaders
==============

Load domain-specific corpora for different model types.
"""

import numpy as np
from typing import List


def load_words(n_samples: int = 100000) -> List[str]:
    """
    Load most common English words.
    
    For Word2Vec, GloVe, FastText (static embeddings).
    """
    print(f"  Loading {n_samples} common English words...")
    
    # Use a simple word list (can be replaced with actual corpus)
    # For now, generate sample words
    words = []
    
    # Common English words (simplified for demo)
    base_words = [
        'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'I',
        'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
        'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
        'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what',
        'machine', 'learning', 'computer', 'science', 'data', 'algorithm', 'neural',
        'network', 'deep', 'artificial', 'intelligence', 'model', 'training', 'test',
        'python', 'code', 'function', 'class', 'object', 'method', 'variable',
        'research', 'paper', 'study', 'analysis', 'experiment', 'result', 'conclusion',
        'technology', 'software', 'hardware', 'system', 'application', 'program',
        'database', 'server', 'client', 'network', 'internet', 'web', 'cloud',
    ]
    
    # Extend with variations
    for i in range(n_samples):
        if i < len(base_words):
            words.append(base_words[i])
        else:
            # Cycle through base words
            words.append(base_words[i % len(base_words)])
    
    return words[:n_samples]


def load_sentences(n_samples: int = 10000) -> List[str]:
    """
    Load real sentences from Wikipedia.
    
    For BERT, RoBERTa, GPT-2, etc. (contextual embeddings).
    """
    print(f"  Loading {n_samples} sentences from Wikipedia...")
    
    try:
        from datasets import load_dataset
        
        # Load Wikipedia dataset with streaming
        dataset = load_dataset(
            'wikimedia/wikipedia',
            '20231101.en',
            split='train',
            streaming=True
        )
        
        sentences = []
        for i, article in enumerate(dataset):
            if len(sentences) >= n_samples:
                break
            
            # Extract sentences from article text
            text = article['text']
            if text and len(text) > 50:
                # Split into sentences (simple split on periods)
                article_sentences = [s.strip() + '.' for s in text.split('.') if len(s.strip()) > 20]
                sentences.extend(article_sentences[:5])  # Take first 5 sentences per article
        
        print(f"  ✓ Loaded {len(sentences[:n_samples])} real Wikipedia sentences")
        return sentences[:n_samples]
        
    except Exception as e:
        print(f"  ⚠️  Could not load Wikipedia dataset: {e}")
        print(f"  Falling back to Google word list...")
        
        # Fallback: Use Google common words
        try:
            import urllib.request
            url = "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english.txt"
            with urllib.request.urlopen(url) as response:
                text = response.read().decode('utf-8')
            words = [l.strip() for l in text.splitlines() if len(l) > 3]
            
            # Make sentences from words
            sentences = [f"The word {w} is commonly used in English." for w in words]
            return sentences[:n_samples]
        except Exception as e2:
            print(f"  ⚠️  Could not load word list: {e2}")
            print(f"  Using minimal fallback data...")
            
            # Last resort fallback
            return [f"This is sample sentence number {i}." for i in range(n_samples)]



def load_code(n_samples: int = 5000) -> List[str]:
    """
    Load real Python code snippets from GitHub.
    
    For CodeBERT (code embeddings).
    """
    print(f"  Loading {n_samples} code snippets...")
    
    try:
        from datasets import load_dataset
        
        # Load code dataset (using CodeSearchNet Python)
        dataset = load_dataset('code_search_net', 'python', split='train', streaming=True)
        
        code_snippets = []
        for i, example in enumerate(dataset):
            if len(code_snippets) >= n_samples:
                break
            
            code = example.get('func_code_string', '') or example.get('whole_func_string', '')
            if code and len(code) > 50:
                code_snippets.append(code)
        
        print(f"  ✓ Loaded {len(code_snippets)} real Python code snippets")
        return code_snippets
        
    except Exception as e:
        print(f"  ⚠️  Could not load code dataset: {e}")
        print(f"  Using synthetic code fallback...")
        
        # Fallback to synthetic code
        code_templates = [
            "def process_data(x):\n    return x * 2",
            "class DataProcessor:\n    def __init__(self):\n        self.value = 0",
            "for i in range(100):\n    print(i)",
            "if value > 0:\n    result = value\nelse:\n    result = 0",
            "import numpy as np\n\ndef main():\n    pass",
            "try:\n    process()\nexcept Exception as e:\n    print(e)",
            "with open('data.txt', 'r') as f:\n    data = f.read()",
            "[x**2 for x in range(10)]",
            "lambda x: x + 1",
            "{'key': 'value', 'data': [1, 2, 3]}",
        ]
        
        return [code_templates[i % len(code_templates)] for i in range(n_samples)]



def load_science(n_samples: int = 5000) -> List[str]:
    """
    Load real scientific abstracts from PubMed/arXiv.
    
    For SciBERT (scientific text embeddings).
    """
    print(f"  Loading {n_samples} scientific abstracts...")
    
    try:
        from datasets import load_dataset
        
        # Try to load scientific papers dataset
        dataset = load_dataset('scientific_papers', 'pubmed', split='train', streaming=True)
        
        abstracts = []
        for i, paper in enumerate(dataset):
            if len(abstracts) >= n_samples:
                break
            
            abstract = paper.get('abstract', '')
            if abstract and len(abstract) > 100:
                abstracts.append(abstract)
        
        print(f"  ✓ Loaded {len(abstracts)} real scientific abstracts")
        return abstracts
        
    except Exception as e:
        print(f"  ⚠️  Could not load scientific papers dataset: {e}")
        print(f"  Using synthetic science fallback...")
        
        # Fallback to synthetic scientific text
        templates = [
            "We present a novel approach to protein folding using machine learning. Our method achieves state-of-the-art results on benchmark datasets.",
            "In this paper, we investigate the relationship between gene expression and climate modeling. We find that computational modeling significantly impacts experimental validation.",
            "Recent advances in quantum mechanics have enabled new applications in drug discovery. We demonstrate this through experiments on molecular dynamics.",
            "We propose a new framework for neural plasticity that combines statistical analysis with theoretical framework. Experimental results show improvements over baseline methods.",
            "This study examines the effects of astrophysics on materials science. Our findings suggest that numerical simulation plays a crucial role in chemical synthesis.",
        ]
        
        return [templates[i % len(templates)] for i in range(n_samples)]



def load_corpus(corpus_type: str, n_samples: int) -> List[str]:
    """
    Load corpus based on type.
    
    Args:
        corpus_type: 'words', 'sentences', 'code', or 'science'
        n_samples: Number of samples to load
        
    Returns:
        List of text samples
    """
    if corpus_type == 'words':
        return load_words(n_samples)
    elif corpus_type == 'sentences':
        return load_sentences(n_samples)
    elif corpus_type == 'code':
        return load_code(n_samples)
    elif corpus_type == 'science':
        return load_science(n_samples)
    else:
        raise ValueError(f"Unknown corpus type: {corpus_type}")
