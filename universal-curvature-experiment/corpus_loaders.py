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
    Load sentences from Wikipedia/BookCorpus.
    
    For BERT, RoBERTa, GPT-2, etc. (contextual embeddings).
    """
    print(f"  Loading {n_samples} sentences...")
    
    # Generate diverse sample sentences
    templates = [
        "The latest research in {} shows promising results.",
        "Understanding {} is crucial for {}.",
        "Recent developments in {} have transformed {}.",
        "Experts in {} are studying {} extensively.",
        "The impact of {} on {} cannot be overstated.",
        "Advances in {} enable new applications in {}.",
        "The field of {} has grown significantly in recent years.",
        "Researchers are exploring the relationship between {} and {}.",
        "New techniques in {} improve performance on {}.",
        "The integration of {} with {} creates new opportunities.",
    ]
    
    topics = [
        'machine learning', 'artificial intelligence', 'deep learning',
        'natural language processing', 'computer vision', 'robotics',
        'quantum computing', 'blockchain', 'cybersecurity', 'cloud computing',
        'data science', 'big data', 'neural networks', 'reinforcement learning',
        'transfer learning', 'generative models', 'transformers', 'attention mechanisms',
        'optimization', 'gradient descent', 'backpropagation', 'convolutional networks',
        'recurrent networks', 'graph neural networks', 'meta-learning', 'few-shot learning'
    ]
    
    sentences = []
    for i in range(n_samples):
        template = templates[i % len(templates)]
        topic1 = topics[i % len(topics)]
        topic2 = topics[(i + 1) % len(topics)]
        
        if '{}' in template:
            # Count placeholders
            n_placeholders = template.count('{}')
            if n_placeholders == 1:
                sentence = template.format(topic1)
            elif n_placeholders == 2:
                sentence = template.format(topic1, topic2)
            else:
                sentence = template
        else:
            sentence = template
        
        sentences.append(sentence)
    
    return sentences


def load_code(n_samples: int = 5000) -> List[str]:
    """
    Load Python code snippets.
    
    For CodeBERT (code embeddings).
    """
    print(f"  Loading {n_samples} code snippets...")
    
    # Generate sample Python code
    code_templates = [
        "def {}(x):\n    return x * 2",
        "class {}:\n    def __init__(self):\n        self.value = 0",
        "for i in range({}):\n    print(i)",
        "if {} > 0:\n    result = {}\nelse:\n    result = 0",
        "import {}\n\ndef main():\n    pass",
        "try:\n    {}\nexcept Exception as e:\n    print(e)",
        "with open('{}', 'r') as f:\n    data = f.read()",
        "[{} for i in range(10)]",
        "lambda x: x + {}",
        "{'key': {}, 'value': {}}",
    ]
    
    code_snippets = []
    for i in range(n_samples):
        code = code_templates[i % len(code_templates)]
        code_snippets.append(code)
    
    return code_snippets


def load_science(n_samples: int = 5000) -> List[str]:
    """
    Load scientific abstracts.
    
    For SciBERT (scientific text embeddings).
    """
    print(f"  Loading {n_samples} scientific abstracts...")
    
    # Generate sample scientific text
    templates = [
        "We present a novel approach to {} using {}. Our method achieves state-of-the-art results on {}.",
        "In this paper, we investigate the relationship between {} and {}. We find that {} significantly impacts {}.",
        "Recent advances in {} have enabled new applications in {}. We demonstrate this through experiments on {}.",
        "We propose a new framework for {} that combines {} with {}. Experimental results show improvements over baseline methods.",
        "This study examines the effects of {} on {}. Our findings suggest that {} plays a crucial role in {}.",
        "We introduce a {} model for {}. The model is trained on {} and evaluated on {}.",
        "Our work addresses the challenge of {} in {}. We develop a {} approach that outperforms existing methods.",
        "We analyze the performance of {} across different {}. Results indicate that {} is most effective for {}.",
    ]
    
    topics = [
        'protein folding', 'gene expression', 'climate modeling', 'particle physics',
        'quantum mechanics', 'molecular dynamics', 'drug discovery', 'genome sequencing',
        'neural plasticity', 'evolutionary biology', 'astrophysics', 'materials science',
        'chemical synthesis', 'renewable energy', 'nanotechnology', 'bioinformatics'
    ]
    
    methods = [
        'machine learning', 'statistical analysis', 'computational modeling',
        'experimental validation', 'theoretical framework', 'numerical simulation'
    ]
    
    abstracts = []
    for i in range(n_samples):
        template = templates[i % len(templates)]
        topic1 = topics[i % len(topics)]
        topic2 = topics[(i + 1) % len(topics)]
        method = methods[i % len(methods)]
        
        n_placeholders = template.count('{}')
        if n_placeholders == 2:
            abstract = template.format(topic1, method)
        elif n_placeholders == 3:
            abstract = template.format(topic1, topic2, method)
        elif n_placeholders == 4:
            abstract = template.format(topic1, topic2, method, topic1)
        else:
            abstract = template
        
        abstracts.append(abstract)
    
    return abstracts


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
