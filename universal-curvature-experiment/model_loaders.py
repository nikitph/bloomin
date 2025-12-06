"""
Model Loaders
=============

Unified interface for loading and extracting embeddings from all model types.
"""

import numpy as np
import torch
from typing import List, Tuple, Union
from tqdm import tqdm


class ModelLoader:
    """Base class for model loaders."""
    
    def load_model(self, model_name: str):
        """Load the model."""
        raise NotImplementedError
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract embeddings from texts."""
        raise NotImplementedError


class GensimLoader(ModelLoader):
    """Loader for gensim models (Word2Vec, GloVe, FastText)."""
    
    def __init__(self):
        self.model = None
    
    def load_model(self, model_name: str):
        """Load gensim model."""
        import gensim.downloader as api
        
        print(f"  Loading gensim model: {model_name}")
        self.model = api.load(model_name)
        print(f"  ✓ Loaded {len(self.model)} word vectors")
        
        return self.model
    
    def get_embeddings(self, words: List[str]) -> np.ndarray:
        """Get embeddings for words."""
        embeddings = []
        
        for word in tqdm(words, desc="  Extracting embeddings"):
            if word in self.model:
                embeddings.append(self.model[word])
            # Skip words not in vocabulary
        
        return np.array(embeddings)


class TransformerLoader(ModelLoader):
    """Loader for HuggingFace transformer models."""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                   'mps' if torch.backends.mps.is_available() else 'cpu')
    
    def load_model(self, model_name: str):
        """Load transformer model."""
        from transformers import AutoModel, AutoTokenizer
        
        print(f"  Loading transformer model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Fix for GPT-2 and models without pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"  ✓ Loaded on device: {self.device}")
        
        return self.model, self.tokenizer
    
    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Get embeddings for texts using [CLS] token."""
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="  Extracting embeddings"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token (first token)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)


class SentenceTransformerLoader(ModelLoader):
    """Loader for sentence-transformers models."""
    
    def __init__(self):
        self.model = None
    
    def load_model(self, model_name: str):
        """Load sentence-transformer model."""
        from sentence_transformers import SentenceTransformer
        
        print(f"  Loading sentence-transformer: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"  ✓ Loaded")
        
        return self.model
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts."""
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings


def load_model_and_get_embeddings(
    model_config: dict,
    texts: List[str]
) -> np.ndarray:
    """
    Load model and extract embeddings.
    
    Args:
        model_config: Model configuration dict
        texts: List of texts to encode
        
    Returns:
        Embeddings array of shape (N, D)
    """
    source = model_config['source']
    model_name = model_config['model']
    
    if source == 'gensim':
        loader = GensimLoader()
        loader.load_model(model_name)
        embeddings = loader.get_embeddings(texts)
        
    elif source == 'transformers':
        loader = TransformerLoader()
        loader.load_model(model_name)
        embeddings = loader.get_embeddings(texts)
        
    elif source == 'sentence-transformers':
        loader = SentenceTransformerLoader()
        loader.load_model(model_name)
        embeddings = loader.get_embeddings(texts)
        
    else:
        raise ValueError(f"Unknown source: {source}")
    
    return embeddings
