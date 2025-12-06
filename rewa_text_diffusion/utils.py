import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

class BPETokenizer:
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        
        # Special tokens
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.bos_token = "[BOS]"
        self.eos_token = "[EOS]"
        
        self.special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3
        
    def train_from_texts(self, texts):
        """Train BPE tokenizer on a list of texts"""
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            show_progress=True
        )
        
        # Train on the texts
        self.tokenizer.train_from_iterator(texts, trainer)
        
        # Add post-processing to add BOS/EOS tokens
        self.tokenizer.post_processor = TemplateProcessing(
            single=f"{self.bos_token} $A {self.eos_token}",
            special_tokens=[
                (self.bos_token, self.bos_token_id),
                (self.eos_token, self.eos_token_id),
            ],
        )
        
        # Enable padding
        self.tokenizer.enable_padding(pad_id=self.pad_token_id, pad_token=self.pad_token)
        
    def encode(self, text, max_len=None):
        """Encode text to token IDs"""
        if max_len:
            self.tokenizer.enable_truncation(max_length=max_len)
        
        encoding = self.tokenizer.encode(text)
        tokens = encoding.ids
        
        if max_len and len(tokens) < max_len:
            tokens += [self.pad_token_id] * (max_len - len(tokens))
        
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, tokens, skip_special_tokens=True):
        """Decode token IDs to text"""
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        
        # Filter out padding tokens
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in [self.pad_token_id, self.bos_token_id, self.eos_token_id]]
        
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    
    def save(self, path):
        """Save tokenizer to file"""
        self.tokenizer.save(path)
    
    def load(self, path):
        """Load tokenizer from file"""
        self.tokenizer = Tokenizer.from_file(path)

# Keep SimpleTokenizer for backward compatibility
class SimpleTokenizer:
    def __init__(self, text_data=None):
        # Character level tokenizer for simplicity
        self.chars = sorted(list(set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-_\n")))
        self.vocab_size = len(self.chars) + 2 # +1 for padding, +1 for unknown
        self.char_to_idx = {ch: i+1 for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i+1: ch for i, ch in enumerate(self.chars)}
        self.pad_token_id = 0
        self.unk_token_id = len(self.chars) + 1
        
    def encode(self, text, max_len=None):
        tokens = [self.char_to_idx.get(c, self.unk_token_id) for c in text]
        if max_len:
            if len(tokens) < max_len:
                tokens += [self.pad_token_id] * (max_len - len(tokens))
            else:
                tokens = tokens[:max_len]
        return torch.tensor(tokens, dtype=torch.long)
    
    def decode(self, tokens):
        # tokens: list or tensor
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return "".join([self.idx_to_char.get(t, "") for t in tokens if t != self.pad_token_id])

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=32):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        return self.tokenizer.encode(text, self.max_len)
