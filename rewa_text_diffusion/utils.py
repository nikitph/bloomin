import torch
from torch.utils.data import Dataset

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
