"""
Experiment 1: Scale Test on Real Language Modeling
Critical test: Does observer advantage hold at realistic scale?
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import time
from rssa_v3 import RSSA
import os

class ScaleExperiment:
    """Test RSSA scaling on real language modeling"""
    
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load dataset
        print(f"Loading dataset {config['data']}...")
        if config['data'] == 'wikitext-2':
            dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
        elif config['data'] == 'wikitext-103':
            dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')
        
        self.train_data = dataset['train']
        self.val_data = dataset['validation']
        
        # Tokenizer (simple word-level)
        self.vocab = self._build_vocab(self.train_data)
        self.vocab_size = len(self.vocab)
        print(f"Vocab size: {self.vocab_size}")
        
        # Models
        # 1. RSSA with Observer
        self.model_rssa = RSSA(
            vocab_size=self.vocab_size,
            d_model=config['d_model'],
            d_observer=config['d_observer'],
            n_layers=config['n_layers'],
            processor_window=512, # Force observer for context > 512
            observer_tau=0.95,
            chunk_size=64
        ).to(self.device)
        
        # 2. Baseline Transformer (Window = Inf)
        self.model_baseline = RSSA(
            vocab_size=self.vocab_size,
            d_model=config['d_model'],
            d_observer=config['d_observer'],
            n_layers=config['n_layers'],
            processor_window=99999, # Global attention
            observer_tau=0.95,
            chunk_size=99999 # Single chunk
        ).to(self.device)
        
        print(f"RSSA parameters: {sum(p.numel() for p in self.model_rssa.parameters()):,}")
        print(f"Baseline parameters: {sum(p.numel() for p in self.model_baseline.parameters()):,}")
    
    def _build_vocab(self, dataset, max_size=10000):
        """Build vocabulary from dataset"""
        from collections import Counter
        word_counts = Counter()
        
        # Scan subset for speed if large
        scan_limit = 50000 if len(dataset) > 50000 else len(dataset)
        for i in range(scan_limit):
            words = dataset[i]['text'].lower().split()
            word_counts.update(words)
        
        vocab = {'<pad>': 0, '<unk>': 1, '<eos>': 2}
        for word, _ in word_counts.most_common(max_size - 3):
            if word not in vocab:
                vocab[word] = len(vocab)
        
        return vocab
    
    def tokenize(self, text):
        """Convert text to token IDs"""
        words = text.lower().split()
        tokens = [self.vocab.get(w, self.vocab['<unk>']) for w in words]
        if not tokens: # Handle empty lines
            tokens = [self.vocab['<eos>']]
        return tokens
    
    def train_epoch(self, model, optimizer, max_steps=100):
        """Train for limited steps"""
        model.train()
        total_loss = 0
        total_tokens = 0
        start_time = time.time()
        
        step_count = 0
        for example in self.train_data:
            if step_count >= max_steps:
                break
            
            tokens = self.tokenize(example['text'])
            if len(tokens) < 16: # Skip very short snippets
                continue
                
            # Truncate if too long for simple implementation
            tokens = tokens[:1024]
            
            input_ids = torch.tensor([tokens[:-1]], device=self.device)
            target_ids = torch.tensor([tokens[1:]], device=self.device)
            
            # Forward
            logits, _ = model(input_ids, reset_observer=True)
            
            # Loss
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_ids.reshape(-1),
                ignore_index=self.vocab['<pad>']
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * len(tokens)
            total_tokens += len(tokens)
            step_count += 1
            
            if step_count % 10 == 0:
                avg = total_loss / total_tokens
                print(f"  Step {step_count:3d} | Loss: {avg:.4f} | Tokens/s: {total_tokens/(time.time()-start_time):.0f}")
                
        return total_loss / total_tokens if total_tokens > 0 else 0
    
    def evaluate(self, model, max_examples=100):
        """Evaluate perplexity"""
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            count = 0
            for example in self.val_data:
                if count >= max_examples:
                    break
                
                tokens = self.tokenize(example['text'])
                if len(tokens) < 16:
                    continue
                
                tokens = tokens[:1024]
                input_ids = torch.tensor([tokens[:-1]], device=self.device)
                target_ids = torch.tensor([tokens[1:]], device=self.device)
                
                logits, _ = model(input_ids, reset_observer=True)
                
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                    ignore_index=self.vocab['<pad>']
                )
                
                total_loss += loss.item() * len(tokens)
                total_tokens += len(tokens)
                count += 1
                
        if total_tokens == 0: return float('inf')
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        return perplexity

    def run(self, n_epochs=1, steps_per_epoch=100):
        print(f"\n--- Starting Experiment: {self.config['name']} ---")
        
        opt_rssa = torch.optim.AdamW(self.model_rssa.parameters(), lr=1e-4)
        opt_base = torch.optim.AdamW(self.model_baseline.parameters(), lr=1e-4)
        
        for epoch in range(n_epochs):
            print(f"\nEpoch {epoch+1}")
            
            print("Training RSSA...")
            loss_rssa = self.train_epoch(self.model_rssa, opt_rssa, steps_per_epoch)
            
            print("Training Baseline...")
            loss_base = self.train_epoch(self.model_baseline, opt_base, steps_per_epoch)
            
            print("Evaluating...")
            ppl_rssa = self.evaluate(self.model_rssa)
            ppl_base = self.evaluate(self.model_baseline)
            
            print(f"\nEpoch {epoch+1} Results:")
            print(f"  RSSA Loss: {loss_rssa:.4f} | PPL: {ppl_rssa:.2f}")
            print(f"  Base Loss: {loss_base:.4f} | PPL: {ppl_base:.2f}")
            print(f"  Advantage: {(ppl_base - ppl_rssa)/ppl_base*100:.1f}%")

import math

if __name__ == "__main__":
    config = {
        'name': 'small_test',
        'data': 'wikitext-2',
        'd_model': 256,
        'd_observer': 64,
        'n_layers': 4
    }
    
    exp = ScaleExperiment(config)
    exp.run(n_epochs=1, steps_per_epoch=50) # Very quick test
