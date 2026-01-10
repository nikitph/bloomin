"""
Experiment: 16-Dimensional Universal Compression
Test if observer's compression generalizes across data types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rssa_v3 import RSSA
import os
import time

class CompressionExperiment:
    """
    Test if 16-D observer can universally compress information
    """
    
    def __init__(self):
        # Hardware acceleration for Mac
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
            
        print(f"Using device: {self.device}")
        self.results = []
        os.makedirs('experiments/scaling', exist_ok=True)
    
    # ========================================================================
    # TEST 1: Information Density Sweep
    # ========================================================================
    
    def test_information_density(self):
        print("\n" + "="*70)
        print("TEST 1: Information Density Sweep | FAST MODE")
        print("="*70)
        
        configs = [
            {'name': 'Ultra-sparse (1 bit)',  'vocab': 6,   'entropy': 1.0},
            {'name': 'Sparse (2 bits)',       'vocab': 8,   'entropy': 2.0},
            {'name': 'Low (4 bits)',          'vocab': 20,  'entropy': 4.0},
            {'name': 'Medium (8 bits)',       'vocab': 260, 'entropy': 8.0},
        ]
        
        results = []
        for config in configs:
            print(f"\nTesting: {config['name']}")
            seq_len = 4
            total_info = seq_len * config['entropy']
            
            # Focus on 16-D as requested
            for d_obs in [16, 64]:
                capacity = d_obs * 32
                print(f"  d_obs={d_obs} (util: {total_info/capacity:.1%})", end=' ', flush=True)
                
                model = RSSA(
                    vocab_size=config['vocab'],
                    d_model=128,
                    d_observer=d_obs,
                    n_layers=2, # Leaner model
                    processor_window=32,
                    observer_tau=0.7, # FASTER INGESTION for compression tests
                    use_ssm=False
                ).to(self.device)
                
                # Fast convergence check: 1200 steps for small signal
                loss = self._train_copy_task(
                    model, 
                    vocab_size=config['vocab'],
                    seq_len=seq_len,
                    delay=40, # Short delay but still > window (32)
                    steps=1200
                )
                
                success = loss < 0.2
                print(f" | Loss: {loss:.6f} | {'✅' if success else '❌'}")
                
                results.append({
                    'test': 'density',
                    'name': config['name'],
                    'entropy': config['entropy'],
                    'total_bits': total_info,
                    'd_obs': d_obs,
                    'utilization': total_info / capacity,
                    'loss': loss,
                    'success': success
                })
        return results
    
    def test_sequence_length(self):
        print("\n" + "="*70)
        print("TEST 2: Sequence Length Sweep | FAST MODE")
        print("="*70)
        
        vocab = 20
        entropy_per_token = np.log2(vocab)
        seq_lengths = [8, 16, 24] # Slightly reduced range
        
        results = []
        for seq_len in seq_lengths:
            total_info = seq_len * entropy_per_token
            print(f"\nSeq Length: {seq_len} ({total_info:.1f} bits)")
            
            for d_obs in [16, 64]:
                capacity = d_obs * 32
                print(f"  d_obs={d_obs:3d} (util: {total_info/capacity:>5.1%})", end=' ', flush=True)
                
                model = RSSA(
                    vocab_size=vocab,
                    d_model=128,
                    d_observer=d_obs,
                    n_layers=2,
                    processor_window=32,
                    observer_tau=0.7,
                    use_ssm=False
                ).to(self.device)
                
                loss = self._train_copy_task(
                    model,
                    vocab_size=vocab,
                    seq_len=seq_len,
                    delay=40,
                    steps=1200
                )
                
                success = loss < 0.1
                print(f" | Loss: {loss:.6f} | {'✅' if success else '❌'}")
                
                results.append({
                    'test': 'length',
                    'seq_len': seq_len,
                    'd_obs': d_obs,
                    'utilization': total_info / capacity,
                    'loss': loss,
                    'success': success
                })
        return results
    
    def test_structure(self):
        print("\n" + "="*70)
        print("TEST 3: Structure Exploitation | FAST MODE")
        print("="*70)
        
        vocab = 20
        seq_len = 24
        
        def gen_markov():
            seq = [np.random.randint(0, vocab-2)]
            for _ in range(seq_len - 1):
                if np.random.rand() < 0.7:
                    seq.append(seq[-1])
                else:
                    seq.append(np.random.randint(0, vocab-2))
            return torch.tensor(seq)

        structures = [
            {'name': 'Random', 'gen': lambda: torch.randint(0, vocab-2, (seq_len,)), 'theory': np.log2(vocab-2) * seq_len},
            {'name': 'Markov-70%', 'gen': gen_markov, 'theory': seq_len * 1.5}
        ]
        
        results = []
        for struct in structures:
            print(f"\nStructure: {struct['name']} ({struct['theory']:.1f} bits)")
            for d_obs in [16, 64]:
                model = RSSA(vocab_size=vocab, d_model=128, d_observer=d_obs, n_layers=2, processor_window=32, observer_tau=0.7, use_ssm=False).to(self.device)
                loss = self._train_structured_copy(model, struct['gen'], delay=40, steps=1200)
                success = loss < 0.2
                print(f"  d_obs={d_obs:3d} | Loss: {loss:.6f} | {'✅' if success else '❌'}")
                results.append({'test': 'structure', 'name': struct['name'], 'd_obs': d_obs, 'loss': loss, 'success': success})
        return results

    def _train_copy_task(self, model, vocab_size, seq_len, delay, steps):
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3) # Higher LR for fast sweep
        pad_token = vocab_size - 1
        query_token = vocab_size - 2
        
        last_loss = 1.0
        for step in range(steps):
            batch_size = 32 # Larger batch
            input_seq = torch.randint(0, vocab_size - 2, (batch_size, seq_len))
            
            x = torch.cat([
                input_seq,
                torch.full((batch_size, delay), pad_token),
                torch.full((batch_size, 1), query_token),
                torch.full((batch_size, seq_len), pad_token)
            ], dim=1).to(self.device)
            
            y = torch.cat([
                torch.full((batch_size, seq_len + delay + 1), -100),
                input_seq
            ], dim=1).to(self.device)
            
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            last_loss = loss.item()
            if (step + 1) % 100 == 0:
                print(".", end="", flush=True)
                
        return last_loss

    def _train_structured_copy(self, model, generator, delay, steps):
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        vocab_size = model.vocab_size
        pad_token = vocab_size - 1
        query_token = vocab_size - 2
        
        last_loss = 1.0
        for step in range(steps):
            batch_size = 32
            seqs = torch.stack([generator() for _ in range(batch_size)])
            seq_len = seqs.shape[1]
            
            x = torch.cat([
                seqs,
                torch.full((batch_size, delay), pad_token),
                torch.full((batch_size, 1), query_token),
                torch.full((batch_size, seq_len), pad_token)
            ], dim=1).to(self.device)
            
            y = torch.cat([
                torch.full((batch_size, seq_len + delay + 1), -100),
                seqs
            ], dim=1).to(self.device)
            
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            last_loss = loss.item()
            if (step + 1) % 100 == 0:
                print(".", end="", flush=True)
                
        return last_loss

    def run_all(self):
        density = self.test_information_density()
        length = self.test_sequence_length()
        structure = self.test_structure()
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        # Check success
        successes = [r for r in density + length + structure if r['success']]
        print(f"Total Successes: {len(successes)} / {len(density + length + structure)}")
        
        if all(r['success'] for r in density if r['entropy'] <= 4.0):
            print("🎉 Lower-entropy Universal Compression confirmed!")
        else:
            print("⚠️ Mixed results on low-entropy compression.")

if __name__ == "__main__":
    exp = CompressionExperiment()
    exp.run_all()
