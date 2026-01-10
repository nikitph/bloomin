"""
Experiment: Hierarchical vs Standard Observer (Deep Decoder Fix)
Test if multi-layer decompression fixes the retention wall.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rssa_v3 import RSSA
import os
import time

class HierarchicalTest:
    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        print(f"Using device: {self.device}")

    def _train(self, model, delay, steps=2000):
        # Lower LR for more stable convergence with deep decoder
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4) 
        vocab_size = model.vocab_size
        pad_token = vocab_size - 1
        query_token = vocab_size - 2
        needle_len = 8
        
        last_loss = 10.0
        for step in range(steps):
            batch_size = 32
            input_seq = torch.randint(0, vocab_size - 2, (batch_size, needle_len))
            
            x = torch.cat([
                input_seq,
                torch.full((batch_size, delay), pad_token),
                torch.full((batch_size, 1), query_token),
                torch.full((batch_size, needle_len), pad_token)
            ], dim=1).to(self.device)
            
            y = torch.cat([
                torch.full((batch_size, needle_len + delay + 1), -100),
                input_seq
            ], dim=1).to(self.device)
            
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            last_loss = loss.item()
            if (step + 1) % 500 == 0:
                print(".", end="", flush=True)

        # DIAGNOSTIC: Check signal retention
        with torch.no_grad():
            _, hist = model(x) # [batch, n_chunks, d_obs]
            # Needle is in first chunk (L=8, Chunk=64)
            # Query is in last chunk
            needle_state = hist[:, 0, :]
            query_state = hist[:, -1, :]
            sim = F.cosine_similarity(needle_state, query_state, dim=-1).mean().item()
            print(f" (Sim: {sim:.3f})", end="", flush=True)
                
        return last_loss

    def run_sweep(self):
        delays = [32, 64, 128]
        results = []
        
        print("\n" + "="*70)
        print("HIERARCHICAL PERSISTENCE TEST (Deep Decoder Fix)")
        print("="*70)

        for D in delays:
            print(f"\nPersistence Delay D={D}")
            
            # 1. Standard Baseline (Sequential)
            print(f"  Standard (Sequential)  ", end=" ")
            std_model = RSSA(
                vocab_size=16, d_model=128, 
                d_obs_fast=16, d_obs_slow=1, 
                n_layers=2, processor_window=32, 
                tau_fast=0.7, tau_slow=0.0,
                use_ssm=False, use_deep_decoder=False
            ).to(self.device)
            std_loss = self._train(std_model, D)
            print(f" | Loss: {std_loss:.4f} | {'✅' if std_loss < 0.2 else '❌'}")

            # 2. Hierarchical (Shallow Integration)
            print(f"  Hierarchical (Shallow) ", end=" ")
            sha_model = RSSA(
                vocab_size=16, d_model=128, 
                d_obs_fast=16, d_obs_slow=64, 
                n_layers=2, processor_window=32, 
                tau_fast=0.7, tau_slow=0.95,
                use_ssm=False, use_deep_decoder=False
            ).to(self.device)
            sha_loss = self._train(sha_model, D)
            print(f" | Loss: {sha_loss:.4f} | {'✅' if sha_loss < 0.2 else '❌'}")

            # 3. Hierarchical (Deep Decoder)
            print(f"  Hierarchical (Deep)    ", end=" ")
            deep_model = RSSA(
                vocab_size=16, d_model=128, 
                d_obs_fast=16, d_obs_slow=64, 
                n_layers=2, processor_window=32, 
                tau_fast=0.7, tau_slow=0.95,
                use_ssm=False, use_deep_decoder=True
            ).to(self.device)
            deep_loss = self._train(deep_model, D)
            print(f" | Loss: {deep_loss:.4f} | {'✅' if deep_loss < 0.2 else '❌'}")
            
            results.append({'D': D, 'std': std_loss, 'sha': sha_loss, 'deep': deep_loss})
            
        return results

if __name__ == "__main__":
    test = HierarchicalTest()
    test.run_sweep()
