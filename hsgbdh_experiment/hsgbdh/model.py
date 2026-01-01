import torch
import torch.nn as nn
from .layer import HSGBDHLevel
from .pooling import GraphCoarsening
from .semiring import AdaptiveSemiring, ExponentialSchedule

class HSGBDH(nn.Module):
    """
    Hierarchical-Semiring-Graph-BDH with all refinements.
    """
    def __init__(self, n, d, K, block_size=32):
        super().__init__()
        self.n = n
        self.d = d
        self.K = K
        
        self.semiring = AdaptiveSemiring(
            temperature_schedule=ExponentialSchedule(1.0, 0.01, steps=1000)
        )
        
        # Components at each level
        self.levels = nn.ModuleList([
            HSGBDHLevel(
                n_k=n // (2**k),
                d=d,
                semiring=self.semiring,
                block_size=block_size
            ) for k in range(K)
        ])
        
        # Pooling between levels
        self.poolers = nn.ModuleList([
            GraphCoarsening(
                n_k=n // (2**k),
                n_kplus1=n // (2**(k+1)),
                d=d
            ) for k in range(K-1)
        ])
        
    def forward(self, x_seq):
        """
        x_seq: Input sequence (batch, seq_len, d) or just (1, d) for step-by-step
        For now, let's implement the step logic (batch=1, single step) 
        and assume the caller manages the sequence loop or x_seq is just (1, d)
        
        Let's support taking a single token vector x_t (1, d)
        """
        # Process through hierarchy (bottom-up)
        level_outputs = [] # activations (y_t)
        features = x_seq # Input feature vector
        
        # G_current for pooling
        G_current = None
        
        for k in range(self.K):
            # Update graph and get output at this level
            # Level k Input: comes from x_seq (if k=0) or pooling (if k>0)
            
            # Problem: Level k logic expects (1, d) vector or neuron activations?
            # HSGBDHLevel expects embeddings (1, d) as input to project to neurons.
            # But pooling outputs (n_kplus1, d) features if we pool FEATURES.
            # Or does it pool activations?
            # In the prompt: "poolers[k].forward(G_k, output_k)".
            # output_k is "y_t" (activations / neurons).
            # GraphCoarsening returns "features_kplus1" (n_kplus1, d)??
            
            # Let's adjust HSGBDHLevel to handle "pooled input" vs "raw input".
            # If we pool activations (n_k), we get (n_kplus1) activations.
            # But HSGBDHLevel performs E @ x -> neurons.
            # If we already have neurons (pooled), we should bypass E?
            # Or maybe we pool the FEATURES (d-dim)?
            # The prompt `output_k` is used.
            # `output_k` in HSGBDHLevel is `y_t` (neuron activations).
            
            # Let's assume we pool ACTIVATIONS.
            # GraphCoarsening: features_k (n_k) -> features_kplus1 (n_kplus1).
            # Then Level k+1 needs to take these activations.
            # We need a `forward_with_activations` or check input type in Level.
            
            if k == 0:
                output_k, G_k = self.levels[k](features)
            else:
                # features is pooled activations from previous level
                # We need to bypass E projection in HSGBDHLevel
                # Let's assume valid implementation for now:
                # We can project (n_kplus1) activations back to d using Dy?
                # Then feed to E? That seems redundant.
                # Or just use the activations directly.
                
                # Let's modify HSGBDHLevel to take "neuron_input"
                output_k, G_k = self.levels[k].forward_neurons(features)
                
            level_outputs.append(output_k)
            G_current = G_k
            
            # Pool to next level
            if k < self.K - 1:
                # Pool activations: output_k (B, n_k)
                # GraphCoarsening expects features_k (n_k, d) or (B, n_k)? 
                # Let's ensure GraphCoarsening handles (B, n_k).
                
                # G_k might be BlockSparse.
                # output_k is y_t (B, n_k).
                
                # We need to transpose output_k for GraphCoarsening if it expects (n_k, d) ?
                # No, GraphCoarsening expects features_k. 
                # If we pass (B, n_k), S is (n_k, n_kplus1).
                # features_val = (1, n_k). (1, n_k) @ S -> (1, n_kplus1)?? No.
                # dimensions: S.T @ features_k. S.T is (n_kplus1, n_k).
                # (n_kplus1, n_k) @ (n_k, d) -> (n_kplus1, d).
                # If features_k is (B, n_k) layout is wrong for S.T @ features.
                # If features_k is (1, n_k), we want result (1, n_kplus1).
                # Result = features_k @ S. (1, n_k) @ (n_k, n_kplus1) -> (1, n_kplus1).
                
                # We will handle this in general loop logic:
                pool_input = output_k.transpose(0, 1) if output_k.ndim == 2 else output_k
                # But output_k is (1, n_k). Transpose -> (n_k, 1). 
                # S.T @ (n_k, 1) -> (n_kplus1, 1). Transpose back -> (1, n_kplus1).
                
                G_next, features_next = self.poolers[k](G_current, pool_input)
                
                features = features_next.transpose(0, 1) # Back to (1, n_kplus1)
        
        # Top-down refinement
        # "refine(level_outputs[k], unpooled=...)"
        for k in range(self.K - 2, -1, -1):
            # Unpool from k+1 to k
            unpooled = self.poolers[k].unpool(level_outputs[k+1])
            # Refine level k
            # refined = level_outputs[k] + unpooled? Or logic?
            # Prompt: `refined = self.levels[k].refine(...)`
            # We need to implement `refine` in HSGBDHLevel.
            
            # Simple residual connection for now:
            # level_outputs[k] += unpooled
            # But better to have a method.
            level_outputs[k] = self.levels[k].refine(level_outputs[k], unpooled)
        
        return level_outputs[0]  # Base level output
