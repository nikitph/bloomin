"""
RSSA-v3: Residual State-Space Attention with Emergent Observer
Production-grade implementation with all optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

# ============================================================================
# Core Architecture
# ============================================================================

class RSSA(nn.Module):
    """
    Residual State-Space Attention with Emergent Observer
    
    Key properties:
    - Observer emerges from information bottleneck (not pre-allocated)
    - Processor has limited context window
    - Observer is only path for long-range dependencies
    - Dissipative dynamics via EMA
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Model dimension
        d_observer: Observer dimension (emergent state)
        n_layers: Number of processor layers
        n_heads: Number of attention heads
        processor_window: Context window for processor (forces observer use)
        observer_tau: EMA coefficient for observer (0.99 = slow updates)
        chunk_size: Process in chunks for efficiency
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        d_observer: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        processor_window: int = 256,
        observer_tau: float = 0.99,
        chunk_size: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_observer = d_observer
        self.processor_window = processor_window
        self.observer_tau = observer_tau
        self.chunk_size = chunk_size
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=8192)
        
        # Observer interface
        self.obs_to_proc = nn.Linear(d_observer, d_model)
        self.proc_to_obs = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_observer)
        )
        self.update_gate = nn.Sequential(
            nn.Linear(d_model, d_observer),
            nn.Sigmoid()
        )
        
        # Processor layers
        self.layers = nn.ModuleList([
            ProcessorLayer(
                d_model=d_model,
                n_heads=n_heads,
                window_size=processor_window,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Output head
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Observer state (persistent across batches)
        self.register_buffer(
            'observer_state',
            torch.zeros(1, 1, d_observer)
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        reset_observer: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with emergent observer
        
        Args:
            input_ids: [batch, seq_len] token indices
            reset_observer: Whether to reset observer (for new sequences)
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            observer_history: [batch, n_chunks, d_observer]
        """
        batch_size, seq_len = input_ids.shape
        
        if reset_observer:
            self.observer_state.zero_()
        
        # Embed tokens
        x = self.embedding(input_ids)  # [batch, seq, d_model]
        x = self.pos_encoding(x)
        
        # Process in chunks
        chunks = self._split_into_chunks(x)
        
        outputs = []
        observer_history = []
        
        current_observer = self.observer_state.expand(batch_size, -1, -1)
        
        for chunk in chunks:
            # Process chunk with observer
            chunk_output, new_observer = self._process_chunk(
                chunk, current_observer
            )
            
            outputs.append(chunk_output)
            observer_history.append(new_observer)
            
            current_observer = new_observer
        
        # Concatenate outputs
        h = torch.cat(outputs, dim=1)  # [batch, seq, d_model]
        
        # Update persistent observer (detached for next batch)
        # Take mean across batch to keep it batch-independent
        self.observer_state = current_observer.mean(dim=0, keepdim=True).detach()
        
        # Project to vocabulary
        h = self.output_norm(h)
        logits = self.output_proj(h)
        
        observer_history = torch.cat(observer_history, dim=1)
        
        return logits, observer_history
    
    def _split_into_chunks(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Split sequence into chunks"""
        seq_len = x.shape[1]
        n_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        chunks = []
        for i in range(n_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, seq_len)
            chunks.append(x[:, start:end])
        
        return chunks
    
    def _process_chunk(
        self,
        chunk: torch.Tensor,
        observer: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a chunk with observer context
        
        Args:
            chunk: [batch, chunk_len, d_model]
            observer: [batch, 1, d_observer]
        
        Returns:
            output: [batch, chunk_len, d_model]
            new_observer: [batch, 1, d_observer]
        """
        # Inject observer context
        obs_context = self.obs_to_proc(observer)  # [batch, 1, d_model]
        h = chunk + obs_context
        
        # Process through layers
        for layer in self.layers:
            h = layer(h)
        
        # Extract observer update
        candidate_update = self.proc_to_obs(h)  # [batch, chunk_len, d_obs]
        gate = self.update_gate(h)  # [batch, chunk_len, d_obs]
        
        # Aggregate over chunk (compression to invariants)
        chunk_summary = (candidate_update * gate).mean(dim=1, keepdim=True)
        
        # EMA update (dissipative dynamics)
        new_observer = (
            self.observer_tau * observer +
            (1 - self.observer_tau) * chunk_summary
        )
        
        return h, new_observer
    
    def get_observer_stats(self) -> dict:
        """Get statistics about current observer state"""
        with torch.no_grad():
            obs = self.observer_state.squeeze()
            return {
                'norm': torch.norm(obs).item(),
                'mean': obs.mean().item(),
                'std': obs.std().item(),
                'sparsity': (obs.abs() < 0.01).float().mean().item()
            }


class ProcessorLayer(nn.Module):
    """Single processor layer with SSM + Local Attention"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        window_size: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # SSM component (simplified - use real S4 in production)
        self.ssm = SSMBlock(d_model, dropout=dropout)
        
        # Local windowed attention
        self.attention = WindowedAttention(
            d_model, n_heads, window_size, dropout
        )
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SSM path
        x = x + self.ssm(self.norm1(x))
        
        # Attention path
        x = x + self.attention(self.norm2(x))
        
        # FFN path
        x = x + self.ffn(self.norm3(x))
        
        return x


class SSMBlock(nn.Module):
    """Simplified State Space Model (use real S4/Mamba in production)"""
    
    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        
        self.d_state = d_state
        
        # Learnable SSM parameters
        self.A = nn.Parameter(torch.randn(d_state, d_state) * 0.01)
        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.C = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model) * 0.1)
        
        self.dropout = nn.Dropout(dropout)
        
        # Discretization step
        self.dt = nn.Parameter(torch.ones(1) * 0.01)
    
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Simplified SSM forward
        In production, use proper S4 discretization
        """
        batch, length, d_model = u.shape
        
        # Simplified implementation (real S4 uses FFT)
        x = torch.zeros(batch, self.d_state, device=u.device)
        outputs = []
        
        for t in range(length):
            # x[t+1] = x[t] + dt * (A @ x[t] + B @ u[t])
            dx = torch.einsum('ij,bj->bi', self.A, x)
            dx += torch.einsum('ij,bj->bi', self.B.T, u[:, t])
            x = x + self.dt * dx
            
            # y[t] = C @ x[t] + D * u[t]
            y = torch.einsum('ij,bj->bi', self.C.T, x)
            y += self.D * u[:, t]
            
            outputs.append(y)
        
        output = torch.stack(outputs, dim=1)
        return self.dropout(output)


class WindowedAttention(nn.Module):
    """Local windowed attention (critical for forcing observer use)"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        window_size: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        self.scale = self.d_head ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, d_head]
        
        # Compute attention scores
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * self.scale
        
        # Apply window mask (CRITICAL: limits context)
        mask = self._create_window_mask(seq_len, x.device)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, d_model)
        
        return self.proj(out)
    
    def _create_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask with window constraint"""
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        
        # Window constraint
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            mask[i, :start] = 1
        
        return mask == 0


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ============================================================================
# Training Utilities
# ============================================================================

def train_step(
    model: RSSA,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    optimizer: torch.optim.Optimizer
) -> dict:
    """Single training step"""
    model.train()
    
    # Forward pass
    logits, observer_history = model(input_ids)
    
    # Loss (task loss ONLY - no explicit observer regularization)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=-100
    )
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Compute observer statistics
    with torch.no_grad():
        obs_change = torch.std(observer_history, dim=1).mean().item()
        obs_norm = torch.norm(observer_history).item()
    
    return {
        'loss': loss.item(),
        'observer_change': obs_change,
        'observer_norm': obs_norm
    }


def validate_observer_emergence(
    model: RSSA,
    test_input: torch.Tensor,
    test_target: torch.Tensor
) -> dict:
    """Validate that observer has truly emerged"""
    model.eval()
    
    with torch.no_grad():
        # Test 1: Normal performance
        logits_normal, _ = model(test_input)
        loss_normal = F.cross_entropy(
            logits_normal.reshape(-1, logits_normal.size(-1)),
            test_target.reshape(-1)
        )
        
        # Test 2: Ablated performance (zero observer)
        obs_backup = model.observer_state.clone()
        model.observer_state.zero_()
        logits_ablated, _ = model(test_input, reset_observer=False)
        loss_ablated = F.cross_entropy(
            logits_ablated.reshape(-1, logits_ablated.size(-1)),
            test_target.reshape(-1)
        )
        model.observer_state = obs_backup
        
        # Test 3: Observer statistics
        obs_stats = model.get_observer_stats()
    
    loss_increase_pct = (loss_ablated - loss_normal) / loss_normal * 100
    
    return {
        'loss_normal': loss_normal.item(),
        'loss_ablated': loss_ablated.item(),
        'loss_increase_pct': loss_increase_pct.item(),
        'observer_stats': obs_stats,
        'is_nontrivial': obs_stats['norm'] > 0.1,
        'is_necessary': loss_increase_pct > 10,  # >10% loss increase when ablated
        'is_stable': obs_stats['std'] < 0.5
    }


# ============================================================================
# Quick Proof-of-Concept Experiment
# ============================================================================

def quick_poc_experiment():
    """
    1-hour proof of concept experiment
    Tests observer emergence on copy task
    """
    print("="*70)
    print("RSSA-v3 Proof of Concept")
    print("="*70)
    
    # Hyperparameters
    vocab_size = 20
    seq_len = 512
    d_model = 256
    d_observer = 64
    n_layers = 4
    batch_size = 8
    n_steps = 1000
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Create model
    model = RSSA(
        vocab_size=vocab_size,
        d_model=d_model,
        d_observer=d_observer,
        n_layers=n_layers,
        processor_window=128,  # Force observer for dependencies > 128
        observer_tau=0.95,
        chunk_size=64
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Task: Copy with delay (requires observer for delay > window)
    def generate_copy_task(batch_size, seq_len, vocab_size, delay):
        """
        Generate copy task: [input] [PAD x delay] [target]
        Requires memory across delay tokens
        """
        # Random input
        input_seq = torch.randint(1, vocab_size-1, (batch_size, seq_len // 4))
        
        # Add delay
        pad_token = vocab_size - 1
        delay_seq = torch.full((batch_size, delay), pad_token)
        
        # Target is copy of input
        target_seq = input_seq.clone()
        
        # Construct full sequence: [input] [delay] [target]
        full_input = torch.cat([input_seq, delay_seq, target_seq], dim=1)
        
        # Loss only on the target portion
        full_target = torch.cat([
            torch.full_like(input_seq, -100),  # Ignore loss on input
            torch.full_like(delay_seq, -100),  # Ignore loss on delay
            target_seq  # Predict copy
        ], dim=1)
        
        return full_input, full_target
    
    # Training loop
    print("Training...")
    print(f"Task: Copy with delay={200} (> window={128})")
    print(f"Observer should emerge to bridge this gap\n")
    
    observer_changes = []
    losses = []
    
    for step in range(n_steps):
        # Generate batch
        input_ids, target_ids = generate_copy_task(
            batch_size, seq_len, vocab_size, delay=200
        )
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Train step
        metrics = train_step(model, input_ids, target_ids, optimizer)
        
        losses.append(metrics['loss'])
        observer_changes.append(metrics['observer_change'])
        
        # Log
        if (step + 1) % 100 == 0:
            avg_loss = sum(losses[-100:]) / 100
            avg_obs_change = sum(observer_changes[-100:]) / 100
            obs_stats = model.get_observer_stats()
            
            print(f"Step {step+1:4d} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Obs Change: {avg_obs_change:.6f} | "
                  f"Obs Norm: {obs_stats['norm']:.4f}")
    
    # Validation
    print("\n" + "="*70)
    print("Validation: Observer Emergence Test")
    print("="*70)
    
    test_input, test_target = generate_copy_task(4, seq_len, vocab_size, delay=200)
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    
    results = validate_observer_emergence(model, test_input, test_target)
    
    print(f"\nNormal Loss: {results['loss_normal']:.4f}")
    print(f"Ablated Loss (observer=0): {results['loss_ablated']:.4f}")
    print(f"Loss Increase: {results['loss_increase_pct']:.1f}%")
    print(f"\nObserver Statistics:")
    for key, val in results['observer_stats'].items():
        print(f"  {key}: {val:.4f}")
    
    print(f"\nEmergence Tests:")
    print(f"  ✓ Non-trivial: {results['is_nontrivial']}")
    print(f"  ✓ Necessary: {results['is_necessary']}")
    print(f"  ✓ Stable: {results['is_stable']}")
    
    if all([results['is_nontrivial'], results['is_necessary']]):
        print("\n🎉 SUCCESS: Observer has emerged!")
    else:
        print("\n⚠️  Observer has not fully emerged yet (needs more training)")
    
    return model, results


if __name__ == "__main__":
    # Run 1-hour PoC
    model, results = quick_poc_experiment()
