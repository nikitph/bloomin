"""
RSSA-v3: Residual State-Space Attention with Emergent Observer
Production-grade implementation with all optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
import numpy as np

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
        d_obs_fast: int = 16,
        d_obs_slow: int = 64,
        n_layers: int = 6,
        n_heads: int = 8,
        processor_window: int = 256,
        tau_fast: float = 0.7,
        tau_slow: float = 0.95,
        chunk_size: int = 64,
        dropout: float = 0.1,
        use_ssm: bool = True,
        use_deep_decoder: bool = True
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_obs_fast = d_obs_fast
        self.d_obs_slow = d_obs_slow
        self.d_observer = d_obs_fast + d_obs_slow
        self.processor_window = processor_window
        self.tau_fast = tau_fast
        self.tau_slow = tau_slow
        self.chunk_size = chunk_size
        self.use_ssm = use_ssm
        self.use_deep_decoder = use_deep_decoder
        
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=8192)
        
        # Hierarchical Observer interface
        # Ingestion paths (stays same)
        self.proc_to_obs_fast = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_obs_fast)
        )
        self.fast_gate = nn.Sequential(
            nn.Linear(d_model, d_obs_fast),
            nn.Sigmoid()
        )
        
        self.proc_to_obs_slow = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_obs_slow)
        )
        self.slow_gate = nn.Sequential(
            nn.Linear(d_model, d_obs_slow),
            nn.Sigmoid()
        )
        
        # Decompression (Decoder) paths
        if use_deep_decoder:
            # Deep Decoder: Multi-layer MLP to map compressed state to context
            self.decoder = nn.Sequential(
                nn.Linear(d_obs_fast + d_obs_slow, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model)
            )
        else:
            # Shallow Baseline: Simple linear projection (per-state)
            self.obs_fast_to_proc = nn.Linear(d_obs_fast, d_model // 2)
            self.obs_slow_to_proc = nn.Linear(d_obs_slow, d_model // 2)
            self.integrate_ctx = nn.Linear(d_model, d_model)
        
        # Processor layers
        self.layers = nn.ModuleList([
            ProcessorLayer(
                d_model=d_model,
                n_heads=n_heads,
                window_size=processor_window,
                dropout=dropout,
                use_ssm=use_ssm
            )
            for _ in range(n_layers)
        ])
        
        # Output head
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Hierarchical observer states (persistent across batches)
        self.register_buffer('obs_fast_state', torch.zeros(1, 1, d_obs_fast))
        self.register_buffer('obs_slow_state', torch.zeros(1, 1, d_obs_slow))
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters"""
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        reset_observer: bool = True,
        ablate_observer: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with emergent observer
        
        Args:
            input_ids: [batch, seq_len] token indices
            reset_observer: Whether to reset observer (for training/independent samples)
            ablate_observer: Whether to zero out observer influence (for validation)
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            observer_history: [batch, n_chunks, d_observer]
        """
        batch_size, seq_len = input_ids.shape
        
        # Per-batch observer states (prevents cross-sample contamination during training)
        if reset_observer:
            curr_fast = torch.zeros(batch_size, 1, self.d_obs_fast, device=input_ids.device)
            curr_slow = torch.zeros(batch_size, 1, self.d_obs_slow, device=input_ids.device)
        else:
            # Carry over from persistent buffers
            curr_fast = self.obs_fast_state.expand(batch_size, -1, -1)
            curr_slow = self.obs_slow_state.expand(batch_size, -1, -1)
        
        # Embed tokens
        x = self.embedding(input_ids)  # [batch, seq, d_model]
        x = self.pos_encoding(x)
        
        # Process in chunks
        chunks = self._split_into_chunks(x)
        
        outputs = []
        observer_history = []  # We'll store concatenated (fast, slow) for history
        
        for chunk in chunks:
            # Process chunk with hierarchical observer
            chunk_output, next_fast, next_slow = self._process_chunk(
                chunk, curr_fast, curr_slow, ablate_observer=ablate_observer
            )
            
            outputs.append(chunk_output)
            observer_history.append(torch.cat([next_fast, next_slow], dim=-1))
            
            curr_fast = next_fast
            curr_slow = next_slow
        
        # Update persistent observers
        if not reset_observer:
            self.obs_fast_state = curr_fast.mean(dim=0, keepdim=True).detach()
            self.obs_slow_state = curr_slow.mean(dim=0, keepdim=True).detach()
        
        # Concatenate outputs
        h = torch.cat(outputs, dim=1)  # [batch, seq, d_model]
        
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
        obs_fast: torch.Tensor,
        obs_slow: torch.Tensor,
        ablate_observer: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process a chunk with hierarchical observer context
        
        Args:
            chunk: [batch, chunk_len, d_model]
            obs_fast: [batch, 1, d_obs_fast]
            obs_slow: [batch, 1, d_obs_slow]
        """
        # Inject hierarchical context
        if not ablate_observer:
            if self.use_deep_decoder:
                # Deep Decompression
                combined_obs = torch.cat([obs_fast, obs_slow], dim=-1) # [batch, 1, total_d_obs]
                h = chunk + self.decoder(combined_obs)
            else:
                # Shallow Integration
                fast_ctx = self.obs_fast_to_proc(obs_fast) 
                slow_ctx = self.obs_slow_to_proc(obs_slow)
                combined_ctx = torch.cat([fast_ctx, slow_ctx], dim=-1)
                h = chunk + self.integrate_ctx(combined_ctx)
        else:
            h = chunk
        
        # Process through layers
        for layer in self.layers:
            h = layer(h)
        
        # Extract dual state updates
        if not ablate_observer:
            # Update Fast State (tau_fast=0.7)
            fast_candidate = self.proc_to_obs_fast(h)
            fast_gate = self.fast_gate(h)
            fast_summary = (fast_candidate * fast_gate).mean(dim=1, keepdim=True)
            new_fast = self.tau_fast * obs_fast + (1 - self.tau_fast) * fast_summary
            
            # Update Slow State (tau_slow=0.95)
            slow_candidate = self.proc_to_obs_slow(h)
            slow_gate = self.slow_gate(h)
            slow_summary = (slow_candidate * slow_gate).mean(dim=1, keepdim=True)
            new_slow = self.tau_slow * obs_slow + (1 - self.tau_slow) * slow_summary
        else:
            new_fast = torch.zeros_like(obs_fast)
            new_slow = torch.zeros_like(obs_slow)
        
        return h, new_fast, new_slow
    
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
        dropout: float = 0.1,
        use_ssm: bool = True
    ):
        super().__init__()
        
        # Toggleable memory component
        if use_ssm:
            self.memory = SSMBlock(d_model, dropout=dropout)
        else:
            self.memory = nn.Identity()
        
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
        # Memory path (SSM or Identity)
        x = x + self.memory(self.norm1(x))
        
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
        
        # Pre-compute projections to speed up loop
        # u @ B -> [batch, length, d_state]
        u_proj = torch.einsum('ij,blj->bli', self.B.T, u)
        
        x = torch.zeros(batch, self.d_state, device=u.device)
        outputs = []
        
        # Recurrent loop (bottleneck, but necessary for sequential logic)
        for t in range(length):
            # x[t+1] = x[t] + dt * (A @ x[t] + B @ u[t])
            dx = torch.einsum('ij,bj->bi', self.A, x) + u_proj[:, t]
            x = x + self.dt * dx
            outputs.append(x)
        
        # y[t] = C @ x[t] + D * u[t]
        x_stacked = torch.stack(outputs, dim=1)  # [batch, length, d_state]
        y = torch.einsum('ij,blj->bli', self.C.T, x_stacked) + self.D * u
        
        return self.dropout(y)


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
        """Create causal mask with window constraint (Vectorized)"""
        # Causal part (upper triangle)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        # Window part (far past)
        cols = torch.arange(seq_len, device=device).view(1, -1)
        rows = torch.arange(seq_len, device=device).view(-1, 1)
        window_mask = (rows - cols) >= self.window_size
        
        # Combine: mask if in future OR far past
        full_mask = mask | window_mask
        
        return ~full_mask


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
        logits_normal, _ = model(test_input, reset_observer=True)
        loss_normal = F.cross_entropy(
            logits_normal.reshape(-1, logits_normal.size(-1)),
            test_target.reshape(-1)
        )
        
        # Test 2: Ablated performance (observer decoupled)
        logits_ablated, _ = model(test_input, reset_observer=True, ablate_observer=True)
        loss_ablated = F.cross_entropy(
            logits_ablated.reshape(-1, logits_ablated.size(-1)),
            test_target.reshape(-1)
        )
        
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
        if (step + 1) % 10 == 0:
            avg_loss = sum(losses[-10:]) / 10
            avg_obs_change = sum(observer_changes[-10:]) / 10
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


# ============================================================================
# Phase 1: Linearized Dynamics Analysis
# ============================================================================

def compute_jacobian_spectrum(
    model: RSSA,
    test_input: torch.Tensor,
    n_samples: int = 10
) -> dict:
    """
    Compute Jacobian spectrum of observer update function using finite differences
    
    Measures ∂O_{t+1}/∂O_t to analyze linearized dynamics
    """
    model.eval()
    
    eigenvalues_real = []
    eigenvalues_imag = []
    
    eps = 1e-4  # Finite difference epsilon
    
    with torch.no_grad():
        for _ in range(n_samples):
            # Get baseline observer state
            model.observer_state = torch.randn_like(model.observer_state) * 0.1
            obs_0 = model.observer_state.clone()
            
            # Forward pass to get updated observer
            _, _ = model(test_input)
            obs_1_baseline = model.observer_state.clone()
            
            # Compute Jacobian via finite differences
            d_obs = model.d_observer
            jacobian = torch.zeros(d_obs, d_obs)
            
            for i in range(d_obs):
                # Perturb i-th dimension
                obs_perturbed = obs_0.clone()
                obs_perturbed[0, 0, i] += eps
                
                # Forward pass with perturbation
                model.observer_state = obs_perturbed
                _, _ = model(test_input)
                obs_1_perturbed = model.observer_state.clone()
                
                # Finite difference: (f(x+eps) - f(x)) / eps
                jacobian[:, i] = (obs_1_perturbed[0, 0] - obs_1_baseline[0, 0]) / eps
            
            # Compute eigenvalues
            try:
                eigvals = torch.linalg.eigvals(jacobian)
                eigenvalues_real.append(eigvals.real.cpu().numpy())
                eigenvalues_imag.append(eigvals.imag.cpu().numpy())
            except:
                pass
    
    if len(eigenvalues_real) == 0:
        return {
            'eigenvalues_real': [],
            'eigenvalues_imag': [],
            'max_eigenvalue': 0.0,
            'spectral_radius': 0.0,
            'is_contractive': False
        }
    
    eigenvalues_real = np.concatenate(eigenvalues_real)
    eigenvalues_imag = np.concatenate(eigenvalues_imag)
    
    # Compute spectral radius (max |λ|)
    spectral_radius = np.max(np.abs(eigenvalues_real + 1j * eigenvalues_imag))
    max_eigenvalue = np.max(np.abs(eigenvalues_real))
    
    return {
        'eigenvalues_real': eigenvalues_real,
        'eigenvalues_imag': eigenvalues_imag,
        'max_eigenvalue': float(max_eigenvalue),
        'spectral_radius': float(spectral_radius),
        'is_contractive': spectral_radius < 1.0
    }


def measure_contraction_rate(
    model: RSSA,
    test_input: torch.Tensor,
    n_steps: int = 50,
    perturbation_scale: float = 0.1
) -> dict:
    """
    Measure contraction rate under perturbation
    
    Perturbs observer and measures convergence rate
    """
    model.eval()
    
    # Get baseline observer state
    with torch.no_grad():
        _, _ = model(test_input)
        baseline_obs = model.observer_state.clone()
    
    # Perturb observer
    perturbation = torch.randn_like(baseline_obs) * perturbation_scale
    model.observer_state = baseline_obs + perturbation
    
    distances = []
    
    with torch.no_grad():
        for step in range(n_steps):
            # Forward pass
            _, _ = model(test_input)
            
            # Measure distance to baseline
            dist = torch.norm(model.observer_state - baseline_obs).item()
            distances.append(dist)
    
    # Fit exponential decay: d(t) = d(0) * exp(-λt)
    distances = np.array(distances)
    if distances[0] > 0:
        log_distances = np.log(distances + 1e-10)
        
        # Linear fit in log space
        t = np.arange(len(distances))
        coeffs = np.polyfit(t, log_distances, 1)
        contraction_rate = -coeffs[0]
    else:
        contraction_rate = 0.0
    
    return {
        'distances': distances.tolist(),
        'contraction_rate': float(contraction_rate),
        'initial_distance': float(distances[0]),
        'final_distance': float(distances[-1]),
        'converged': distances[-1] < 0.01
    }


def return_to_manifold_experiment(
    model: RSSA,
    test_input: torch.Tensor,
    n_perturbations: int = 10,
    perturbation_scales: list = [0.1, 0.5, 1.0]
) -> dict:
    """
    Return-to-manifold experiment
    
    Randomly perturbs trained observer and measures return time
    """
    model.eval()
    
    # Get trained observer manifold
    with torch.no_grad():
        _, _ = model(test_input)
        manifold_obs = model.observer_state.clone()
    
    results = {scale: [] for scale in perturbation_scales}
    
    for scale in perturbation_scales:
        for _ in range(n_perturbations):
            # Random perturbation
            perturbation = torch.randn_like(manifold_obs) * scale
            model.observer_state = manifold_obs + perturbation
            
            # Measure return time
            return_time = None
            threshold = 0.05 * scale  # 5% of perturbation scale
            
            with torch.no_grad():
                for step in range(100):
                    _, _ = model(test_input)
                    
                    dist = torch.norm(model.observer_state - manifold_obs).item()
                    
                    if dist < threshold and return_time is None:
                        return_time = step
                        break
            
            if return_time is None:
                return_time = 100  # Did not converge
            
            results[scale].append(return_time)
    
    # Compute statistics
    summary = {}
    for scale in perturbation_scales:
        times = results[scale]
        summary[f'scale_{scale}'] = {
            'mean_return_time': float(np.mean(times)),
            'std_return_time': float(np.std(times)),
            'max_return_time': int(np.max(times)),
            'convergence_rate': float(np.sum(np.array(times) < 100) / len(times))
        }
    
    return summary


def analyze_observer_dynamics(model: RSSA, test_input: torch.Tensor) -> dict:
    """
    Comprehensive linearized dynamics analysis
    """
    print("\n" + "="*70)
    print("Phase 1: Linearized Dynamics Analysis")
    print("="*70)
    
    # 1. Jacobian spectrum
    print("\n1. Computing Jacobian spectrum...")
    jacobian_results = compute_jacobian_spectrum(model, test_input, n_samples=5)
    
    print(f"   Spectral radius: {jacobian_results['spectral_radius']:.4f}")
    print(f"   Max eigenvalue: {jacobian_results['max_eigenvalue']:.4f}")
    print(f"   Is contractive: {jacobian_results['is_contractive']}")
    
    # 2. Contraction rate
    print("\n2. Measuring contraction rate...")
    contraction_results = measure_contraction_rate(model, test_input, n_steps=50)
    
    print(f"   Contraction rate: {contraction_results['contraction_rate']:.4f}")
    print(f"   Initial distance: {contraction_results['initial_distance']:.4f}")
    print(f"   Final distance: {contraction_results['final_distance']:.4f}")
    print(f"   Converged: {contraction_results['converged']}")
    
    # 3. Return to manifold
    print("\n3. Running return-to-manifold experiment...")
    manifold_results = return_to_manifold_experiment(
        model, test_input, n_perturbations=10
    )
    
    for scale_key, stats in manifold_results.items():
        scale = scale_key.replace('scale_', '')
        print(f"   Perturbation scale {scale}:")
        print(f"     Mean return time: {stats['mean_return_time']:.1f} steps")
        print(f"     Convergence rate: {stats['convergence_rate']*100:.0f}%")
    
    return {
        'jacobian': jacobian_results,
        'contraction': contraction_results,
        'manifold': manifold_results
    }


if __name__ == "__main__":
    # Run 1-hour PoC
    model, results = quick_poc_experiment()

