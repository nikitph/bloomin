"""
Test Models for Operator-Guided Training Validation.

We use simple models that are known to have multiple local minima
to validate that operator guidance leads to more consistent convergence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class MultiWellMLP(nn.Module):
    """
    MLP designed to have multiple local minima.

    The key property: without proper guidance, different seeds
    converge to different local minima with varying quality.

    With operator guidance: seeds should converge to equivalent
    solutions (same function class).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Build layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.backbone = nn.Sequential(*layers)

        # Output layer (separate for representation access)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # Store last hidden representations
        self._last_hidden = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get hidden representation
        hidden = self.backbone(x)
        self._last_hidden = hidden

        # Output
        return self.output_layer(hidden)

    def get_representations(self) -> Optional[torch.Tensor]:
        """Get last hidden layer representations."""
        return self._last_hidden


class SmallTransformer(nn.Module):
    """
    Small transformer for sequence tasks.

    This is a minimal transformer to test operator guidance
    on attention-based architectures.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)

        # Store representations
        self._last_hidden = None

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = x.shape

        # Embeddings
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.token_embedding(x) + self.position_embedding(positions)

        # Transformer
        if attention_mask is not None:
            # Convert to transformer mask format
            mask = attention_mask == 0
        else:
            mask = None

        hidden = self.transformer(x, src_key_padding_mask=mask)

        # Pool (use [CLS] position = first token, or mean pool)
        pooled = hidden[:, 0, :]  # CLS token
        self._last_hidden = pooled

        # Classify
        return self.classifier(pooled)

    def get_representations(self) -> Optional[torch.Tensor]:
        return self._last_hidden


class DeepNonConvexMLP(nn.Module):
    """
    Deep MLP with highly non-convex loss landscape.

    Uses techniques known to create multiple local minima:
    - Deep architecture
    - Skip connections (creating saddle points)
    - Varying layer widths

    This is the stress-test for operator guidance.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Tuple[int, ...] = (128, 256, 128, 64),
        output_dim: int = 10,
        use_skip: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()

        self.use_skip = use_skip
        self.hidden_dims = hidden_dims

        # Build layers with varying widths
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Skip connection projections (if dimensions don't match)
        if use_skip:
            self.skip_projs = nn.ModuleList()
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                if prev_dim != hidden_dim:
                    self.skip_projs.append(nn.Linear(prev_dim, hidden_dim))
                else:
                    self.skip_projs.append(nn.Identity())
                prev_dim = hidden_dim

        # Output
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

        # Representation storage
        self._layer_representations = []
        self._last_hidden = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._layer_representations = []

        hidden = x
        for i, (layer, norm, dropout) in enumerate(zip(self.layers, self.norms, self.dropouts)):
            # Main path
            new_hidden = layer(hidden)
            new_hidden = F.gelu(new_hidden)
            new_hidden = norm(new_hidden)
            new_hidden = dropout(new_hidden)

            # Skip connection
            if self.use_skip and i > 0:
                skip = self.skip_projs[i](hidden)
                new_hidden = new_hidden + 0.1 * skip

            hidden = new_hidden
            self._layer_representations.append(hidden)

        self._last_hidden = hidden
        return self.output_layer(hidden)

    def get_representations(self) -> Optional[torch.Tensor]:
        return self._last_hidden

    def get_all_layer_representations(self):
        return self._layer_representations


def create_representation_extractor(model_type: str):
    """
    Create a function to extract representations from model during forward pass.

    This is passed to the operator trainer to track representation dynamics.
    """

    def extractor(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        # Run forward pass if not already done
        with torch.no_grad():
            _ = model(x)
        return model.get_representations()

    return extractor


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
