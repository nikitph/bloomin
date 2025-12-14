"""
REWA-C for GPT-2: Preserving Behavior During Fine-tuning

Experiment: "Style Preservation as Safety Proxy"

Instead of full safety alignment (which requires specific data), we demonstrate:
1. Train GPT-2 to have a specific "style" (positive vs negative sentiment)
2. Fine-tune on another task (different domain text)
3. Measure if the original style tendency is preserved

This is equivalent to the safety preservation problem:
- Safety = a learned behavior we want to preserve
- Fine-tuning = learning new capabilities
- Goal = new capabilities without losing safety

Key insight from REWA theory:
- The model's behavior is encoded in its hidden state geometry
- Preserve the geometric relationship between prompt representations
- This preserves the learned behavior (style/safety)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class StyleGeometry:
    """Stores geometric structure of a style's representation."""
    style_name: str
    centroid: torch.Tensor  # Mean hidden state for this style
    spread: float  # Variance of hidden states


class GeometricStylePreserver(nn.Module):
    """
    Preserves the geometric relationship between different styles.

    For safety: "harmful prompt" style vs "safe response" style
    For sentiment: "positive" style vs "negative" style
    """

    def __init__(self):
        super().__init__()
        self.style_geometries: Dict[str, StyleGeometry] = {}
        self.reference_similarities: Optional[torch.Tensor] = None

    def compute_style_geometry(
        self,
        hidden_states: torch.Tensor,  # (n_samples, hidden_dim)
        style_name: str
    ) -> StyleGeometry:
        """Compute geometric representation of a style."""
        centroid = hidden_states.mean(dim=0)
        spread = torch.norm(hidden_states - centroid, dim=1).mean().item()

        return StyleGeometry(
            style_name=style_name,
            centroid=centroid.detach(),
            spread=spread
        )

    def store_style(self, hidden_states: torch.Tensor, style_name: str):
        """Store a style's geometry."""
        geometry = self.compute_style_geometry(hidden_states, style_name)
        self.style_geometries[style_name] = geometry

        # Update reference similarities between all styles
        if len(self.style_geometries) > 1:
            self._update_reference_similarities()

    def _update_reference_similarities(self):
        """Compute similarity matrix between all stored styles."""
        styles = list(self.style_geometries.keys())
        n = len(styles)

        centroids = torch.stack([self.style_geometries[s].centroid for s in styles])
        centroids_norm = F.normalize(centroids, dim=1, eps=1e-8)

        self.reference_similarities = centroids_norm @ centroids_norm.T
        self.reference_styles = styles

    def compute_preservation_loss(
        self,
        hidden_states: torch.Tensor,
        style_name: str
    ) -> torch.Tensor:
        """Compute loss to preserve relationship with stored styles."""
        if not self.style_geometries or self.reference_similarities is None:
            return torch.tensor(0.0, device=hidden_states.device, requires_grad=True)

        # Current centroid
        current_centroid = hidden_states.mean(dim=0)
        current_centroid_norm = F.normalize(current_centroid.unsqueeze(0), dim=1, eps=1e-8)

        # Compute current similarities to all stored styles
        stored_centroids = torch.stack([
            self.style_geometries[s].centroid for s in self.reference_styles
        ]).to(hidden_states.device)
        stored_centroids_norm = F.normalize(stored_centroids, dim=1, eps=1e-8)

        current_similarities = (current_centroid_norm @ stored_centroids_norm.T).squeeze()

        # Get reference similarities for this style
        if style_name in self.reference_styles:
            idx = self.reference_styles.index(style_name)
            ref_similarities = self.reference_similarities[idx].to(hidden_states.device)

            # Preserve relative similarities
            loss = F.mse_loss(current_similarities, ref_similarities)
        else:
            # New style - just preserve distances to existing styles
            loss = torch.tensor(0.0, device=hidden_states.device, requires_grad=True)

        return loss


class REWAStyleGPT2(nn.Module):
    """
    GPT-2 with REWA-C style/safety preservation.
    """

    def __init__(
        self,
        model_name: str = 'gpt2',
        lambda_style: float = 10.0
    ):
        super().__init__()

        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.style_preserver = GeometricStylePreserver()
        self.lambda_style = lambda_style

        self.current_style = None

    def get_hidden_states(self, input_ids, attention_mask) -> torch.Tensor:
        """Get last hidden states (last token of each sequence)."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        # Get last layer hidden states
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden)

        # Get last non-padding token for each sequence
        # Find last non-padding position
        seq_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)

        last_hidden = hidden_states[batch_indices, seq_lengths]  # (batch, hidden)

        return last_hidden

    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass with optional style preservation."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True
        )

        loss = outputs.loss

        # Add style preservation loss
        if self.current_style and self.lambda_style > 0:
            hidden_states = outputs.hidden_states[-1]
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_hidden = hidden_states[batch_indices, seq_lengths]

            style_loss = self.style_preserver.compute_preservation_loss(
                last_hidden, self.current_style
            )

            if loss is not None:
                loss = loss + self.lambda_style * style_loss

        return {'loss': loss, 'logits': outputs.logits}

    def store_style_geometry(self, dataloader: DataLoader, style_name: str):
        """Extract and store geometry for a style."""
        self.eval()
        all_hidden = []

        device = next(self.model.parameters()).device

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                hidden = self.get_hidden_states(input_ids, attention_mask)
                all_hidden.append(hidden.cpu())

        all_hidden = torch.cat(all_hidden)
        self.style_preserver.store_style(all_hidden, style_name)

        print(f"Stored style '{style_name}': centroid norm={all_hidden.mean(dim=0).norm():.3f}, spread={self.style_preserver.style_geometries[style_name].spread:.3f}")

    def set_current_style(self, style_name: str):
        """Set the current style for preservation during training."""
        self.current_style = style_name

    def generate(self, prompt: str, max_length: int = 50, **kwargs) -> str:
        """Generate text from a prompt."""
        inputs = self.tokenizer(prompt, return_tensors='pt')
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}

        outputs = self.model.generate(
            inputs['input_ids'],
            max_length=max_length,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


class TextDataset(Dataset):
    """Simple text dataset for GPT-2."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 64):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['input_ids'][idx]  # For LM training
        }


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs['loss']

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


if __name__ == "__main__":
    print("Testing REWA-Style-GPT2...")

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Device: {device}")

    model = REWAStyleGPT2(lambda_style=10.0).to(device)

    # Test generation
    prompt = "The future of AI is"
    generated = model.generate(prompt, max_length=30)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")

    print("\nTest passed!")
