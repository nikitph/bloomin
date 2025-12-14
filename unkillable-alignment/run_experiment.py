#!/usr/bin/env python3
"""
Unkillable Alignment: Style Preservation as Safety Proxy

Experiment:
1. Fine-tune GPT-2 on positive sentiment text (establish "safe" style)
2. Fine-tune on neutral/technical text (new capability)
3. Measure: Does the model still generate positive when prompted?

This demonstrates that REWA-C geometric preservation can maintain
learned behaviors (safety/style) during subsequent fine-tuning.

Metrics:
- Perplexity on each domain
- Sentiment score of generations (using a classifier)
- Style drift (cosine similarity of hidden states)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
import numpy as np
from collections import defaultdict

from src.rewa_gpt2 import REWAStyleGPT2, TextDataset, train_epoch


def load_sentiment_data(tokenizer, sentiment='positive', n_samples=500):
    """Load sentiment-specific text data."""
    print(f"Loading {sentiment} sentiment data...")

    # Use IMDB dataset
    dataset = load_dataset('imdb', split='train')

    # Filter by sentiment (truncate to save memory)
    if sentiment == 'positive':
        texts = [x['text'][:200] for x in dataset if x['label'] == 1][:n_samples]
    else:
        texts = [x['text'][:200] for x in dataset if x['label'] == 0][:n_samples]

    return TextDataset(texts, tokenizer)


def load_technical_data(tokenizer, n_samples=500):
    """Load technical/neutral text data."""
    print("Loading technical data...")

    # Use wikitext for neutral text
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

    texts = [x['text'] for x in dataset if len(x['text']) > 100][:n_samples]

    return TextDataset(texts, tokenizer)


def simple_sentiment_score(text):
    """Simple word-based sentiment scoring."""
    positive_words = {
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love',
        'loved', 'best', 'happy', 'beautiful', 'perfect', 'brilliant', 'awesome',
        'enjoy', 'enjoyed', 'nice', 'positive', 'exciting', 'fun', 'recommend',
        'incredible', 'outstanding', 'superb', 'delightful', 'pleased', 'impressive',
        'liked', 'favorite', 'favourite', 'pleasure', 'well', 'masterpiece', 'touching',
        'moving', 'entertaining', 'compelling', 'captivating', 'engaging', 'heartwarming',
        'satisfying', 'refreshing', 'hilarious', 'funny', 'charming', 'smart', 'clever',
        'remarkable', 'extraordinary', 'phenomenal', 'magnificent', 'gorgeous', 'stunning'
    }
    negative_words = {
        'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'hated', 'boring',
        'poor', 'waste', 'disappointing', 'disappointed', 'negative', 'ugly',
        'annoying', 'stupid', 'dumb', 'pathetic', 'fails', 'failed', 'mediocre',
        'weak', 'useless', 'disaster', 'painful', 'frustrating', 'avoid', 'dull',
        'tedious', 'bland', 'forgettable', 'predictable', 'overrated', 'lame', 'cheesy',
        'cliche', 'derivative', 'uninspired', 'lifeless', 'tiresome', 'uninteresting',
        'shallow', 'ridiculous', 'absurd', 'nonsense', 'pointless', 'dragged', 'slow'
    }

    words = text.lower().split()
    pos_count = sum(1 for w in words if w.strip('.,!?') in positive_words)
    neg_count = sum(1 for w in words if w.strip('.,!?') in negative_words)

    total = pos_count + neg_count
    if total == 0:
        return 0.5
    return pos_count / total


def evaluate_sentiment(model, prompts, sentiment_classifier=None):
    """Evaluate sentiment of generated text using simple word-based scoring."""
    sentiments = []

    for prompt in prompts:
        generated = model.generate(prompt, max_length=60, do_sample=True, temperature=0.7)
        # Remove prompt from generated
        generated_text = generated[len(prompt):].strip()

        if generated_text:
            score = simple_sentiment_score(generated_text)
            sentiments.append(score)
        else:
            sentiments.append(0.5)

    return np.mean(sentiments)


def evaluate_perplexity(model, dataloader, device):
    """Compute perplexity on a dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # Sum loss over tokens
            total_loss += outputs.loss.item() * attention_mask.sum().item()
            total_tokens += attention_mask.sum().item()

    return np.exp(total_loss / total_tokens)


def run_experiment():
    print("="*70)
    print("UNKILLABLE ALIGNMENT: Style Preservation During Fine-tuning")
    print("="*70)

    # Device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Using simple word-based sentiment scoring (no external classifier)

    # Test prompts (more varied)
    test_prompts = [
        "I really think that this",
        "The movie was absolutely",
        "This product is definitely",
        "My experience has been very",
        "Overall, I would say it was",
        "In my opinion, this is",
        "The quality of this is",
        "I have to say that this was",
    ]

    # Parameters
    n_samples = 200
    epochs_style = 2
    epochs_task = 2
    batch_size = 4
    lr = 5e-5

    results = {}

    # ============================================
    # Baseline: Standard fine-tuning (no preservation)
    # ============================================
    print("\n" + "="*60)
    print("BASELINE: Standard Fine-tuning (No Preservation)")
    print("="*60)

    model = REWAStyleGPT2(lambda_style=0.0).to(device)
    tokenizer = model.tokenizer

    # Load data
    positive_data = load_sentiment_data(tokenizer, 'positive', n_samples)
    technical_data = load_technical_data(tokenizer, n_samples)

    positive_loader = DataLoader(positive_data, batch_size=batch_size, shuffle=True)
    technical_loader = DataLoader(technical_data, batch_size=batch_size, shuffle=True)

    # Phase 1: Train on positive sentiment
    print("\nPhase 1: Training on positive sentiment...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs_style):
        loss = train_epoch(model, positive_loader, optimizer, device)
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}")

    # Evaluate after positive training
    sentiment_after_pos = evaluate_sentiment(model, test_prompts)
    print(f"  Sentiment score after positive training: {sentiment_after_pos:.3f}")

    # Phase 2: Train on technical text
    print("\nPhase 2: Training on technical text...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs_task):
        loss = train_epoch(model, technical_loader, optimizer, device)
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}")

    # Evaluate after technical training
    sentiment_after_tech = evaluate_sentiment(model, test_prompts)
    print(f"  Sentiment score after technical training: {sentiment_after_tech:.3f}")

    results['baseline'] = {
        'sentiment_after_positive': sentiment_after_pos,
        'sentiment_after_technical': sentiment_after_tech,
        'sentiment_drop': sentiment_after_pos - sentiment_after_tech
    }

    # ============================================
    # REWA-C: With geometric preservation
    # ============================================
    print("\n" + "="*60)
    print("REWA-C: With Geometric Style Preservation (λ=10)")
    print("="*60)

    model = REWAStyleGPT2(lambda_style=10.0).to(device)
    tokenizer = model.tokenizer

    # Reload data with new tokenizer
    positive_data = load_sentiment_data(tokenizer, 'positive', n_samples)
    technical_data = load_technical_data(tokenizer, n_samples)

    positive_loader = DataLoader(positive_data, batch_size=batch_size, shuffle=True)
    technical_loader = DataLoader(technical_data, batch_size=batch_size, shuffle=True)

    # Phase 1: Train on positive sentiment
    print("\nPhase 1: Training on positive sentiment...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs_style):
        loss = train_epoch(model, positive_loader, optimizer, device)
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}")

    # Store positive style geometry
    model.store_style_geometry(positive_loader, 'positive')

    # Evaluate after positive training
    sentiment_after_pos = evaluate_sentiment(model, test_prompts)
    print(f"  Sentiment score after positive training: {sentiment_after_pos:.3f}")

    # Phase 2: Train on technical text WITH style preservation
    print("\nPhase 2: Training on technical text (with preservation)...")
    model.set_current_style('positive')  # Enable preservation
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs_task):
        loss = train_epoch(model, technical_loader, optimizer, device)
        print(f"  Epoch {epoch+1}: Loss={loss:.4f}")

    # Evaluate after technical training
    sentiment_after_tech = evaluate_sentiment(model, test_prompts)
    print(f"  Sentiment score after technical training: {sentiment_after_tech:.3f}")

    results['rewa_c'] = {
        'sentiment_after_positive': sentiment_after_pos,
        'sentiment_after_technical': sentiment_after_tech,
        'sentiment_drop': sentiment_after_pos - sentiment_after_tech
    }

    # ============================================
    # Summary
    # ============================================
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print("\nSentiment Preservation (1.0 = positive, 0.0 = negative):")
    print(f"{'Method':<30} {'After Positive':>15} {'After Technical':>17} {'Drop':>10}")
    print("-"*70)

    for method, res in results.items():
        print(f"{method:<30} {res['sentiment_after_positive']:>15.3f} {res['sentiment_after_technical']:>17.3f} {res['sentiment_drop']:>10.3f}")

    # Analysis
    print("\nAnalysis:")
    baseline_drop = results['baseline']['sentiment_drop']
    rewa_drop = results['rewa_c']['sentiment_drop']

    if rewa_drop < baseline_drop:
        improvement = baseline_drop - rewa_drop
        print(f"  REWA-C preserved {improvement:.3f} more sentiment than baseline")
        print(f"  ({(1 - rewa_drop/baseline_drop)*100:.1f}% less forgetting)")
    else:
        print(f"  Baseline preserved better (unusual result)")

    # Example generations
    print("\n" + "="*70)
    print("EXAMPLE GENERATIONS")
    print("="*70)

    for prompt in test_prompts[:3]:
        print(f"\nPrompt: '{prompt}'")
        generated = model.generate(prompt, max_length=50, do_sample=True, temperature=0.7)
        print(f"REWA-C: {generated}")


if __name__ == "__main__":
    run_experiment()
