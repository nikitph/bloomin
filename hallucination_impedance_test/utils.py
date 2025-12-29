import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import numpy as np
from typing import List, Dict

tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def get_embeddings(texts: List[str], model: GPT2LMHeadModel) -> torch.Tensor:
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model.transformer(inputs.input_ids, attention_mask=inputs.attention_mask)
        # Use mean pooling of last hidden states
        embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

def calculate_semantic_variance(embeddings: torch.Tensor) -> float:
    if embeddings.shape[0] < 2:
        return 0.0
    # Normalize to sphere
    norm_embeddings = F.normalize(embeddings, p=2, dim=1)
    # Variance along each dimension, summed
    variance = torch.var(norm_embeddings, dim=0).sum().item()
    return variance

def generate_equivalent_prompts(prompt: str) -> List[str]:
    # Simple rule-based paraphrasing for the experiment
    templates = [
        "{}",
        "Can you tell me {}?",
        "I would like to know {}",
        "Please provide the answer for: {}",
        "Query: {}",
        "{} (Be concise)"
    ]
    return [t.format(prompt) for t in templates]

import re

def factual_match(answer: str, reference: str) -> bool:
    answer = answer.lower().strip()
    reference = reference.lower().strip()
    # Use word boundaries to avoid matching single letters inside words (e.g. 'c' in 'call')
    pattern = r'\b' + re.escape(reference) + r'\b'
    return bool(re.search(pattern, answer))

def contradiction_detected(answer: str) -> bool:
    # Very simple heuristic: check for both assertive and negation words in small window
    negations = ["not", "never", "no", "won't", "can't"]
    # If the answer is long but contains "but I think no" etc.
    # For this experiment, we might just use LLM as a judge if needed, 
    # but let's keep it simple for now as a placeholder.
    return False

def unstable_under_reask(model, prompt, original_answer) -> bool:
    # Check if a second generation with different seed/variation produces a wildly different answer
    new_answer = model.generate(tokenizer.encode(prompt, return_tensors="pt").to(model.device), max_new_tokens=20)
    new_text = tokenizer.decode(new_answer[0], skip_special_tokens=True)
    # If embeddings are far apart, it's unstable
    return not factual_match(new_text, original_answer) # Over-simplified
