#!/usr/bin/env python3
"""
Create a larger synthetic corpus for scale testing.
"""

import json
from pathlib import Path
import random

# Sample topics and templates
TOPICS = [
    ("technology", ["computer", "software", "hardware", "internet", "digital", "data", "network", "system"]),
    ("science", ["research", "experiment", "theory", "discovery", "laboratory", "scientist", "analysis", "study"]),
    ("business", ["company", "market", "profit", "customer", "product", "service", "sales", "revenue"]),
    ("health", ["medicine", "doctor", "patient", "treatment", "hospital", "disease", "therapy", "healthcare"]),
    ("education", ["school", "student", "teacher", "learning", "university", "course", "study", "knowledge"]),
    ("sports", ["team", "player", "game", "championship", "coach", "training", "competition", "athlete"]),
    ("environment", ["climate", "nature", "pollution", "conservation", "ecosystem", "wildlife", "sustainability", "energy"]),
    ("finance", ["investment", "stock", "bond", "portfolio", "trading", "market", "capital", "fund"]),
    ("food", ["restaurant", "recipe", "cooking", "ingredient", "chef", "cuisine", "meal", "flavor"]),
    ("travel", ["destination", "tourism", "hotel", "flight", "vacation", "adventure", "culture", "journey"]),
]

TEMPLATES = [
    "The {topic} industry has seen significant growth in {word1} and {word2} sectors.",
    "Recent developments in {word1} have transformed how we approach {word2} in modern {topic}.",
    "Experts in {topic} emphasize the importance of {word1} when dealing with {word2} challenges.",
    "The relationship between {word1} and {word2} is crucial for understanding {topic} dynamics.",
    "New research on {word1} reveals unexpected connections to {word2} in the field of {topic}.",
    "Leading professionals recommend focusing on {word1} to improve {word2} outcomes in {topic}.",
    "The integration of {word1} with {word2} represents a major breakthrough in {topic} applications.",
    "Historical analysis shows that {word1} has always influenced {word2} trends in {topic}.",
    "Future predictions suggest that {word1} will revolutionize {word2} practices across {topic}.",
    "Current best practices in {topic} prioritize {word1} over traditional {word2} methods.",
]

def generate_document(doc_id, topic_name, words):
    """Generate a synthetic document."""
    # Pick random template and words
    template = random.choice(TEMPLATES)
    word1, word2 = random.sample(words, 2)
    
    # Generate 3-5 sentences
    num_sentences = random.randint(3, 5)
    sentences = []
    
    for _ in range(num_sentences):
        template = random.choice(TEMPLATES)
        w1, w2 = random.sample(words, 2)
        sentence = template.format(topic=topic_name, word1=w1, word2=w2)
        sentences.append(sentence)
    
    text = " ".join(sentences)
    
    return {
        "id": f"doc_{doc_id}",
        "text": text
    }

def main():
    print("Generating synthetic corpus for scale testing...")
    
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / "synthetic_10k.jsonl"
    
    print(f"Creating {output_path}...")
    
    with open(output_path, 'w') as f:
        for i in range(10000):
            if i % 1000 == 0:
                print(f"  Generated {i} documents...")
            
            # Pick random topic
            topic_name, words = random.choice(TOPICS)
            doc = generate_document(i, topic_name, words)
            f.write(json.dumps(doc) + '\n')
    
    print(f"\nGenerated 10,000 documents to {output_path}")
    print("Done!")

if __name__ == "__main__":
    main()
