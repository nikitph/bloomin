from .ast_extractor import extract_witnesses as extract_witnesses_ast
from .encoder import encode
from .decoder import decode
import yaml
import os

# Load vocabulary
# Assuming running from the project root or relative import
# Let's try to find the witnesses.yaml dynamically or assume standard pos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VOCAB_PATH = os.path.join(BASE_DIR, "witnesses.yaml")

def load_vocabulary():
    with open(VOCAB_PATH, "r") as f:
        data = yaml.safe_load(f)
    
    vocab = set()
    for category, items in data.items():
        if items:
            for item in items:
                vocab.add(item)
    return list(vocab)

VOCAB = load_vocabulary()

def semantic_diff(code_before: str, code_after: str):
    w1 = extract_witnesses_ast(code_before)
    w2 = extract_witnesses_ast(code_after)

    b1 = encode(w1)
    b2 = encode(w2)

    delta = b1 ^ b2
    
    # added: bits present in AFTER but not BEFORE
    # delta & b2  => (b1 ^ b2) & b2 => (b1 & b2) ^ b2... wait.
    # Truth table:
    # b1 b2 | delta | delta & b2
    # 0  0  | 0     | 0
    # 0  1  | 1     | 1  <-- Added
    # 1  0  | 1     | 0  <-- Removed
    # 1  1  | 0     | 0
    # Correct.
    
    added_bits = delta & b2
    removed_bits = delta & b1

    added = decode(added_bits, VOCAB)
    removed = decode(removed_bits, VOCAB)

    return {
        "added_semantics": added,
        "removed_semantics": removed,
        "semantic_distance": delta.count()
    }
