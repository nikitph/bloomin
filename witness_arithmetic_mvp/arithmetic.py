from .ast_extractor import extract_witnesses as extract_witnesses_code
from .iam_extractor import extract_witnesses_iam
from .k8s_extractor import extract_witnesses_k8s
from .api_extractor import extract_witnesses_api
from .encoder import encode
from .decoder import decode
import yaml
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_vocabulary(domain="code"):
    filename = "witnesses.yaml" # Default
    if domain == "iam": filename = "witnesses_iam.yaml"
    elif domain == "k8s": filename = "witnesses_k8s.yaml"
    elif domain == "api": filename = "witnesses_api.yaml"
        
    vocab_path = os.path.join(BASE_DIR, filename)
    with open(vocab_path, "r") as f:
        data = yaml.safe_load(f)
    
    vocab = set()
    for category, items in data.items():
        if items:
            for item in items:
                vocab.add(item)
    return list(vocab)

def get_extractor(domain="code"):
    if domain == "iam": return extract_witnesses_iam
    if domain == "k8s": return extract_witnesses_k8s
    if domain == "api": return extract_witnesses_api
    return extract_witnesses_code

def semantic_diff(input_before: str, input_after: str, domain="code"):
    extractor = get_extractor(domain)
    vocab = load_vocabulary(domain)
    
    w1 = extractor(input_before)
    w2 = extractor(input_after)

    b1 = encode(w1)
    b2 = encode(w2)

    delta = b1 ^ b2
    
    # added: bits present in AFTER but not BEFORE
    added_bits = delta & b2
    removed_bits = delta & b1

    added = decode(added_bits, vocab)
    removed = decode(removed_bits, vocab)

    return {
        "added_semantics": added,
        "removed_semantics": removed,
        "semantic_distance": delta.count(),
        "domain": domain
    }
