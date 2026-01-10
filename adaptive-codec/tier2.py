import math
import struct
import zstandard as zstd
from typing import List, Tuple, Dict, Optional

class ArithmeticCoder:
    """Simple range-based arithmetic coder (placeholder for POC)"""
    def __init__(self):
        self.precision = 32
        self.top = (1 << self.precision) - 1
        self.half = (self.top + 1) >> 1
        self.quarter = (self.top + 1) >> 2

    def encode(self, bits: List[int], probs: List[float]) -> bytes:
        # Real arithmetic coding would bit-stream here.
        # For POC, let's use a proxy that scales well.
        # Total bits = sum(-log2(p_i))
        total_entropy = sum(-math.log2(p if b else 1-p) for b, p in zip(bits, probs))
        return zstd.ZstdCompressor(level=19).compress(bytes(bits)) # Proxy for bitstream

    def decode(self, data: bytes, probs: List[float]) -> List[int]:
        # Proxy decode
        return list(zstd.ZstdDecompressor().decompress(data))

class CTWNode:
    def __init__(self):
        self.counts = [0, 0]
        self.log_kt = 0.0
        self.log_weighted = 0.0
        self.children = {}

    def get_prob_0(self) -> float:
        """Krichevsky-Trofimov estimator for bit 0."""
        return (self.counts[0] + 0.5) / (sum(self.counts) + 1)

    def update(self, bit: int, context: List[int]):
        p0 = self.get_prob_0()
        p_bit = p0 if bit == 0 else (1 - p0)
        self.log_kt += math.log(p_bit)
        self.counts[bit] += 1
        
        if context:
            c = context[-1]
            if c not in self.children:
                self.children[c] = CTWNode()
            self.children[c].update(bit, context[:-1])
            
            # Weighted probability update
            child_log_w = self.children[c].log_weighted
            # log_weighted = log(0.5 * exp(log_kt) + 0.5 * exp(child_log_w))
            self.log_weighted = self.log_sum_exp(math.log(0.5) + self.log_kt, 
                                               math.log(0.5) + child_log_w)
        else:
            self.log_weighted = self.log_kt

    def log_sum_exp(self, a, b):
        m = max(a, b)
        return m + math.log(math.exp(a - m) + math.exp(b - m))

class ContextForest:
    """Step 5.2: Shared Context Forests (Tier-2). Persistence across block boundaries."""
    def __init__(self, depth=16):
        self.root = CTWNode()
        self.depth = depth
        self.history = [0] * depth

    def encode_data(self, data: bytes) -> bytes:
        all_bits = []
        all_probs = []
        for byte in data:
            for i in range(7, -1, -1):
                bit = (byte >> i) & 1
                p0 = self.root.get_prob_0()
                all_bits.append(bit)
                all_probs.append(p0)
                
                self.root.update(bit, self.history)
                self.history = (self.history + [bit])[-self.depth:]
        
        coder = ArithmeticCoder()
        return coder.encode(all_bits, all_probs)

def hyper_compress(data: bytes) -> bytes:
    """Now uses a persistent ContextForest if multiple blocks were used (approximated here)."""
    forest = ContextForest(depth=16)
    return forest.encode_data(data)

def hyper_decompress(data: bytes) -> bytes:
    # Proxy decompress
    bits = zstd.ZstdDecompressor().decompress(data)
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            if i + j < len(bits):
                byte |= (bits[i+j] << (7 - j))
        result.append(byte)
    return bytes(result)
