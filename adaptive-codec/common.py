import math
import struct
from dataclasses import dataclass, field
from typing import List, Optional, Any, Tuple

@dataclass
class Block:
    id: int
    raw_bytes: bytes
    entropy: float = 0.0
    repetition_score: float = 0.0
    structure_score: float = 0.0
    compressed_bytes: Optional[bytes] = None
    codec_used: Optional[str] = None
    boundary_info: Any = None

@dataclass
class GlobalDictionary:
    phrases: List[bytes] = field(default_factory=list)
    grammar_rules: Optional[Any] = None
    version_hash: str = "v2"

@dataclass
class SeekIndex:
    block_offsets: List[int] = field(default_factory=list)
    block_sizes: List[int] = field(default_factory=list)

@dataclass
class CompressedFile:
    header: dict
    global_dictionary: GlobalDictionary
    L2_model: Optional[bytes]
    L1_model: Optional[bytes]
    compressed_blocks_data: bytes
    seek_index: SeekIndex

def estimate_entropy(data: bytes) -> float:
    if not data: return 0.0
    counts = {}
    for b in data: counts[b] = counts.get(b, 0) + 1
    entropy = 0.0
    for count in counts.values():
        p = count / len(data)
        entropy -= p * math.log2(p)
    return entropy

def estimate_repetition(data: bytes, sample_len: int = 16) -> float:
    if len(data) < sample_len * 2: return 0.0
    seen = set()
    collisions = 0
    total = 0
    step = max(1, len(data) // 100)
    for i in range(0, len(data) - sample_len, step):
        chunk = data[i:i+sample_len]
        if chunk in seen: collisions += 1
        else: seen.add(chunk)
        total += 1
    return collisions / total if total > 0 else 0.0

def detect_structure(data: bytes) -> float:
    if not data: return 0.0
    ascii_count = sum(1 for b in data if 32 <= b <= 126 or b in (9, 10, 13))
    return ascii_count / len(data)

CODEC_MAP = {
    "RLE": "R",
    "LZ_SHARED": "L",
    "GRAMMAR": "G",
    "RAW": "W"
}
REV_CODE_MAP = {v: k for k, v in CODEC_MAP.items()}

# Binary Layout Utilities
MAGIC = b"ACDC" # Adaptive Codec

def encode_varint(n: int) -> bytes:
    """Encode an integer as a variable-length integer."""
    out = bytearray()
    while n >= 0x80:
        out.append((n & 0x7f) | 0x80)
        n >>= 7
    out.append(n)
    return bytes(out)

def decode_varint(data: bytes, pos: int) -> Tuple[int, int]:
    """Decode a varint and return (value, new_pos)."""
    res = 0
    shift = 0
    while True:
        b = data[pos]
        res |= (b & 0x7f) << shift
        pos += 1
        if not (b & 0x80):
            return res, pos
        shift += 7
