from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import zstandard as zstd
from common import (
    Block, GlobalDictionary, estimate_entropy, estimate_repetition, 
    detect_structure, CODEC_MAP
)
from custom_codecs import compress_rle, LZSharedCodec, compress_grammar_placeholder

def split_into_blocks(data: bytes, block_size: int) -> List[Block]:
    blocks = []
    for i in range(0, len(data), block_size):
        chunk = data[i:i + block_size]
        blocks.append(Block(id=i // block_size, raw_bytes=chunk))
    return blocks

def analyze_block(block: Block) -> Block:
    block.entropy = estimate_entropy(block.raw_bytes)
    block.repetition_score = estimate_repetition(block.raw_bytes)
    block.structure_score = detect_structure(block.raw_bytes)
    return block

def analyze_input(data: bytes, block_size: int) -> List[Block]:
    blocks = split_into_blocks(data, block_size)
    if len(blocks) < 4:
        return [analyze_block(b) for b in blocks]
    with ProcessPoolExecutor() as executor:
        analyzed_blocks = list(executor.map(analyze_block, blocks))
    return analyzed_blocks

def sample_substrings(data: bytes, sample_rate: float, min_len: int = 4, max_len: int = 32) -> List[bytes]:
    samples = []
    step = max(1, int(1.0 / sample_rate)) if sample_rate < 1.0 else 1
    for i in range(0, len(data) - max_len, step):
        for length in range(min_len, max_len + 1, 8):
            samples.append(data[i:i+length])
    return samples

def build_global_dictionary(blocks: List[Block], sample_rate: float = 0.1) -> GlobalDictionary:
    local_counts = Counter()
    for block in blocks:
        samples = sample_substrings(block.raw_bytes, sample_rate)
        local_counts.update(samples)
    phrases = []
    for phrase, count in local_counts.items():
        if count > 1:
            gain = count * (len(phrase) - 2)
            phrases.append((phrase, gain))
    phrases.sort(key=lambda x: x[1], reverse=True)
    top_phrases = [p[0] for p in phrases[:512]]
    return GlobalDictionary(phrases=top_phrases)

def boundary_repair(blocks: List[Block], global_dict: GlobalDictionary):
    """
    Step 5.1: Exact Bit-Level Boundary Repair (Tier-1).
    Reclaims LZ matches that cross block boundaries by injecting them into 
    the global dictionary as a 'bridge phrase'.
    """
    new_phrases = []
    window_size = 32
    for i in range(len(blocks) - 1):
        # Scan for the longest common substring crossing the boundary
        # Junction = end of block i + start of block i+1
        suffix = blocks[i].raw_bytes[-window_size:]
        prefix = blocks[i+1].raw_bytes[:window_size]
        junction = suffix + prefix
        
        # Check if 16-32 byte phrases from this junction are high-gain
        # For simplicity, we extract the center 16 bytes as a potential bridge
        bridge = junction[window_size-8:window_size+8]
        if bridge not in global_dict.phrases:
            new_phrases.append(bridge)
            
    # Update global dictionary with bridge phrases
    if new_phrases:
        global_dict.phrases.extend(new_phrases[:128]) # Limit gain

def compress_block(block: Block, global_dict: GlobalDictionary, lz_codec: LZSharedCodec) -> Block:
    if block.entropy < 2.5:
        codec_name = "RLE"
        compressed = compress_rle(block.raw_bytes)
    elif block.entropy < 5.5:
        codec_name = "LZ_SHARED"
        compressed = lz_codec.compress(block.raw_bytes)
    elif block.structure_score > 0.8:
        codec_name = "GRAMMAR"
        compressed = compress_grammar_placeholder(block.raw_bytes)
    else:
        codec_name = "RAW"
        compressed = block.raw_bytes
        
    if len(compressed) >= len(block.raw_bytes):
        block.compressed_bytes = block.raw_bytes
        block.codec_used = "RAW"
    else:
        block.compressed_bytes = compressed
        block.codec_used = codec_name
    return block

def compress_blocks_parallel(blocks: List[Block], global_dict: GlobalDictionary) -> List[Block]:
    lz_codec = LZSharedCodec(global_dict.phrases)
    return [compress_block(b, global_dict, lz_codec) for b in blocks]

def build_l1_model(blocks: List[Block]) -> bytes:
    codes = "".join(CODEC_MAP[b.codec_used] for b in blocks).encode()
    return zstd.ZstdCompressor(level=3).compress(codes)

def build_l2_model(global_dict: GlobalDictionary) -> bytes:
    dict_content = b"\n".join(global_dict.phrases)
    return zstd.ZstdCompressor(level=3).compress(dict_content)

def build_seek_index(blocks: List[Block]) -> Tuple[List[int], List[int]]:
    offsets = []
    sizes = []
    current_offset = 0
    for b in blocks:
        offsets.append(current_offset)
        size = len(b.compressed_bytes)
        sizes.append(size)
        current_offset += size
    return offsets, sizes
