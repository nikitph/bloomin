import struct
import zstandard as zstd
from typing import Optional, Tuple, List
from common import (
    CompressedFile, SeekIndex, REV_CODE_MAP, GlobalDictionary, 
    MAGIC, encode_varint, decode_varint
)
from tier1 import (
    analyze_input, build_global_dictionary, compress_blocks_parallel, 
    build_l1_model, build_l2_model, build_seek_index, boundary_repair
)
from tier2 import hyper_compress, hyper_decompress
from custom_codecs import decompress_rle, LZSharedCodec, decompress_grammar_placeholder

class Archive:
    @staticmethod
    def save(compressed_file: CompressedFile, path: str):
        """Flat Binary Layout (Step 2)"""
        with open(path, "wb") as f:
            # 1. Header
            f.write(MAGIC) # 4
            f.write(struct.pack("B", 2)) # Version 2
            f.write(struct.pack("Q", compressed_file.header["total_raw_size"])) # 8
            f.write(struct.pack("I", len(compressed_file.seek_index.block_sizes))) # 4
            
            # 2. Global Dictionary (Packed)
            dict_data = b"\x00".join(compressed_file.global_dictionary.phrases)
            f.write(struct.pack("I", len(dict_data)))
            f.write(dict_data)
            
            # 3. Models
            l1 = compressed_file.L1_model if compressed_file.L1_model else b""
            l2 = compressed_file.L2_model if compressed_file.L2_model else b""
            f.write(struct.pack("II", len(l1), len(l2)))
            f.write(l1)
            f.write(l2)
            
            # 4. Seek Index (Varint encoded)
            for size in compressed_file.seek_index.block_sizes:
                f.write(encode_varint(size))
            
            # 5. Block Data
            f.write(compressed_file.compressed_blocks_data)

    @staticmethod
    def load(path: str) -> CompressedFile:
        with open(path, "rb") as f:
            data = f.read()
        
        pos = 0
        if data[pos:pos+4] != MAGIC: raise ValueError("Invalid format")
        pos += 4
        version = data[pos]; pos += 1
        raw_size = struct.unpack("Q", data[pos:pos+8])[0]; pos += 8
        block_count = struct.unpack("I", data[pos:pos+4])[0]; pos += 4
        
        dict_size = struct.unpack("I", data[pos:pos+4])[0]; pos += 4
        dict_data = data[pos:pos+dict_size]; pos += dict_size
        phrases = dict_data.split(b"\x00") if dict_data else []
        
        l1_size, l2_size = struct.unpack("II", data[pos:pos+8]); pos += 8
        l1 = data[pos:pos+l1_size]; pos += l1_size
        l2 = data[pos:pos+l2_size]; pos += l2_size
        
        sizes = []
        for _ in range(block_count):
            val, pos = decode_varint(data, pos)
            sizes.append(val)
        
        offsets = []
        curr = 0
        for s in sizes:
            offsets.append(curr)
            curr += s
            
        blocks_data = data[pos:]
        
        return CompressedFile(
            header={"total_raw_size": raw_size, "version": version},
            global_dictionary=GlobalDictionary(phrases=phrases),
            L1_model=l1,
            L2_model=l2,
            compressed_blocks_data=blocks_data,
            seek_index=SeekIndex(block_offsets=offsets, block_sizes=sizes)
        )

def compress(data: bytes, block_size: int = 16384, hyper: bool = False) -> CompressedFile:
    if hyper:
        hyper_data = hyper_compress(data)
        return CompressedFile(
            header={"total_raw_size": len(data), "hyper": True},
            global_dictionary=GlobalDictionary(),
            L1_model=None, L2_model=None,
            compressed_blocks_data=hyper_data,
            seek_index=SeekIndex()
        )
    
    blocks = analyze_input(data, block_size)
    global_dict = build_global_dictionary(blocks)
    boundary_repair(blocks, global_dict)
    
    compressed_blocks = compress_blocks_parallel(blocks, global_dict)
    l1_model = build_l1_model(compressed_blocks)
    l2_model = build_l2_model(global_dict)
    offsets, sizes = build_seek_index(compressed_blocks)
    
    return CompressedFile(
        header={"total_raw_size": len(data)},
        global_dictionary=global_dict,
        L1_model=l1_model, L2_model=l2_model,
        compressed_blocks_data=b"".join(b.compressed_bytes for b in compressed_blocks),
        seek_index=SeekIndex(block_offsets=offsets, block_sizes=sizes)
    )

def decompress(archive: CompressedFile) -> bytes:
    if archive.header.get("hyper"):
        return hyper_decompress(archive.compressed_blocks_data)
        
    lz_codec = LZSharedCodec(archive.global_dictionary.phrases)
    l1_codes_data = zstd.ZstdDecompressor().decompress(archive.L1_model)
    codec_names = [REV_CODE_MAP[chr(c)] for c in l1_codes_data]
    
    result = bytearray()
    for i, (offset, size) in enumerate(zip(archive.seek_index.block_offsets, archive.seek_index.block_sizes)):
        block_data = archive.compressed_blocks_data[offset:offset+size]
        codec_name = codec_names[i]
        
        if codec_name == "RLE": raw = decompress_rle(block_data)
        elif codec_name == "LZ_SHARED": raw = lz_codec.decompress(block_data)
        elif codec_name == "GRAMMAR": raw = decompress_grammar_placeholder(block_data)
        else: raw = block_data
        result.extend(raw)
    return bytes(result)
