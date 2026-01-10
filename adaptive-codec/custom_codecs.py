import zstandard as zstd
from typing import List, Tuple

def compress_rle(data: bytes) -> bytes:
    if not data: return b""
    result = bytearray()
    i = 0
    while i < len(data):
        count = 1
        while i + count < len(data) and data[i + count] == data[i] and count < 255:
            count += 1
        result.append(count)
        result.append(data[i])
        i += count
    return bytes(result)

def decompress_rle(data: bytes) -> bytes:
    result = bytearray()
    for i in range(0, len(data), 2):
        count = data[i]
        char = data[i+1:i+2]
        result.extend(char * count)
    return bytes(result)

class LZSharedCodec:
    def __init__(self, phrases: List[bytes]):
        self.dict_data = b"".join(phrases)
        if self.dict_data:
            self.dict = zstd.ZstdCompressionDict(self.dict_data)
            self.cctx = zstd.ZstdCompressor(dict_data=self.dict)
            self.dctx = zstd.ZstdDecompressor(dict_data=self.dict)
        else:
            self.cctx = zstd.ZstdCompressor()
            self.dctx = zstd.ZstdDecompressor()
    def compress(self, data: bytes) -> bytes:
        return self.cctx.compress(data)
    def decompress(self, data: bytes) -> bytes:
        return self.dctx.decompress(data)

def decompress_grammar_placeholder(data: bytes) -> bytes:
    return zstd.ZstdDecompressor().decompress(data)

def compress_grammar_placeholder(data: bytes) -> bytes:
    return zstd.ZstdCompressor(level=10).compress(data)
