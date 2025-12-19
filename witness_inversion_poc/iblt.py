import mmh3
import random

class IBLT:
    def __init__(self, m=1024, k=3):
        self.m = m
        self.k = k
        self.cells = [[0, 0, 0] for _ in range(m)] # [count, key_sum, hash_sum]
        
    def _hashes(self, key):
        """Returns K indices and a checksum for the given key."""
        indices = []
        for i in range(self.k):
            indices.append(mmh3.hash(str(key), i) % self.m)
        checksum = mmh3.hash(str(key), self.k + 1)
        return indices, checksum

    def insert(self, key_id):
        indices, checksum = self._hashes(key_id)
        for idx in indices:
            self.cells[idx][0] += 1
            self.cells[idx][1] ^= key_id
            self.cells[idx][2] ^= checksum

    def delete(self, key_id):
        indices, checksum = self._hashes(key_id)
        for idx in indices:
            self.cells[idx][0] -= 1
            self.cells[idx][1] ^= key_id
            self.cells[idx][2] ^= checksum

    def invert(self):
        """
        Attempts to recover all keys from the table using the peeling process.
        Returns (recovered_keys, is_complete).
        """
        recovered = set()
        working_cells = [list(c) for c in self.cells]
        pure_cells = [i for i, c in enumerate(working_cells) if abs(c[0]) == 1]
        
        while pure_cells:
            idx = pure_cells.pop()
            count, key_sum, hash_sum = working_cells[idx]
            
            if count == 0: continue
            
            # Check if it's truly pure
            _, expected_checksum = self._hashes(key_sum)
            if expected_checksum != hash_sum:
                continue # Collision or partially overlapping cell
            
            # Found a key
            recovered.add(key_sum)
            
            # Peel this key from the table
            indices, checksum = self._hashes(key_sum)
            for i in indices:
                working_cells[i][0] -= 1
                working_cells[i][1] ^= key_sum
                working_cells[i][2] ^= checksum
                if abs(working_cells[i][0]) == 1:
                    pure_cells.append(i)
                    
        is_complete = all(c[0] == 0 for c in working_cells)
        return recovered, is_complete

def get_sketch_id_map(vocabulary):
    """Helper to map witness strings to deterministic integer IDs."""
    return {w: mmh3.hash(w) for w in vocabulary}

def invert_from_sketch(iblt_obj, id_to_witness):
    """Utility to map recovered IDs back to witness strings."""
    ids, complete = iblt_obj.invert()
    return {id_to_witness[i] for i in ids if i in id_to_witness}, complete
