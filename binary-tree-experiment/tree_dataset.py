"""
Binary Tree Dataset Generator for Path Prediction Task

Task: Given a random binary tree of depth D and a path (sequence of L/R directions),
predict the value at the leaf node reached by following that path.

This is a challenging hierarchical reasoning task that requires:
- Understanding tree structure from sequential representation
- Following long-range dependencies (depth 20 = 20 decisions)
- Compositional reasoning through nested structure
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import random


class RandomBinaryTree:
    """Random binary tree with integer values at leaves."""
    
    def __init__(self, depth, value_range=1000):
        """
        Args:
            depth: Tree depth (depth=0 is just a leaf)
            value_range: Range of random integer values at leaves [0, value_range)
        """
        self.depth = depth
        self.value_range = value_range
        self.num_leaves = 2 ** depth
        
        # Generate random values for all leaves
        self.leaf_values = np.random.randint(0, value_range, size=self.num_leaves)
    
    def get_value_at_path(self, path):
        """
        Get value at leaf reached by following path.
        
        Args:
            path: List of 0s (left) and 1s (right), length = depth
        
        Returns:
            Integer value at that leaf
        """
        assert len(path) == self.depth, f"Path length {len(path)} != depth {self.depth}"
        
        # Convert binary path to leaf index
        leaf_idx = 0
        for direction in path:
            leaf_idx = (leaf_idx << 1) | direction
        
        return self.leaf_values[leaf_idx]
    
    def serialize(self):
        """
        Serialize tree to sequence using in-order traversal.
        Format: [LEFT_SUBTREE, VALUE, RIGHT_SUBTREE] for internal nodes
                [VALUE] for leaves
        
        Returns:
            List of tokens representing the tree structure
        """
        def serialize_node(depth, start_idx):
            if depth == 0:
                # Leaf node
                return [self.leaf_values[start_idx]]
            
            # Internal node - recursively serialize left and right subtrees
            mid = start_idx + (2 ** (depth - 1))
            left = serialize_node(depth - 1, start_idx)
            right = serialize_node(depth - 1, mid)
            
            # Use special tokens to mark structure
            # -1 = open subtree, -2 = close subtree
            return [-1] + left + [-2, -1] + right + [-2]
        
        return serialize_node(self.depth, 0)


class BinaryTreePathDataset(Dataset):
    """
    Dataset for binary tree path prediction task.
    
    Each sample consists of:
    - Tree serialization (sequence of tokens)
    - Path (sequence of L/R directions)
    - Target value (integer at leaf)
    """
    
    def __init__(self, num_samples, depth=20, value_range=1000, seed=None):
        """
        Args:
            num_samples: Number of training examples
            depth: Tree depth
            value_range: Range of leaf values
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.num_samples = num_samples
        self.depth = depth
        self.value_range = value_range
        
        # Pre-generate all trees and paths
        self.samples = []
        for _ in range(num_samples):
            tree = RandomBinaryTree(depth, value_range)
            path = [random.randint(0, 1) for _ in range(depth)]
            target = tree.get_value_at_path(path)
            tree_seq = tree.serialize()
            
            self.samples.append({
                'tree_seq': tree_seq,
                'path': path,
                'target': target
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
    def get_vocab_size(self):
        """Return vocabulary size needed for tokenization."""
        # -2, -1 for structure tokens, 0 to value_range-1 for values, 0-1 for path
        return self.value_range + 3  # -2, -1, 0, 1, ..., value_range-1
    
    def get_max_seq_length(self):
        """Return maximum sequence length (tree + path)."""
        # Tree serialization length for depth D:
        # T(0) = 1 (just leaf)
        # T(D) = 2 + T(D-1) + 2 + T(D-1) = 4 + 2*T(D-1)
        # Solution: T(D) = 2^(D+2) - 3
        tree_len = 2 ** (self.depth + 2) - 3
        path_len = self.depth
        return tree_len + path_len + 2  # +2 for separator tokens


def collate_batch(batch):
    """
    Collate batch of samples into padded tensors.
    
    Returns:
        input_ids: [B, max_len] - tree sequence + separator + path
        targets: [B] - target values
        attention_mask: [B, max_len] - mask for padding
    """
    batch_size = len(batch)
    
    # Find max length in this batch
    max_len = max(len(sample['tree_seq']) + len(sample['path']) + 1 for sample in batch)
    
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    targets = torch.zeros(batch_size, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    for i, sample in enumerate(batch):
        tree_seq = sample['tree_seq']
        path = sample['path']
        
        # Concatenate: [tree_seq, SEP=-3, path]
        # Offset all tokens to be non-negative: add 3
        seq = [t + 3 for t in tree_seq] + [0] + [p + 1 for p in path]  # SEP=0, path: 1=L, 2=R
        
        seq_len = len(seq)
        input_ids[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
        targets[i] = sample['target']
        attention_mask[i, :seq_len] = True
    
    return {
        'input_ids': input_ids,
        'targets': targets,
        'attention_mask': attention_mask
    }


if __name__ == '__main__':
    # Test the dataset
    print("Testing Binary Tree Path Dataset...")
    
    # Create small dataset for testing
    dataset = BinaryTreePathDataset(num_samples=10, depth=3, value_range=100, seed=42)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocab size: {dataset.get_vocab_size()}")
    print(f"Max sequence length: {dataset.get_max_seq_length()}")
    
    # Test a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Tree sequence length: {len(sample['tree_seq'])}")
    print(f"  Path: {sample['path']}")
    print(f"  Target value: {sample['target']}")
    
    # Test batch collation
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_batch)
    batch = next(iter(loader))
    
    print(f"\nBatch shapes:")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  targets: {batch['targets'].shape}")
    print(f"  attention_mask: {batch['attention_mask'].shape}")
    
    print("\n✓ Dataset test passed!")
