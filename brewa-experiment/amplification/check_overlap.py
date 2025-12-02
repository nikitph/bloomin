
import sys
import os
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_20newsgroups import load_and_embed_data, split_categories

def check_overlap():
    embeddings, labels, target_names = load_and_embed_data()
    seen_emb, seen_labels, unseen_emb, unseen_labels = split_categories(
        embeddings, labels, target_names, n_unseen=5
    )
    
    seen_classes = torch.unique(seen_labels).cpu().numpy()
    unseen_classes = torch.unique(unseen_labels).cpu().numpy()
    
    intersection = np.intersect1d(seen_classes, unseen_classes)
    
    print(f"Seen classes: {seen_classes}")
    print(f"Unseen classes: {unseen_classes}")
    print(f"Intersection: {intersection}")
    
    if len(intersection) == 0:
        print("CONFIRMED: Sets are disjoint.")
    else:
        print("SURPRISE: Sets overlap.")

if __name__ == "__main__":
    check_overlap()
