import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from data_gen import HierarchicalGraphGenerator, GraphConfig
from models import TacitReasoner, DirectModel
from train import train_model, evaluate, GraphDataset

def run_test_1_length_extrapolation():
    print("\n=== Test 1: Path Length Extrapolation ===")
    config = GraphConfig(num_clusters=5, nodes_per_cluster=20)
    gen = HierarchicalGraphGenerator(config)
    gen.generate()
    
    train_data, test_data = gen.get_length_split_dataset(train_range=(1, 4), test_range=(5, 8))
    print(f"Train samples (len 1-4): {len(train_data)}")
    print(f"Test samples (len 5-8): {len(test_data)}")
    
    train_loader = DataLoader(GraphDataset(train_data), batch_size=64, shuffle=True)
    test_loader = DataLoader(GraphDataset(test_data), batch_size=64, shuffle=False)
    
    num_nodes = config.num_clusters * config.nodes_per_cluster
    
    # Direct Model
    direct = DirectModel(num_nodes)
    train_model(direct, train_loader, test_loader, epochs=15, name="Direct Model")
    _, direct_acc = evaluate(direct, test_loader)
    
    # Tacit Reasoner
    tacit = TacitReasoner(num_nodes)
    train_model(tacit, train_loader, test_loader, epochs=15, name="Tacit Reasoner")
    _, tacit_acc = evaluate(tacit, test_loader)
    
    print(f"Test 1 Results - Direct: {direct_acc:.2f}%, Tacit: {tacit_acc:.2f}%")

def run_test_2_novel_clusters():
    print("\n=== Test 2: Novel Cluster Combinations ===")
    # Need enough clusters to have a chain A-B-C
    config = GraphConfig(num_clusters=5, nodes_per_cluster=20)
    gen = HierarchicalGraphGenerator(config)
    gen.generate()
    
    # Train on 0, 1, 2 intra, plus 0-1 and 1-2. Test on 0-2.
    train_data, test_data = gen.get_cluster_split_dataset(train_clusters=[0, 1, 2], test_pair=(0, 2))
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples (Novel 0-2): {len(test_data)}")
    
    train_loader = DataLoader(GraphDataset(train_data), batch_size=64, shuffle=True)
    test_loader = DataLoader(GraphDataset(test_data), batch_size=64, shuffle=False)
    
    num_nodes = config.num_clusters * config.nodes_per_cluster
    
    # Direct Model
    direct = DirectModel(num_nodes)
    train_model(direct, train_loader, test_loader, epochs=15, name="Direct Model")
    _, direct_acc = evaluate(direct, test_loader)
    
    # Tacit Reasoner
    tacit = TacitReasoner(num_nodes)
    train_model(tacit, train_loader, test_loader, epochs=15, name="Tacit Reasoner")
    _, tacit_acc = evaluate(tacit, test_loader)
    
    print(f"Test 2 Results - Direct: {direct_acc:.2f}%, Tacit: {tacit_acc:.2f}%")

def run_test_3_capacity_scaling():
    print("\n=== Test 3: Witness Capacity Scaling ===")
    config = GraphConfig(num_clusters=5, nodes_per_cluster=20)
    gen = HierarchicalGraphGenerator(config)
    gen.generate()
    
    raw_data = gen.get_dataset(num_samples=10000)
    split = int(0.8 * len(raw_data))
    train_loader = DataLoader(GraphDataset(raw_data[:split]), batch_size=64, shuffle=True)
    test_loader = DataLoader(GraphDataset(raw_data[split:]), batch_size=64, shuffle=False)
    
    num_nodes = config.num_clusters * config.nodes_per_cluster
    
    capacities = [4, 8, 16, 32, 64]
    results = {}
    
    for cap in capacities:
        # Distribute capacity across 3 levels roughly evenly
        dims = [max(1, cap//4), max(1, cap//4), max(1, cap//2)]
        print(f"\nTesting Capacity {cap} (Dims: {dims})")
        
        tacit = TacitReasoner(num_nodes, witness_dims=dims, code_dim=cap)
        train_model(tacit, train_loader, test_loader, epochs=10, name=f"Tacit-{cap}")
        _, acc = evaluate(tacit, test_loader)
        results[cap] = acc
        
    print("\nCapacity Scaling Results:")
    for cap, acc in results.items():
        print(f"Capacity {cap}: {acc:.2f}%")

if __name__ == "__main__":
    run_test_1_length_extrapolation()
    run_test_2_novel_clusters()
    run_test_3_capacity_scaling()
