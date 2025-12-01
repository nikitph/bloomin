import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_gen import HierarchicalGraphGenerator, GraphConfig
from models_with_losses import TacitReasonerWithLosses, triangle_inequality_loss, witness_consistency_loss
from train import GraphDataset, evaluate
from train_with_losses import train_model_with_aux_losses

def run_test_1_with_aux_losses():
    print("\n" + "="*60)
    print("Test 1: Path Length Extrapolation (WITH Auxiliary Losses)")
    print("="*60)
    
    config = GraphConfig(num_clusters=5, nodes_per_cluster=20)
    gen = HierarchicalGraphGenerator(config)
    gen.generate()
    
    train_data, test_data = gen.get_length_split_dataset(train_range=(1, 4), test_range=(5, 8))
    print(f"Train samples (len 1-4): {len(train_data)}")
    print(f"Test samples (len 5-8): {len(test_data)}")
    
    train_loader = DataLoader(GraphDataset(train_data), batch_size=64, shuffle=True)
    test_loader = DataLoader(GraphDataset(test_data), batch_size=64, shuffle=False)
    
    num_nodes = config.num_clusters * config.nodes_per_cluster
    
    # Train with auxiliary losses
    model = TacitReasonerWithLosses(num_nodes)
    train_model_with_aux_losses(model, train_loader, test_loader, gen.node_to_cluster,
                                  epochs=20, triangle_weight=0.1, consistency_weight=0.1,
                                  name="Tacit+AuxLosses")
    
    _, acc = evaluate(model, test_loader)
    print(f"\n>>> Test 1 Result (WITH Aux Losses): {acc:.2f}%")
    return acc

def run_test_2_with_aux_losses():
    print("\n" + "="*60)
    print("Test 2: Novel Cluster Combinations (WITH Auxiliary Losses)")
    print("="*60)
    
    config = GraphConfig(num_clusters=5, nodes_per_cluster=20)
    gen = HierarchicalGraphGenerator(config)
    gen.generate()
    
    train_data, test_data = gen.get_cluster_split_dataset(train_clusters=[0, 1, 2], test_pair=(0, 2))
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples (Novel 0-2): {len(test_data)}")
    
    train_loader = DataLoader(GraphDataset(train_data), batch_size=64, shuffle=True)
    test_loader = DataLoader(GraphDataset(test_data), batch_size=64, shuffle=False)
    
    num_nodes = config.num_clusters * config.nodes_per_cluster
    
    # Train with auxiliary losses
    model = TacitReasonerWithLosses(num_nodes)
    train_model_with_aux_losses(model, train_loader, test_loader, gen.node_to_cluster,
                                  epochs=20, triangle_weight=0.1, consistency_weight=0.1,
                                  name="Tacit+AuxLosses")
    
    _, acc = evaluate(model, test_loader)
    print(f"\n>>> Test 2 Result (WITH Aux Losses): {acc:.2f}%")
    return acc

def run_test_3_with_aux_losses():
    print("\n" + "="*60)
    print("Test 3: Capacity Scaling (WITH Auxiliary Losses)")
    print("="*60)
    
    config = GraphConfig(num_clusters=5, nodes_per_cluster=20)
    gen = HierarchicalGraphGenerator(config)
    gen.generate()
    
    raw_data = gen.get_dataset(num_samples=10000)
    split = int(0.8 * len(raw_data))
    
    train_loader = DataLoader(GraphDataset(raw_data[:split]), batch_size=64, shuffle=True)
    test_loader = DataLoader(GraphDataset(raw_data[split:]), batch_size=64, shuffle=False)
    
    num_nodes = config.num_clusters * config.nodes_per_cluster
    
    capacities = [8, 16, 32]
    results = {}
    
    for cap in capacities:
        dims = [max(1, cap//4), max(1, cap//4), max(1, cap//2)]
        print(f"\nTesting Capacity {cap} (Dims: {dims})")
        
        model = TacitReasonerWithLosses(num_nodes, witness_dims=dims, code_dim=cap)
        train_model_with_aux_losses(model, train_loader, test_loader, gen.node_to_cluster,
                                      epochs=15, triangle_weight=0.1, consistency_weight=0.1,
                                      name=f"Tacit+Aux-{cap}")
        _, acc = evaluate(model, test_loader)
        results[cap] = acc
        
    print("\n>>> Capacity Scaling Results (WITH Aux Losses):")
    for cap, acc in results.items():
        print(f"  Capacity {cap}: {acc:.2f}%")
    
    return results

if __name__ == "__main__":
    print("\n" + "="*60)
    print("EXPERIMENT 4: Auxiliary Losses for Structural Learning")
    print("="*60)
    
    results = {}
    
    # Run all three tests with auxiliary losses
    results['test1'] = run_test_1_with_aux_losses()
    results['test2'] = run_test_2_with_aux_losses()
    results['test3'] = run_test_3_with_aux_losses()
    
    print("\n" + "="*60)
    print("SUMMARY OF RESULTS")
    print("="*60)
    print(f"Test 1 (Path Extrapolation): {results['test1']:.2f}%")
    print(f"Test 2 (Novel Clusters): {results['test2']:.2f}%")
    print(f"Test 3 (Capacity Scaling): {results['test3']}")
    print("\nCompare these to baseline (without aux losses):")
    print("  Test 1 Baseline: Direct 5.60%, Tacit 0.00%")
    print("  Test 2 Baseline: Direct 27.25%, Tacit 18.75%")
