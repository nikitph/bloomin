import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from data_gen import HierarchicalGraphGenerator, GraphConfig
from models_with_losses import TacitReasonerWithLosses, triangle_inequality_loss, witness_consistency_loss
from train import GraphDataset, evaluate
from analyze_witnesses import analyze_witness_structure

def curriculum_train(model, graph_gen, max_epochs_per_stage=10, 
                     triangle_weight=0.1, consistency_weight=0.1):
    """
    Curriculum learning: Train on progressively longer paths.
    Forces the model to learn compositional structure incrementally.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("\n" + "="*60)
    print("CURRICULUM LEARNING: Progressive Path Lengths")
    print("="*60)
    
    # Generate all pairs with distances
    all_pairs = []
    nodes = list(graph_gen.graph.nodes())
    for u in nodes:
        for v in nodes:
            if u >= v:
                continue
            if v in graph_gen.shortest_paths[u]:
                dist = graph_gen.shortest_paths[u][v]
                all_pairs.append((u, v, dist))
    
    # Curriculum stages: train on max_length 1, 2, 3, 4, 5, 6, 7, 8
    for max_length in range(1, 9):
        print(f"\n--- Stage {max_length}: Training on paths length 1-{max_length} ---")
        
        # Filter data for this stage
        stage_data = [(u, v, d) for u, v, d in all_pairs if d <= max_length]
        
        if len(stage_data) == 0:
            print(f"No data for max_length {max_length}, skipping...")
            continue
        
        # Split into train/val (80/20)
        np.random.shuffle(stage_data)
        split = int(0.8 * len(stage_data))
        train_data = stage_data[:split]
        val_data = stage_data[split:]
        
        if len(val_data) == 0:
            val_data = train_data[:min(100, len(train_data))]
        
        train_loader = DataLoader(GraphDataset(train_data), batch_size=64, shuffle=True)
        val_loader = DataLoader(GraphDataset(val_data), batch_size=64, shuffle=False)
        
        print(f"  Train samples: {len(train_data)}, Val samples: {len(val_data)}")
        
        # Train for a few epochs at this stage
        for epoch in range(max_epochs_per_stage):
            model.train()
            total_loss = 0
            
            for u, v, dist in train_loader:
                optimizer.zero_grad()
                
                pred = model(u, v).squeeze()
                pred_loss = criterion(pred, dist)
                
                # Auxiliary losses
                batch_size = u.size(0)
                w = torch.randint(0, model.num_nodes, (batch_size,))
                tri_loss = triangle_inequality_loss(model, u, v, w)
                cons_loss = witness_consistency_loss(model, graph_gen.node_to_cluster)
                
                loss = pred_loss + triangle_weight * tri_loss + consistency_weight * cons_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                _, val_acc = evaluate(model, val_loader)
                print(f"    Epoch {epoch+1}/{max_epochs_per_stage} | Val Acc: {val_acc:.2f}%")
    
    print("\nCurriculum training complete!")
    return model

def test_curriculum_extrapolation():
    """
    Test if curriculum learning enables path length extrapolation.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 5: Curriculum Learning for Path Extrapolation")
    print("="*60)
    
    config = GraphConfig(num_clusters=5, nodes_per_cluster=20)
    gen = HierarchicalGraphGenerator(config)
    gen.generate()
    
    num_nodes = config.num_clusters * config.nodes_per_cluster
    
    # Train with curriculum
    model = TacitReasonerWithLosses(num_nodes)
    curriculum_train(model, gen, max_epochs_per_stage=5, 
                    triangle_weight=0.1, consistency_weight=0.1)
    
    # Test on different length ranges
    print("\n" + "="*60)
    print("TESTING EXTRAPOLATION")
    print("="*60)
    
    test_ranges = [
        (1, 2, "Very Short"),
        (3, 4, "Short (Trained)"),
        (5, 6, "Medium (Extrapolation)"),
        (7, 8, "Long (Extrapolation)")
    ]
    
    results = {}
    for min_len, max_len, label in test_ranges:
        train_data, test_data = gen.get_length_split_dataset(
            train_range=(1, 4), test_range=(min_len, max_len)
        )
        
        if len(test_data) == 0:
            print(f"{label} ({min_len}-{max_len}): No data")
            continue
        
        test_loader = DataLoader(GraphDataset(test_data), batch_size=64, shuffle=False)
        _, acc = evaluate(model, test_loader)
        results[label] = acc
        print(f"{label} ({min_len}-{max_len}): {acc:.2f}%")
    
    # Analyze witness structure
    print("\n" + "="*60)
    print("WITNESS STRUCTURE ANALYSIS")
    print("="*60)
    witness_corr = analyze_witness_structure(model, gen, 
                                             save_path="tacit-reasoning-experiment/witness_analysis_curriculum.png")
    
    return results, witness_corr

if __name__ == "__main__":
    results, witness_corr = test_curriculum_extrapolation()
    
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print("Extrapolation Results:", results)
    print("Witness Correlations:", witness_corr)
