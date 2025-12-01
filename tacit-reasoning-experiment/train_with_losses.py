import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from data_gen import HierarchicalGraphGenerator, GraphConfig
from models_with_losses import TacitReasonerWithLosses, triangle_inequality_loss, witness_consistency_loss
from train import GraphDataset, evaluate

def train_model_with_aux_losses(model, train_loader, test_loader, node_to_cluster, 
                                  epochs=20, lr=0.001, name="Model",
                                  triangle_weight=0.1, consistency_weight=0.1):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training {name} with auxiliary losses...")
    print(f"  Triangle weight: {triangle_weight}, Consistency weight: {consistency_weight}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_pred_loss = 0
        total_tri_loss = 0
        total_cons_loss = 0
        
        for u, v, dist in train_loader:
            optimizer.zero_grad()
            
            # Main prediction loss
            pred = model(u, v).squeeze()
            pred_loss = criterion(pred, dist)
            
            # Triangle inequality loss (sample random third nodes)
            batch_size = u.size(0)
            w = torch.randint(0, model.num_nodes, (batch_size,))
            tri_loss = triangle_inequality_loss(model, u, v, w)
            
            # Witness consistency loss (computed once per batch)
            cons_loss = witness_consistency_loss(model, node_to_cluster)
            
            # Combined loss
            loss = pred_loss + triangle_weight * tri_loss + consistency_weight * cons_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_pred_loss += pred_loss.item()
            total_tri_loss += tri_loss.item()
            total_cons_loss += cons_loss.item()
            
        avg_loss = total_loss / len(train_loader)
        avg_pred = total_pred_loss / len(train_loader)
        avg_tri = total_tri_loss / len(train_loader)
        avg_cons = total_cons_loss / len(train_loader)
        
        if (epoch + 1) % 5 == 0:
            val_loss, val_acc = evaluate(model, test_loader)
            print(f"Epoch {epoch+1}/{epochs} | Total: {avg_loss:.4f} | Pred: {avg_pred:.4f} | Tri: {avg_tri:.4f} | Cons: {avg_cons:.4f} | Val Acc: {val_acc:.2f}%")
            
    return model

def main():
    print("\n=== Training with Auxiliary Losses ===")
    config = GraphConfig(num_clusters=5, nodes_per_cluster=20)
    gen = HierarchicalGraphGenerator(config)
    gen.generate()
    
    raw_data = gen.get_dataset(num_samples=20000)
    split = int(0.8 * len(raw_data))
    
    train_loader = DataLoader(GraphDataset(raw_data[:split]), batch_size=64, shuffle=True)
    test_loader = DataLoader(GraphDataset(raw_data[split:]), batch_size=64, shuffle=False)
    
    num_nodes = config.num_clusters * config.nodes_per_cluster
    
    # Train with auxiliary losses
    model = TacitReasonerWithLosses(num_nodes)
    train_model_with_aux_losses(model, train_loader, test_loader, gen.node_to_cluster,
                                  epochs=20, triangle_weight=0.1, consistency_weight=0.1,
                                  name="Tacit+AuxLosses")
    
    # Final evaluation
    _, acc = evaluate(model, test_loader)
    print(f"\nFinal Accuracy: {acc:.2f}%")
    
    return model, gen

if __name__ == "__main__":
    main()
