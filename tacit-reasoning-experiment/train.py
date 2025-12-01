import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from data_gen import HierarchicalGraphGenerator, GraphConfig
from models import TacitReasoner, DirectModel

class GraphDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        u, v, dist = self.samples[idx]
        return torch.tensor(u), torch.tensor(v), torch.tensor(dist, dtype=torch.float32)

def train_model(model, train_loader, test_loader, epochs=20, lr=0.001, name="Model"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Training {name}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for u, v, dist in train_loader:
            optimizer.zero_grad()
            pred = model(u, v).squeeze()
            loss = criterion(pred, dist)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        if (epoch + 1) % 5 == 0:
            val_loss, val_acc = evaluate(model, test_loader)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc (tol=0.5): {val_acc:.2f}%")
            
    return model

def evaluate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for u, v, dist in loader:
            pred = model(u, v).squeeze()
            loss = criterion(pred, dist)
            total_loss += loss.item()
            
            # Accuracy within tolerance of 0.5
            diff = torch.abs(pred - dist)
            correct += (diff < 0.5).sum().item()
            total += len(dist)
            
    return total_loss / len(loader), 100 * correct / total

def main():
    # 1. Generate Data
    config = GraphConfig(num_clusters=5, nodes_per_cluster=20)
    gen = HierarchicalGraphGenerator(config)
    gen.generate()
    
    # Create dataset
    raw_data = gen.get_dataset(num_samples=20000)
    
    # Split
    split_idx = int(0.8 * len(raw_data))
    train_data = raw_data[:split_idx]
    test_data = raw_data[split_idx:]
    
    train_loader = DataLoader(GraphDataset(train_data), batch_size=64, shuffle=True)
    test_loader = DataLoader(GraphDataset(test_data), batch_size=64, shuffle=False)
    
    num_nodes = config.num_clusters * config.nodes_per_cluster
    
    # 2. Train Direct Model
    direct_model = DirectModel(num_nodes)
    train_model(direct_model, train_loader, test_loader, epochs=20, name="Direct Model")
    
    # 3. Train Tacit Reasoner
    tacit_model = TacitReasoner(num_nodes)
    train_model(tacit_model, train_loader, test_loader, epochs=20, name="Tacit Reasoner")
    
    # 4. Final Comparison
    print("\nFinal Results:")
    _, direct_acc = evaluate(direct_model, test_loader)
    _, tacit_acc = evaluate(tacit_model, test_loader)
    print(f"Direct Model Accuracy: {direct_acc:.2f}%")
    print(f"Tacit Reasoner Accuracy: {tacit_acc:.2f}%")

if __name__ == "__main__":
    main()
