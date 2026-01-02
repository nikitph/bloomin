import torch
import torch.nn.functional as F
from hsgbdh.dual_model import DualStateHSGBDH

def extract_reasoning_path(G, start_idx, end_idx):
    """
    Greedy backtracking from end_idx to start_idx on Graph G.
    Returns path indices and edge weights.
    """
    path = [end_idx]
    weights = []
    current = end_idx
    
    # Safety
    max_steps = 100
    
    for _ in range(max_steps):
        if current == start_idx:
            break
            
        # Look at incoming edges to 'current'
        # G[:, current] is column of incoming weights
        incoming = G[:, current]
        
        # Don't go back to self (if loop exists) or already visited
        # Mask visited?
        
        # Find strongest predecessor
        prev = incoming.argmax().item()
        weight = incoming[prev].item()
        
        weights.append(weight)
        path.append(prev)
        current = prev
        
    return list(reversed(path)), list(reversed(weights))

def run_extraction_demo():
    print("Training Dual Model for Path Extraction...")
    n = 64
    d = 64
    model = DualStateHSGBDH(n, d)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Train on L=3
    L_train = 3
    for epoch in range(200):
        optimizer.zero_grad()
        chain = torch.randperm(n)[:L_train]
        emb = torch.eye(n, d)
        x_seq = torch.stack([emb[i] for i in chain]).unsqueeze(0)
        outputs = model(x_seq, targets=x_seq)
        
        loss = 0
        for t in range(L_train-1):
            loss += (1.0 - F.cosine_similarity(outputs[0, t], x_seq[0, t+1], dim=0))
        loss.backward()
        optimizer.step()
        
    print("Training Complete. Testing L=12 Generalization path.")
    
    # Test L=12
    L_test = 12
    chain = torch.randperm(n)[:L_test]
    emb = torch.eye(n, d)
    x_seq = torch.stack([emb[i] for i in chain]).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        model(x_seq, targets=x_seq)
        G_trans = model.last_G_transition[0]
        
        # Determine start/end neurons
        # (Assuming Identity projection for simplicity of display, 
        # but in reality E maps emb->neurons. We simply take argmax of projection)
        start_emb = x_seq[0, 0]
        end_emb = x_seq[0, -1]
        
        start_neuron = F.relu(F.layer_norm(start_emb @ model.E, (n,))).argmax().item()
        end_neuron = F.relu(F.layer_norm(end_emb @ model.E, (n,))).argmax().item()
        
        print(f"Query: Start {start_neuron} -> End {end_neuron} (Chain Length {L_test})")
        
        path, weights = extract_reasoning_path(G_trans, start_neuron, end_neuron)
        
        print("Path Found:", path)
        print("Edge Weights:", [f"{w:.2f}" for w in weights])
        
        # Validate path correctness
        # The chain ground truth is mapped to neurons
        gt_path = []
        for i in range(L_test):
             gt_path.append(F.relu(F.layer_norm(x_seq[0, i] @ model.E, (n,))).argmax().item())
             
        print("Ground Truth Path:", gt_path)
        
        # Check overlap
        correct_nodes = len(set(path) & set(gt_path))
        print(f"Overlap: {correct_nodes}/{L_test}")

if __name__ == "__main__":
    run_extraction_demo()
