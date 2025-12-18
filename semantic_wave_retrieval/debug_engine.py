import torch
import numpy as np
from semantic_wave_retrieval.engine import WaveRetrievalEngine
from semantic_wave_retrieval.utils import generate_synthetic_data

def debug():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Data
    data = generate_synthetic_data(100, 10).to(device)
    query = data[0] + torch.randn(10).to(device) * 0.1
    
    # 2. Init Engine
    engine = WaveRetrievalEngine(data, k_neighbors=5, use_cuda=torch.cuda.is_available())
    print(f"Laplacian indices: {engine.L._indices().shape}")
    print(f"Laplacian values: {engine.L._values().shape}")
    
    # 3. Wave Step
    print("\n--- Wave Step ---")
    psi, psi_t = engine.wave_step(query, T_wave=10, dt=0.1)
    print(f"Psi (max): {psi.max().item():.4f}, (min): {psi.min().item():.4f}, (mean): {psi.mean().item():.4f}")
    
    # 4. Telegrapher
    print("\n--- Telegrapher Step ---")
    u = engine.telegrapher_step(psi, psi_t, T_damp=10)
    print(f"U (max): {u.max().item():.4f}, (min): {u.min().item():.4f}")
    
    # 5. Poisson
    print("\n--- Poisson Step ---")
    phi = engine.poisson_solve(u)
    print(f"Phi (max): {phi.max().item():.4f}, (min): {phi.min().item():.4f}")
    
    # Check Top K
    print("\n--- Retrieval ---")
    # Wave: Descending test
    _, indices_desc = torch.sort(phi, descending=True)
    print(f"Top 5 (Descending Phi): {indices_desc[:5].cpu().numpy()}")
    
    # Wave: Ascending test
    _, indices_asc = torch.sort(phi, descending=False)
    print(f"Top 5 (Ascending Phi): {indices_asc[:5].cpu().numpy()}")
    
    # Heat: Largest U
    _, indices_u = torch.topk(u, 5, largest=True)
    print(f"Top 5 (Largest U): {indices_u.cpu().numpy()}")
    
if __name__ == "__main__":
    debug()
