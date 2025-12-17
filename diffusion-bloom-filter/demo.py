
import numpy as np
import matplotlib.pyplot as plt
from .dbf import DiffusionBloomFilterFloat, DiffusionBloomFilterBitset

def generate_cluster(center, count, spread=0.1):
    return center + np.random.normal(0, spread, (count, len(center)))

def main():
    dim = 256
    dbf_size = 1024
    sigma = 30.0
    
    # 1. Setup
    print("Initializing Diffusion Bloom Filters...")
    dbf_float = DiffusionBloomFilterFloat(dbf_size, dim, sigma=sigma)
    dbf_bit = DiffusionBloomFilterBitset(dbf_size, dim, sigma=sigma)
    
    # 2. Data Generation
    # Concept A: "Cats"
    concept_a = np.random.randn(dim)
    concept_a /= np.linalg.norm(concept_a)
    
    # Neighbors of A: "Kitten", "Feline"
    neighbors_a = generate_cluster(concept_a, 5, spread=0.01)
    
    # Concept B: "Spaceships" (Orthogonal/Far)
    concept_b = np.random.randn(dim)
    concept_b /= np.linalg.norm(concept_b)
    
    neighbors_b = generate_cluster(concept_b, 5, spread=0.01)
    
    # 3. Insertion (Only inserting Concept A)
    print("Inserting Concept A (and a slight variation) into DBF...")
    dbf_float.insert(concept_a)
    dbf_bit.insert(concept_a)
    
    # 4. Querying
    print("\n--- Query Results (Float DBF) ---")
    
    heat_a = dbf_float.query(concept_a)
    print(f"Concept A (Exact): {heat_a:.4f} (Should be HOT ~1.0)")
    
    for i, n in enumerate(neighbors_a):
        heat = dbf_float.query(n)
        dist = np.linalg.norm(concept_a - n)
        print(f"Neighbor A[{i}] (dist={dist:.2f}): {heat:.4f} (Should be WARM)")
        
    heat_b = dbf_float.query(concept_b)
    print(f"Concept B (Far): {heat_b:.4f} (Should be COLD ~0.0)")

    print("\n--- Query Results (Bitset DBF) ---")
    
    dens_a = dbf_bit.query(concept_a)
    print(f"Concept A (Exact): {dens_a:.4f}")
    
    for i, n in enumerate(neighbors_a):
        dens = dbf_bit.query(n)
        dist = np.linalg.norm(concept_a - n)
        print(f"Neighbor A[{i}] (dist={dist:.2f}): {dens:.4f}")
        
    dens_b = dbf_bit.query(concept_b)
    print(f"Concept B (Far): {dens_b:.4f}")

    # 5. Visualization (Optional - generating plot)
    try:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(2, 1, 1)
        plt.plot(dbf_float.array)
        plt.title("Float DBF Heat Map")
        plt.ylabel("Intensity")
        
        plt.subplot(2, 1, 2)
        # For bitset, showing the bits directly or smoothed
        plt.plot(dbf_bit.array, alpha=0.5, label='Bits')
        # Smoothing for viz
        window = int(sigma)
        smoothed = np.convolve(dbf_bit.array, np.ones(window)/window, mode='same')
        plt.plot(smoothed, color='red', label='Density')
        plt.title("Bitset DBF (Dithered)")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("dbf_visualization.png")
        print("\nSaved visualization to dbf_visualization.png")
    except Exception as e:
        print(f"Could not save plot: {e}")

if __name__ == "__main__":
    main()
