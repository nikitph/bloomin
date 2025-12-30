# Model Card: GPT-2 Small Shadow (Shadow Theory Decomposition)

## Architecture Interpretability

| Layer | Parent Structure | Semantic Meaning |
|-------|------------------|------------------|
| 0 | Metric tensor (g_ij) operator on Euclidean manifold | Potential energy (u) operator on Euclidean manifold, Gradient flow (ẋ = -∇u) operator on Euclidean manifold, Symmetry detector for Permutation (S_n) (dim=768), Symmetry detector for Translation (Z) (dim=768) |
| 1 | Metric tensor (g_ij) operator on Euclidean manifold | Potential energy (u) operator on Euclidean manifold, Gradient flow (ẋ = -∇u) operator on Euclidean manifold, Symmetry detector for Permutation (S_n) (dim=768), Symmetry detector for Translation (Z) (dim=768) |
| 2 | Metric tensor (g_ij) operator on Euclidean manifold | Potential energy (u) operator on Euclidean manifold, Gradient flow (ẋ = -∇u) operator on Euclidean manifold, Symmetry detector for Permutation (S_n) (dim=768), Symmetry detector for Translation (Z) (dim=768) |
| 3 | Metric tensor (g_ij) operator on Euclidean manifold | Potential energy (u) operator on Euclidean manifold, Gradient flow (ẋ = -∇u) operator on Euclidean manifold, Symmetry detector for Pattern_Detector (S_n) (dim=384), Symmetry detector for Copy_Operator (Z) (dim=384) |
| 4 | Metric tensor (g_ij) operator on Euclidean manifold | Potential energy (u) operator on Euclidean manifold, Gradient flow (ẋ = -∇u) operator on Euclidean manifold, Symmetry detector for Pattern_Detector (S_n) (dim=384), Symmetry detector for Copy_Operator (Z) (dim=384) |
| 5 | Metric tensor (g_ij) operator on Euclidean manifold | Potential energy (u) operator on Euclidean manifold, Gradient flow (ẋ = -∇u) operator on Euclidean manifold, Symmetry detector for Pattern_Detector (S_n) (dim=384), Symmetry detector for Copy_Operator (Z) (dim=384) |
| 6 | Metric tensor (g_ij) operator on Euclidean manifold | Potential energy (u) operator on Euclidean manifold, Gradient flow (ẋ = -∇u) operator on Euclidean manifold, Symmetry detector for Pattern_Detector (S_n) (dim=384), Symmetry detector for Copy_Operator (Z) (dim=384) |
| 7 | Metric tensor (g_ij) operator on Euclidean manifold | Potential energy (u) operator on Euclidean manifold, Gradient flow (ẋ = -∇u) operator on Euclidean manifold, Symmetry detector for Pattern_Detector (S_n) (dim=384), Symmetry detector for Copy_Operator (Z) (dim=384) |
| 8 | Metric tensor (g_ij) operator on Euclidean manifold | Potential energy (u) operator on Euclidean manifold, Gradient flow (ẋ = -∇u) operator on Euclidean manifold, Symmetry detector for Pattern_Detector (S_n) (dim=384), Symmetry detector for Copy_Operator (Z) (dim=384) |
| 9 | Metric tensor (g_ij) operator on Euclidean manifold | Potential energy (u) operator on Euclidean manifold, Gradient flow (ẋ = -∇u) operator on Euclidean manifold, Symmetry detector for Pattern_Detector (S_n) (dim=384), Symmetry detector for Copy_Operator (Z) (dim=384) |
| 10 | Metric tensor (g_ij) operator on Euclidean manifold | Potential energy (u) operator on Euclidean manifold, Gradient flow (ẋ = -∇u) operator on Euclidean manifold, RG Coarse-graining: Semantic fixed point at scale 0.05 |
| 11 | Metric tensor (g_ij) operator on Euclidean manifold | Potential energy (u) operator on Euclidean manifold, Gradient flow (ẋ = -∇u) operator on Euclidean manifold, RG Coarse-graining: Semantic fixed point at scale 0.05 |

## Discovered Circuits

- **Induction Heads**: Detected (S_n \otimes Z symmetry)
- **Semantic Flows**: Detected (Gradient flow on Semantic Manifold)
