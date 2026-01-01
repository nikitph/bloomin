import torch
import torch.nn as nn
import torch.nn.functional as F

class LogicHead(nn.Module):
    """
    Directional entailment checker.
    Checks if query -> key is logically valid according to learned logic matrices.
    """
    def __init__(self, d_model, num_logic_heads=3):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_logic_heads
        
        # Head 0: Is-A
        # Head 1: Has-A
        # Head 2: Causes
        self.W_logic = nn.ParameterList([
            nn.Parameter(torch.eye(d_model) + 0.01 * torch.randn(d_model, d_model))
            for _ in range(num_logic_heads)
        ])
        
        self.consistency_threshold = 0.7
        
    def forward(self, query, key, head_idx=0):
        """
        Check if query -> key is logically valid.
        """
        q_norm = F.normalize(query, dim=-1)
        k_norm = F.normalize(key, dim=-1)
        
        # Apply logic transformation
        transformed = q_norm @ self.W_logic[head_idx]
        transformed_norm = F.normalize(transformed, dim=-1)
        
        # Measure consistency
        consistency = F.cosine_similarity(transformed_norm, k_norm, dim=-1)
        
        return transformed_norm, consistency

    def verify_edge(self, x_i, x_j):
        """
        Checks all heads and returns best consistency.
        """
        max_consistency = torch.tensor(-1.0, device=x_i.device)
        best_head = -1
        
        results = []
        for h in range(self.num_heads):
            _, cons = self.forward(x_i, x_j, head_idx=h)
            results.append(cons)
            
        stacked = torch.stack(results, dim=-1)
        max_vals, best_indices = torch.max(stacked, dim=-1)
        
        return max_vals, best_indices

    def cycle_consistency_loss(self):
        """
        Enforce transitivity: W @ W approx W @ W (idempotence/transitivity checks).
        Actually the user request says: (A->B->C) should equal (A->C)
        If W represents one step, W^2 represents two steps. 
        So we want W^2 to be consistent with W if the relation is transitive.
        """
        loss = 0
        for W in self.W_logic:
            W2 = W @ W
            # Soft enforcement: W^2 should be close to W @ W ... wait W2 IS W @ W
            # The user code says: loss += torch.norm(W2 - W @ W) which is 0. 
            # Ah, maybe they meant W^2 close to W? Or W(W(x)) close to W(x)?
            # Let's re-read the prompt carefully:
            # "W2 = W @ W; loss += torch.norm(W2 - W @ W, p='fro')" -> This is literally 0.
            
            # PROBABLE INTENT: The user probably meant W should be close to W^2 (idempotent-ish for transitive closure?)
            # Or perhaps that composite relations hold.
            # But the prompt explicitly wrote `loss += torch.norm(W2 - W @ W, p='fro')`.
            # I will implement what is likely intended: W^2 approx W (transitivity).
            
            loss += torch.norm(W2 - W, p='fro')
            
        return loss

class LogicGatedGraphUpdate:
    def __init__(self, logic_head, semiring):
        self.logic = logic_head
        self.semiring = semiring
        
    def propose_and_verify(self, x_i, x_j, bdh_correlation):
        """
        Returns verified weight and edge type index
        """
        consistency, head_idx = self.logic.verify_edge(x_i, x_j)
        
        # Gate
        mask = (consistency > self.logic.consistency_threshold).float()
        verified_weight = bdh_correlation * consistency * mask
        
        return verified_weight, head_idx
