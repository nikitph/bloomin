"""
Subspace-REWA for BERT.

Implements the "Subspace-REWA" algorithm:
1. Identify shared semantic subspace via PCA on Task A.
2. Project representations into shared vs. task-specific components.
3. Apply REWA geometric preservation ONLY to the shared component.
4. Use layer-wise lambda schedule (stronger regularization on deeper layers).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertModel
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class SharedSubspaceGeometry:
    """Stores geometric structure of a task projected into the SHARED subspace."""
    task_name: str
    num_classes: int
    # Centroids in the SHARED subspace (projected)
    centroids: torch.Tensor  # (num_classes, hidden_dim) --> actually (num_classes, hidden_dim) but effectively rank d_s
    centroid_distances: torch.Tensor
    centroid_angles: torch.Tensor
    class_spreads: torch.Tensor


class SubspaceREWABert(nn.Module):
    """
    BERT with Subspace-REWA protection.
    
    Key features:
    - Learns shared subspace U_A from Task 1.
    - Regularizes specific layers with quadratic lambda schedule.
    - Only restricts drift in the shared subspace.
    """

    def __init__(
        self,
        num_classes: int = 4, # Task 1 is AG News (4 classes)
        model_name: str = 'bert-base-uncased',
        lambda_max: float = 10.0,
        subspace_dim: int = 64,
        layer_focus_start: int = 0 # 0 means all layers, 7 means start from layer 8 (0-indexed)
    ):
        super().__init__()
        
        self.encoder = BertModel.from_pretrained(model_name)
        self.config = self.encoder.config
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers
        
        # Classifier
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        self.num_classes = num_classes
        
        # Hyperparams
        self.lambda_max = lambda_max
        self.subspace_dim = subspace_dim
        self.layer_focus_start = layer_focus_start
        
        # State
        self.pca_components: Dict[int, torch.Tensor] = {} # layer_idx -> U matrix (D, d_s)
        self.stored_geometries: Dict[int, List[SharedSubspaceGeometry]] = {
            i: [] for i in range(self.num_layers)
        }
        self.subspace_identified = False
        
        # Pre-compute lambda schedule
        self.layer_lambdas = self._compute_lambda_schedule()
        
    def _compute_lambda_schedule(self) -> torch.Tensor:
        """
        Compute quadratic lambda schedule: lambda_l = lambda_max * (l / L)^2
        """
        lambdas = torch.zeros(self.num_layers)
        for i in range(self.num_layers):
            # 1-indexed ratio for formula: (i+1)/L
            ratio = (i + 1) / self.num_layers
            lambdas[i] = self.lambda_max * (ratio ** 2)
            
        return lambdas

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass with Subspace-REWA regularization.
        """
        # Get all hidden states
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Classifier uses the LAST layer's [CLS] token
        last_hidden_state = outputs.last_hidden_state
        cls_embedding = last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)
        
        loss = None
        rewa_loss = 0.0
        
        if labels is not None:
            # 1. Task Loss
            loss = F.cross_entropy(logits, labels)
            
            # 2. Subspace-REWA Loss (if activated)
            if self.subspace_identified and len(self.stored_geometries[0]) > 0:
                hidden_states = outputs.hidden_states[1:] # Skip embedding layer (idx 0 in output)
                # hidden_states tuple has 13 elements: embeddings + 12 layers
                # We want the 12 encoder layers.
                
                batch_rewa_loss = 0.0
                
                # Iterate layers
                for layer_idx, layer_hidden in enumerate(hidden_states):
                    if layer_idx < self.layer_focus_start:
                        continue
                        
                    # Get [CLS] for this layer
                    layer_cls = layer_hidden[:, 0, :] # (B, D)
                    
                    # Project to shared subspace
                    # U is (D, d_s), h is (B, D) -> h_s = h @ U @ U.T
                    # But we only need h @ U to compare in the subspace directly?
                    # The instruction says: mu_s = U U^T mu. 
                    # Comparing || U U^T a - U U^T b ||^2 is equivalent to || U^T a - U^T b ||^2 
                    # if U is orthonormal (which PCA components are).
                    # So we can just project to low-dim: h_low = h @ U
                    
                    if layer_idx in self.pca_components:
                        U = self.pca_components[layer_idx] # (D, d_s)
                        h_low = torch.matmul(layer_cls, U) # (B, d_s)
                        
                        # Calculate current geometry in this subspace
                        curr_geom_loss = self._compute_layer_preservation_loss(
                            h_low, labels, layer_idx
                        )
                        
                        # Apply schedule
                        layer_lambda = self.layer_lambdas[layer_idx].to(loss.device)
                        batch_rewa_loss += layer_lambda * curr_geom_loss
                
                rewa_loss = batch_rewa_loss
                loss += rewa_loss
                
        return {
            'loss': loss, 
            'logits': logits, 
            'rewa_loss': rewa_loss
        }

    def _compute_layer_preservation_loss(
        self, 
        embeddings: torch.Tensor, # (B, d_s)
        labels: torch.Tensor,
        layer_idx: int
    ) -> torch.Tensor:
        """Compute REWA loss for a specific layer in the shared subspace."""
        prev_geoms = self.stored_geometries[layer_idx]
        if not prev_geoms:
            return torch.tensor(0.0, device=embeddings.device)
            
        # 1. Compute current centroids in subspace
        # We need centroids for the current batch classes to compare structure
        # But REWA typically compares the current batch's structure relative to old tasks?
        # Actually, REWA preserves the Old Tasks' structure. 
        # But we don't have old task data.
        # Classic EWC/REWA approach without replay:
        # We want the CURRENT model's representation of OLD tasks to preserve structure.
        # But we can't see old tasks.
        # Wait, the prompt says:
        # "Sum_c || mu_c,old - mu_c,current ||^2"
        # This implies we can measure mu_c,current. 
        # If we don't have replay data, we can't measure mu_c of old classes.
        
        # Re-reading standard REWA/Imprinting:
        # Usually implies some form of replay OR we are preserving the map itself.
        # If no replay, we can't compute centroids of old classes.
        
        # However, the user prompt says:
        # "Apply REWA only to shared components... mu_c are centroids projected"
        
        # If we assume *Prototype Replay* (storing the centroids themselves), we can do this.
        # - We stored mu_c_old (the centroids of Task A).
        # - We want to ensuring that IF we passed Task A data, it would land there.
        # - Without Task A data, we typically constrain the weights or use generative replay.
        
        # BUT, there is an alternative interpretation for "Geometry Preservation" without replay:
        # We enforce that the CURRENT task's classes maintain a valid geometry *compatible* with the space.
        # That doesn't help forgetting.
        
        # Let's look at the "GeometricPreserver" in existing `rewa_bert.py`.
        # It computes `curr_centroids` from the `labels` passed in forward.
        # This means it calculates the geometry of the *current batch*.
        # And compares it to `ref_geom`.
        # This only works if `labels` contains classes from `ref_geom`.
        # BUT THIS IS CONTINUAL LEARNING. The batch contains NEW classes.
        # Comparing New Class Geometry to Old Class Geometry? That doesn't make sense directly unless
        # we assume the *structure* (invariance) should be similar (e.g. "universality").
        
        # Let's check `rewa_bert.py` again.
        # `compute_preservation_loss` compares `curr_dist_norm` to `ref_dist_norm`.
        # And it uses `min_classes`.
        # If Task A had 4 classes, and Task B has 2.
        # It compares 2 classes of B with first 2 of A.
        # This seems to be enforcing that *any* task should have similar spread/angles?
        # That's a strong "structural prior" but doesn't prevent overwriting specific memories.
        
        # OR, does the user expect us to check the *drift* of the representation space?
        # "Drift_l = E [ || h(x) - h_final(x) || ]"
        
        # The prompt equation:
        # L_REWA = lambda * Sum || mu_c,old - mu_c,current ||^2
        # This usually requires re-computing mu_c_current (centroid of class c from OLD task).
        # Since we don't have old data, we strictly can't compute mu_c_current accurately.
        
        # HOWEVER, we can use the "Centroid Anchor" approximation:
        # We assume the *weights* of the classifier for class C approximate the centroid.
        # Or we use a tiny amount of replay (exemplars).
        
        # Given the constraints and the lack of explicit "ReplayBuffer" in the prompt, there are two paths:
        # 1. We assume the user wants us to regularize the *weights* that produce these centroids?
        # 2. We simply regularize the *entire subspace* to stay close to zero-drift?
        # "mu_c,old - mu_c,current"
        
        # Let's verify standard REWA implementation details from previous interactions or standard knowledge.
        # Standard REWA often uses a small exemplar set (Witnesses).
        # The `rewa_bert.py` simply compares batch geometry... which implies it DOESN'T solve forgetting of specific classes,
        # but rather enforces "Geometric Stability" of the space.
        
        # The user's prompt: "Identify subspaces... Apply REWA only to shared components".
        # And "Sum || mu_c,old - mu_c,current ||^2"
        # If I can't calculate mu_c_current (because no data), I can't implement this term exactly without exemplars.
        
        # BUT, the `rewa_bert.py` existing code does:
        # "Compare against all stored geometries... match dimensions... min_classes".
        # This is a Structural Prior implementation.
        
        # User says: "Why centroid-only REWA is insufficient... decisions geometry is task specific".
        # They want to preserve `H_shared` geometry.
        
        # CRITICAL: For this to work for *Forgetting*, we must ground the preservation.
        # If we just preserve abstract shape, we forget the map.
        # I will implement a "Prototype Anchor" regularization on the SHARED subspace.
        # Since we don't have data, I will use the *Stored Centroids* as fixed anchors.
        # And I will regularize the **Mapping** such that if we had input X, it would map to Z.
        # Without X, we are stuck.
        
        # STARTLING REALIZATION: 
        # The user's prompt describes: "Step 1: During Task A... Collect hidden states... PCA".
        # "Step 2: Project... h(s) = U U' h"
        # "L_REWA = ... || mu_c,old - mu_c,cur ||^2 "
        
        # If we cannot compute mu_c,cur, maybe we are analyzing the *weights*?
        # No, "mu" are representations.
        
        # I will check `run_experiment.py` -> `TextDataset`. It's just raw text.
        # There is no replay mechanism in the baseline code.
        
        # OPTION A: Add a tiny replay buffer (10-20 samples per class) to compute mu_c_cur?
        # OPTION B: Use the `rewa_bert.py` logic (Structural Stability) but applied to the subspace.
        
        # The user says: "L_REWA = lambda * || mu_c,old - mu_c,current ||".
        # This implies exact centroid matching, not just relative geometry.
        # To do this without replay, we might need *Generated Replay* or just assume we have access to the old loaders for evaluation/regularization?
        # In `run_experiment.py`, `completed_tasks` contains `test_loader`. It doesn't keep `train_loader`.
        
        # I will add a `exemplar_loader` mechanism. I will store a tiny subset of Task A (e.g. 1 batch) 
        # inside the `SubspaceREWABert` to serve as "Witnesses" for the shared geometry.
        # This is standard for "Witness" based methods (which REWA comes from).
        # The user mentions "Witness" in other contexts.
        # I'll store the *inputs* for the centroids corresponding to Task A.
        
        # Wait, the prompt says "Avoid replay or architectural hacks".
        # "Avoid replay" is explicit.
        
        # If I must avoid replay, then "mu_c,current" must be estimated.
        # Perphaps we simply regularize the weights?
        # Or maybe... we preserve the *entire subspace projection*?
        # If h_shared = U' h, and we want h_shared to be stable.
        # We can penalize change in the *output of the layer* for the *current batch*?
        # i.e. "For current data X, ensure h_shared(X) doesn't move wildly from initialization?"
        # But h_shared was defined on Task A. 
        # If we change weights, h_shared(X_new) changes too.
        # Stabilizing h_shared(X_new) is "Functional Regularization" on the current data.
        # That fits "No Replay".
        # We keep a copy of the "Old Encoder" (frozen)?
        # And minimize || h_shared_old(x) - h_shared_new(x) ||^2 for x in Current Task.
        # This preserves the "Shared Function".
        
        # This is "Learning without Forgetting" (LwF) or "Knowledge Distillation" approach, but restricted to the specific subspace.
        # That makes perfect sense.
        # 1. Keep a frozen copy of the model after Task A.
        # 2. For input x in Task B:
        #    target_s = FrozenModel.layer(x) @ U
        #    current_s = CurrentModel.layer(x) @ U
        #    Loss = || target_s - current_s ||^2
        
        # The Prompt formula || mu_c,old - mu_c,current || refers to centroids.
        # But if we stabilize the function for *all* x, we stabilize the centroids.
        # And the LwF approach (distillation) is the standard non-replay way to do this.
        # I will proceed with **Subspace Distillation**:
        # Preserve the mapping h -> h_shared for the current data, using the previous model as a teacher.
        
        return torch.tensor(0.0, device=embeddings.device)
        
    def compute_subspaces(self, dataloader: DataLoader, device: str):
        """
        Run inference on Task A (or current task) to identify shared subspaces.
        Stores PCA components for each layer.
        """
        print(f"Identifying shared subspaces (dim={self.subspace_dim}) from {len(dataloader.dataset)} samples...")
        self.eval()
        self.to(device)
        
        # Store all hidden states
        # Layer -> List[Tensor]
        layer_activations = {i: [] for i in range(self.num_layers)}
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                
                # Hidden states: (embeddings, layer1, ... layer12)
                # We care about 1..12
                for i in range(self.num_layers):
                    # Get [CLS] token (B, D)
                    hidden = outputs.hidden_states[i+1][:, 0, :].cpu()
                    layer_activations[i].append(hidden)
        
        # Compute PCA per layer
        for i in range(self.num_layers):
            if i < self.layer_focus_start:
                continue
                
            X = torch.cat(layer_activations[i], dim=0) # (N, D)
            # Center
            mu = X.mean(dim=0)
            X_centered = X - mu
            
            # SVD: X = U S V^T
            # V are the principal components (eigenvectors of covariance)
            # torch.linalg.svd returns U, S, Vh
            # Vh is (D, D) or (N, D) — actually Vh is V^T. Rows are components.
            # We want top K rows of Vh.
            
            # Use low-rank SVD if N is large? N=1500, D=768. Full SVD is fine.
            if X_centered.shape[0] < X_centered.shape[1]:
                 # If fewer samples than dim, V is limited
                 _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            else:
                 _, _, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            
            # Top k components
            components = Vh[:self.subspace_dim, :] # (d_s, D)
            
            # Store as projection matrix U: (D, d_s)
            self.pca_components[i] = components.T.to(device)
            
        self.subspace_identified = True
        print(f"Subspaces identified for layers {self.layer_focus_start}-{self.num_layers-1}.")
        
        # Creates a frozen copy of the encoder for distillation reference
        self.frozen_encoder = type(self.encoder).from_pretrained(self.config._name_or_path)
        self.frozen_encoder.load_state_dict(self.encoder.state_dict())
        self.frozen_encoder.eval()
        for param in self.frozen_encoder.parameters():
            param.requires_grad = False
        self.frozen_encoder.to(device)
        print("Reference model frozen for Subspace-REWA.")

    def update_classifier(self, num_classes: int):
        """Update classifier for new task."""
        if num_classes != self.num_classes:
            device = next(self.parameters()).device
            self.classifier = nn.Linear(self.hidden_size, num_classes).to(device)
            self.num_classes = num_classes

    def forward_distillation(self, input_ids, attention_mask, labels=None):
        """
        Actual Forward with Distillation-based Subspace REWA.
        """
        device = input_ids.device
        
        # 1. Current Model Forward
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)
        
        loss = F.cross_entropy(logits, labels) if labels is not None else 0.0
        rewa_loss = torch.tensor(0.0, device=device)
        
        # 2. Subspace Regularization (Distillation from Frozen Model)
        if self.subspace_identified and labels is not None:
             with torch.no_grad():
                 frozen_outputs = self.frozen_encoder(
                     input_ids=input_ids,
                     attention_mask=attention_mask,
                     output_hidden_states=True
                 )
            
             for i in range(self.num_layers):
                 # Skip if not focusing on this layer or no PCA
                 if i < self.layer_focus_start or i not in self.pca_components:
                     continue
                
                 U = self.pca_components[i] # (D, d_s)
                 
                 # Current [CLS]
                 curr_h = outputs.hidden_states[i+1][:, 0, :] # (B, D)
                 # Target [CLS] (Frozen)
                 target_h = frozen_outputs.hidden_states[i+1][:, 0, :] # (B, D)
                 
                 # Project both
                 curr_proj = curr_h @ U # (B, d_s)
                 target_proj = target_h @ U # (B, d_s)
                 
                 # MSE Loss on projection
                 layer_loss = F.mse_loss(curr_proj, target_proj)
                 
                 # Schedule
                 lambda_l = self.layer_lambdas[i].to(device)
                 rewa_loss += lambda_l * layer_loss
                 
        total_loss = loss + rewa_loss
        
        return {
            'loss': total_loss,
            'logits': logits,
            'rewa_loss': rewa_loss
        }

    # Override standard forward to use the distillation logic
    def forward(self, input_ids, attention_mask, labels=None):
        return self.forward_distillation(input_ids, attention_mask, labels)
