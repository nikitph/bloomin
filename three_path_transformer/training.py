import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from data_generation import generate_color_dataset
from model import ThreePathTransformer
from memory import ConceptMemory

class TrainingProtocol:
    """
    Manages the training lifecycle including Wake and Sleep phases.
    """
    def __init__(self, model, memory, config, device='cpu'):
        self.model = model
        self.memory = memory
        self.config = config
        self.device = device
        
        self.optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        # Updated unpacking for new generate_color_dataset signature
        self.dataset, self.ground_truth, self.hierarchy, self.relationships = generate_color_dataset(
            n_samples=config['wake_steps_per_epoch'], 
            dim=config['embedding_dim'],
            max_depth=config.get('max_depth', 5) # Default to 5 for deep exp
        )
        
        # We need a proper tokenizer/vocabulary mapping
        # For this POC, we can map concept names to simple integer IDs
        all_names = sorted(list(self.ground_truth.keys()))
        self.vocab = {name: i for i, name in enumerate(all_names)}
        self.vocab_inv = {i: name for name, i in self.vocab.items()}
        
        # Pre-populate memory with INITIAL random embeddings?
        # NO, the model should LEARN to output embeddings.
        # But for 'sleep consolidation', we need to store what the model *thinks* concepts are.
        # So we update memory periodically during wake, or just use the model's current output.
        # PRD says sleep path loads from memory.
        # So during Wake, we should be storing the model's outputs into memory?
        # Or does memory just track the "canonical" representation?
        
    def get_token_id(self, name):
        return torch.tensor([[self.vocab[name]]], device=self.device)
        
    def train_epoch(self, epoch_idx, use_sleep=True):
        """Run one epoch (Wake + Sleep)"""
        metrics = {'loss': 0.0, 'wake_entropy': 0.0, 'sleep_entropy': 0.0}
        
        # WAKE PHASE
        self.model.train()
        total_loss = 0
        
        # Refresh dataset each epoch? Or reuse? 
        # Generating new samples prevents overfitting to specific pairs
        batch_data, _, _, _ = generate_color_dataset(
            n_samples=self.config['wake_steps_per_epoch'],
            dim=self.config['embedding_dim'],
            max_depth=self.config.get('max_depth', 5)
        )
        
        pbar = tqdm(batch_data, desc=f"Epoch {epoch_idx} Wake")
        for step, example in enumerate(pbar):
            # Decide Fast or Slow path
            # 10% Slow Path
            path = 'slow' if step % int(1.0/self.config['slow_path_frequency']) == 0 else 'fast'
            
            # Prepare inputs
            # Map input names to token IDs
            id1 = self.get_token_id(example['input1'])
            id2 = self.get_token_id(example['input2'])
            
            # Target is the ground truth distribution
            target = example['target'].to(self.device).unsqueeze(0) # [1, dim]
            
            # Forward
            self.optimizer.zero_grad()
            
            emb1 = self.model.encode(id1, path=path)
            emb2 = self.model.encode(id2, path=path)
            
            # Prediction
            pred_mix = self.model.mix(emb1, emb2, method='geometric')
            
            # Loss: MSE or KL?
            # PRD suggested MSE. KL is better for distributions.
            # But let's stick to PRD or what works. 
            # Distributions are normalized, so MSE works fine as a proxy.
            loss = nn.MSELoss()(pred_mix, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update Memory with current understanding?
            # Only occasionally to save time, or for the concepts involved
            current_target_name = example.get('target_name')
            if current_target_name:
                 # We store the PREDICTION as the memory of that concept?
                 # No, we assume the model output for the *single* concept is its representation
                 # But here we predicted a mix.
                 # Let's also run a forward pass for the target name itself to update its memory?
                 pass
                 
        metrics['loss'] = total_loss / len(batch_data)
        
        # Snapshot current model state to Memory
        self.update_memory_from_model()
        metrics['wake_entropy'] = self.memory.measure_sharpness()['mean_entropy']
        
        # SLEEP PHASE
        if use_sleep:
            self.sleep_consolidation()
            metrics['sleep_entropy'] = self.memory.measure_sharpness()['mean_entropy']
        else:
            metrics['sleep_entropy'] = metrics['wake_entropy'] # No change
            
        return metrics
        
    def update_memory_from_model(self):
        """Update ConceptMemory with current model embeddings for all concepts"""
        self.model.eval()
        with torch.no_grad():
            for name, idx in self.vocab.items():
                token_id = torch.tensor([[idx]], device=self.device)
                # Use Slow path for highest fidelity capture? Or Fast?
                # Use Slow path to capture the "best" version
                emb = self.model.encode(token_id, path='slow')
                self.memory.store(name, emb.squeeze(0), generation=self.get_generation(name))
                
    def get_generation(self, name):
        # Look up generation from the generated hierarchy map
        for depth, names in self.hierarchy.items():
            if name in names:
                return depth
        return 0 # Default fallback

    def sleep_consolidation(self):
        """Offline sharpening of all concepts"""
        # 1. Retrieve all embeddings
        all_concepts = self.memory.get_all_embeddings() # dict name -> tensor
        
        # 2. Sharpen
        sharpened = {}
        for name, emb in all_concepts.items():
            emb = emb.to(self.device)
            # Power law sharpening
            # Using the model's sharpener module or logical equivalent
            sharp = torch.pow(torch.abs(emb) + 1e-10, self.config['sharpen_power'])
            sharp = sharp / (sharp.sum() + 1e-10)
            sharpened[name] = sharp
            
        # 3. Update Memory
        self.memory.bulk_update(sharpened)
        
        # 4. (Critical) Train model to match sharpened memories?
        # The PRD implies sleep *updates* memory.
        # But if the model parameters don't change, the memory update is lost next epoch 
        # when we re-encode from the model.
        # We need to fine-tune the model on the sharpened memories! "Replay"
        
        self.train_on_memories(sharpened)
        
    def train_on_memories(self, memories):
        """Fine-tune model to output the sharpened memory embeddings"""
        self.model.train()
        # Simple loop over all concepts
        keys = list(memories.keys())
        
        # Optimization steps for consolidation
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'] * 0.1)
        
        for _ in range(50): # Small number of refinement steps
            # Batching?
            optimizer.zero_grad()
            total_loss = 0
            
            for name in keys:
                target = memories[name].to(self.device)
                token_id = self.get_token_id(name)
                
                # Output should match sharpened target
                pred = self.model.encode(token_id, path='fast')
                loss = nn.MSELoss()(pred.squeeze(0), target)
                total_loss += loss
            
            total_loss.backward()
            optimizer.step()
