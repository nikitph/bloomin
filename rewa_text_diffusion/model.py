import torch
import torch.nn as nn
import torch.nn.functional as F

class WitnessEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Simple transformer encoder or MLP for demonstration
        self.encoder = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.proj_to_witness = nn.Linear(hidden_dim, hidden_dim)

    def rewa_witness_transform(self, embeddings):
        """
        Maps semantic similarity into witness overlap structure.
        """
        # Pairwise cosine similarity
        # embeddings: [batch, seq_len, dim]
        norm_embeddings = F.normalize(embeddings, p=2, dim=-1)
        similarity = torch.bmm(norm_embeddings, norm_embeddings.transpose(1, 2)) # [batch, seq, seq]
        
        # Convert similarity -> witness overlap
        # Using softplus as per pseudocode
        witness_overlap = F.softplus(similarity)
        
        # Collapse into a single witness vector per sequence
        # We can aggregate the overlap matrix or project it.
        # Here we project the weighted sum of embeddings based on overlap
        # This is a simplified interpretation of "linear_project(W)" from pseudocode
        # effectively an attention mechanism where W is the attention map
        
        # Flatten or pool. Let's do a simple weighted pool for now to keep dims consistent
        # [batch, seq, seq] @ [batch, seq, dim] -> [batch, seq, dim]
        weighted_features = torch.bmm(witness_overlap, embeddings)
        
        # Aggregate to single vector [batch, dim]
        witness_vector = weighted_features.mean(dim=1)
        
        return witness_vector

    def forward(self, tokens):
        # tokens: [batch, seq_len]
        embeds = self.embedding(tokens) # [batch, seq, dim]
        hidden = self.encoder(embeds)
        
        # Apply REWA transform
        witness_vector = self.rewa_witness_transform(hidden)
        
        return self.proj_to_witness(witness_vector)

class WitnessDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, max_seq_len=16):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.proj_from_witness = nn.Linear(hidden_dim, hidden_dim * max_seq_len)
        self.hidden_dim = hidden_dim
        
        # Simple Transformer Decoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, witness_vector):
        # witness_vector: [batch, hidden_dim]
        batch_size = witness_vector.size(0)
        
        # Project back to sequence length
        x = self.proj_from_witness(witness_vector)
        x = x.view(batch_size, self.max_seq_len, self.hidden_dim)
        
        # Refine with transformer
        x = self.transformer(x)
        
        # Logits
        logits = self.head(x)
        return logits

class ReverseDiffusionModel(nn.Module):
    def __init__(self, input_dim, time_embed_dim=32):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x, t):
        # x: [batch, dim]
        # t: [batch, 1]
        t_emb = self.time_mlp(t)
        x_input = torch.cat([x, t_emb], dim=-1)
        return self.net(x_input)
