import torch
from model import WitnessEncoder, WitnessDecoder, ReverseDiffusionModel
from diffusion import Diffusion
from utils import SimpleTokenizer
import sys

def generate(prompt=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load("rewa_model.pt", map_location=device)
    tokenizer = checkpoint['tokenizer']
    
    # Dimensions (must match train.py)
    embed_dim = 64
    hidden_dim = 128
    max_seq_len = 32
    
    encoder = WitnessEncoder(tokenizer.vocab_size, embed_dim, hidden_dim).to(device)
    decoder = WitnessDecoder(tokenizer.vocab_size, embed_dim, hidden_dim, max_seq_len).to(device)
    reverse_model = ReverseDiffusionModel(hidden_dim).to(device)
    
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    reverse_model.load_state_dict(checkpoint['reverse_model'])
    
    encoder.eval()
    decoder.eval()
    reverse_model.eval()
    
    diffusion = Diffusion(num_timesteps=100, device=device)
    
    print(f"Generating with prompt: '{prompt}'" if prompt else "Generating from pure noise...")
    
    # 1. Encode prompt if exists (Partial witness)
    z_prompt = None
    if prompt:
        tokens = tokenizer.encode(prompt, max_len=max_seq_len).unsqueeze(0).to(device)
        with torch.no_grad():
            z_prompt = encoder(tokens)
            
    # 2. Sample from noise
    # We generate a single sample
    z = diffusion.p_sample_loop(reverse_model, (1, hidden_dim))
    
    # 3. Merge witnesses (if prompt exists)
    # Simple average or interpolation
    if z_prompt is not None:
        # In a real REWA system, this would be a more complex merge operation
        # Here we just average them to "guide" the generation
        z_final = (z + z_prompt) / 2.0
    else:
        z_final = z
        
    # 4. Decode
    with torch.no_grad():
        logits = decoder(z_final)
        predicted_ids = torch.argmax(logits, dim=-1).squeeze(0)
        text = tokenizer.decode(predicted_ids)
        
    print(f"Generated Text: {text}")
    return text

if __name__ == "__main__":
    prompt = sys.argv[1] if len(sys.argv) > 1 else None
    generate(prompt)
