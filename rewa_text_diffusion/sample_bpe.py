import torch
from model import WitnessEncoder, WitnessDecoder, ReverseDiffusionModel
from diffusion import Diffusion
from utils import BPETokenizer
import sys
import os

def generate(prompt=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load model checkpoint
    model_path = os.path.join(script_dir, "rewa_model_bpe.pt")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    
    # Load tokenizer
    tokenizer_path = os.path.join(script_dir, "bpe_tokenizer.json")
    tokenizer = BPETokenizer(vocab_size=checkpoint['vocab_size'])
    tokenizer.load(tokenizer_path)
    
    # Get dimensions from checkpoint
    vocab_size = checkpoint['vocab_size']
    embed_dim = checkpoint['embed_dim']
    hidden_dim = checkpoint['hidden_dim']
    max_seq_len = checkpoint['max_seq_len']
    
    # Initialize models
    encoder = WitnessEncoder(vocab_size, embed_dim, hidden_dim).to(device)
    decoder = WitnessDecoder(vocab_size, embed_dim, hidden_dim, max_seq_len).to(device)
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
    z = diffusion.p_sample_loop(reverse_model, (1, hidden_dim))
    
    # 3. Merge witnesses (if prompt exists)
    if z_prompt is not None:
        # Weighted interpolation favoring the prompt
        alpha = 0.7  # Weight for prompt
        z_final = alpha * z_prompt + (1 - alpha) * z
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
    prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    generate(prompt)
