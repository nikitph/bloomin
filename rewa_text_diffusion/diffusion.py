import torch
import torch.nn.functional as F
import math

class Diffusion:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        betas_t = self.betas[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1)
        
        # Predict noise
        # t is tensor of shape [batch], model expects [batch, 1] usually or we handle it
        t_in = t.view(-1, 1).float() / self.num_timesteps # Normalize time for model
        eps_hat = model(x, t_in)
        
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * eps_hat / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.posterior_variance[t].view(-1, 1)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape):
        b = shape[0]
        # Start from pure noise
        img = torch.randn(shape, device=self.device)
        
        for i in reversed(range(0, self.num_timesteps)):
            t = torch.full((b,), i, device=self.device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)
            
        return img
