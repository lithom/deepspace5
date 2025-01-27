import torch
import torch.nn.functional as F

def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div