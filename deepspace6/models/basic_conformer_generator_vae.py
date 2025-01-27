import torch
import torch.nn as nn
import torch.nn.functional as F

class ConformationVAE(nn.Module):
    def __init__(self, transformer_dim_a, transformer_dim_b, dim_latent):
        super().__init__()
        self.transformer_dim_a = transformer_dim_a
        self.transformer_dim_b = transformer_dim_b
        self.dim_latent = dim_latent

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(32, transformer_dim_a))

        # Encoder
        self.input_proj = nn.Linear(3, transformer_dim_a)
        self.encoder_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim_a, nhead=8), num_layers=2
        )
        self.fc_mu = nn.Linear(transformer_dim_a, dim_latent // 32)
        self.fc_logvar = nn.Linear(transformer_dim_a, dim_latent // 32)

        # Decoder
        self.decoder_fc = nn.Linear(dim_latent // 32, transformer_dim_b)
        self.decoder_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=transformer_dim_b, nhead=8), num_layers=2
        )
        self.output_proj = nn.Linear(transformer_dim_b, 32)

    def encode(self, x):
        x = self.input_proj(x) + self.positional_encoding
        x = self.encoder_transformer(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.decoder_fc(z)
        z = self.decoder_transformer(z)
        return self.output_proj(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar