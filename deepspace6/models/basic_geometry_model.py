import torch
import torch.nn as nn

# New model for latent space mapping
class GeometryModel(nn.Module):
    def __init__(self, input_size=(96, 16), latent_dim=8192):
        super(GeometryModel, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        self.linear = nn.Linear(input_size[0] * input_size[1], latent_dim)
        self.transformer_dim = 128#256
        self.num_heads = 4
        self.num_layers = 2

        self.view_dim = (64, 128)
        #self.view_dim = (32, 256)

        # Transformer model
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_dim, nhead=self.num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

    def forward(self, x):
        # Flatten input and project to latent space
        x = x.view(x.size(0), -1)  # Flatten
        x = self.linear(x)

        # Reshape to (64, 128) for transformer
        x = x.view(x.size(0), self.view_dim[0], self.view_dim[1])
        x = self.transformer(x)

        # Flatten to final latent space
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 8192)
        return x


