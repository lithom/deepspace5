import torch
import torch.nn as nn


class TransformerAutoencoderWithIngress(nn.Module):
    def __init__(self, seq_lengths=(32, 64), feature_dims=(244, 104), combined_dim=256, latent_dim=16, n_heads=4,
                 n_layers=2): #n_layers=2):
        super(TransformerAutoencoderWithIngress, self).__init__()

        self.seq_lengths = seq_lengths
        self.feature_dims = feature_dims
        self.combined_dim = combined_dim

        # Separate embedding layers for each input type
        self.embedding_1 = nn.Linear(feature_dims[0], combined_dim)
        self.embedding_2 = nn.Linear(feature_dims[1], combined_dim)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, sum(seq_lengths), combined_dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=combined_dim, nhead=n_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Bottleneck (latent space)
        self.bottleneck = nn.Linear(combined_dim, latent_dim)

        # Latent space expansion
        self.latent_expansion = nn.Linear(latent_dim, combined_dim)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=combined_dim, nhead=n_heads, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Separate output projection layers for reconstruction
        self.output_projection_1 = nn.Linear(combined_dim, feature_dims[0])
        self.output_projection_2 = nn.Linear(combined_dim, feature_dims[1])


    def encode(self, input_1, input_2):
        # Embed both input sequences
        embedded_1 = self.embedding_1(input_1)  # Shape: (batch_size, 32, combined_dim)
        embedded_2 = self.embedding_2(input_2)  # Shape: (batch_size, 64, combined_dim)

        # Concatenate sequences
        full_sequence = torch.cat((embedded_1, embedded_2), dim=1)  # Shape: (batch_size, 32+64, combined_dim)

        # Add positional encoding
        full_sequence += self.positional_encoding[:, :full_sequence.size(1), :]

        # Encode with Transformer
        encoded = self.encoder(full_sequence)  # Shape: (batch_size, seq_length, combined_dim)

        # Flatten and apply bottleneck
        # latent = self.bottleneck(encoded.view(encoded.size(0), -1))  # Shape: (batch_size, latent_dim)
        latent = self.bottleneck(encoded)  # Shape: (batch_size, seq_length, latent_dim)
        return latent

    def decode(self, latent):
        # Expand latent to match sequence length and latent_dim
        expanded_latent = self.latent_expansion(latent)  # Shape: (batch_size, seq_length, latent_dim)

        # Decode using Transformer
        decoded = self.decoder(expanded_latent, expanded_latent)  # Shape: (batch_size, seq_length, combined_dim)

        # Split decoded sequence back into two parts
        decoded_1 = self.output_projection_1(
            decoded[:, :self.seq_lengths[0], :])  # Shape: (batch_size, 32, feature_dims[0])
        decoded_2 = self.output_projection_2(
            decoded[:, -self.seq_lengths[1]:, :])  # Shape: (batch_size, 64, feature_dims[1])
        return decoded_1, decoded_2


    def forward(self, input_1, input_2):
        latent = self.encode(input_1, input_2)
        decoded_1, decoded_2 = self.decode(latent)
        return (decoded_1, decoded_2), latent

    def forward_old(self, input_1, input_2):
        # Embed both input sequences
        embedded_1 = self.embedding_1(input_1)  # Shape: (batch_size, 32, combined_dim)
        embedded_2 = self.embedding_2(input_2)  # Shape: (batch_size, 64, combined_dim)

        # Concatenate sequences
        full_sequence = torch.cat((embedded_1, embedded_2), dim=1)  # Shape: (batch_size, 32+64, combined_dim)

        # Add positional encoding
        full_sequence += self.positional_encoding[:, :full_sequence.size(1), :]

        # Encode with Transformer
        encoded = self.encoder(full_sequence)  # Shape: (batch_size, seq_length, combined_dim)

        # Flatten and apply bottleneck
        #latent = self.bottleneck(encoded.view(encoded.size(0), -1))  # Shape: (batch_size, latent_dim)
        latent = self.bottleneck(encoded)  # Shape: (batch_size, seq_length, latent_dim)

        # Expand latent to match sequence length and latent_dim
        expanded_latent = self.latent_expansion(latent)  # Shape: (batch_size, seq_length, latent_dim)

        # Decode using Transformer
        decoded = self.decoder(expanded_latent, expanded_latent)  # Shape: (batch_size, seq_length, combined_dim)

        # Split decoded sequence back into two parts
        decoded_1 = self.output_projection_1(
            decoded[:, :self.seq_lengths[0], :])  # Shape: (batch_size, 32, feature_dims[0])
        decoded_2 = self.output_projection_2(
            decoded[:, -self.seq_lengths[1]:, :])  # Shape: (batch_size, 64, feature_dims[1])

        return (decoded_1, decoded_2), latent