import torch
import torch.nn as nn
import numpy as np
from torch.nn import Linear
from torch.nn.functional import softplus


class HistogramEncoderDecoder(nn.Module):
    def __init__(self, num_bins=256, n2d=16):
        super(HistogramEncoderDecoder, self).__init__()
        self.num_bins = num_bins
        self.n2d = n2d

        if num_bins >= 128:

            # Encoder: Simple feed-forward network
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_bins, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, n2d)
            )

            # Decoder: Simple feed-forward network
            self.decoder = nn.Sequential(
                nn.Linear(n2d, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, num_bins),
                nn.Unflatten(1, (num_bins,))
            )

        else:
            # Encoder: Simple feed-forward network
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_bins, n2d),
                nn.ReLU(),
                nn.Linear(n2d, n2d),
                #nn.ReLU(),
                #nn.Linear(n2d, n2d),
                #nn.ReLU(),
                #nn.Linear(n2d, n2d),
                nn.ReLU(),
                nn.Linear(n2d, n2d)
            )

            # Decoder: Simple feed-forward network
            self.decoder = nn.Sequential(
                nn.Linear(n2d, num_bins),
                nn.ReLU(),
                nn.Linear(num_bins, num_bins),
                nn.ReLU(),
                nn.Linear(num_bins, num_bins),
                nn.Unflatten(1, (num_bins,))
            )


    def encode(self, histogram):
        return self.encoder(histogram)

    def decode(self, encoded):
        return self.decoder(encoded)


class FullHistogramAutoencoder(nn.Module):
    def __init__(self, num_bins=256, n2d=16, dim_latent=4096, n_heads=4, n_layers=2):
        super(FullHistogramAutoencoder, self).__init__()

        self.num_bins = num_bins
        self.n2d = n2d
        self.dim_latent = dim_latent

        # Module for each single 2D histogram
        self.histogram_module = HistogramEncoderDecoder(num_bins=num_bins, n2d=n2d)

        self.ingress_meanvar = nn.Linear(2*32,n2d*32)

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 2*32 , 32*n2d))

        # Transformer Encoder
        self.transformer_dim = n2d * 32
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_dim, nhead=n_heads, batch_first=True)
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Bottleneck
        #self.bottleneck = nn.Linear(32 * 32 * n2d, dim_latent)
        self.bottleneck = nn.Linear( int( 2 * 32 * 32 * n2d / dim_latent ) , 1)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.transformer_dim, nhead=n_heads, batch_first=True)
        self.decoder_transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # decode meanvar
        self.decode_meanvar = nn.Linear(n2d,2)

        # Expand latent space back
        self.expand_bottleneck = nn.Linear( int(dim_latent / (32) ) , n2d*32)

    def encode(self, histograms, meanvar):
        batch_size = histograms.size(0)

        # Encode each 2D histogram independently
        encoded_histograms = torch.zeros(batch_size, 32, 32, self.n2d).to(histograms.device)
        for i in range(32):
            for j in range(32):
                histogram = histograms[:, i, j, :]  # Shape: (batch_size, num_bins)
                encoded = self.histogram_module.encode(histogram)
                # encoded_histograms.append(encoded)
                encoded_histograms[:, i, j] = encoded

        # Combine all encoded histograms
        encoded_histograms = encoded_histograms.view(batch_size, 32,
                                                     -1)  # torch.stack(encoded_histograms, dim=1)  # Shape: (batch_size, 32 , 32*n2d)
        meanvar_reshaped = self.ingress_meanvar( meanvar.view(batch_size, 32, 2*32) )
        encoded_histograms_with_meanvar = torch.cat((encoded_histograms, meanvar_reshaped),dim=1)

        # Add positional encoding
        encoded_histograms_with_meanvar += self.positional_encoding

        # Transformer Encoder
        transformer_encoded = self.encoder_transformer(encoded_histograms_with_meanvar)  # Shape: (batch_size, 32 , 32 * n2d)

        reshaped_height = int((32 * 32 * self.n2d) / self.dim_latent)
        transformer_encoded_reshaped = transformer_encoded.view(batch_size, self.dim_latent, -1)

        # Bottleneck
        # latent = self.bottleneck(transformer_encoded.view(batch_size, -1))  # Shape: (batch_size, dim_latent)
        latent = self.bottleneck(transformer_encoded_reshaped)
        return latent


    def decode(self, latent):
        batch_size = latent.size(0)

        # Expand latent space
        # expanded = self.expand_bottleneck(latent).view(batch_size, 32 * 32, n2d)  # Shape: (batch_size, 32 * 32, n2d)
        expanded = self.expand_bottleneck(latent.view(batch_size, 32, -1))  # Shape: (batch_size, 32 * 32, n2d)

        # Transformer Decoder
        transformer_decoded_raw = self.decoder_transformer(expanded,
                                                           expanded)  # Shape: (batch_size, 32 * 32, n2d)


        decoded_meanvar = self.decode_meanvar(transformer_decoded_raw.view(batch_size, 32 * 32, self.n2d))

        transformer_decoded = transformer_decoded_raw.view(batch_size, 32 * 32, self.n2d)

        # Decode each histogram independently
        final_decoded_histograms = []
        for idx in range(32 * 32):
            decoded_histogram = self.histogram_module.decode(
                transformer_decoded[:, idx, :])  # Shape: (batch_size, num_bins, num_bins)
            final_decoded_histograms.append(decoded_histogram)

        # Combine all decoded histograms
        final_decoded_histograms = torch.stack(final_decoded_histograms,
                                               dim=1)  # Shape: (batch_size, 32, 32, num_bins, num_bins)

        final_decoded_histograms = final_decoded_histograms.view(batch_size, 32, 32, self.num_bins)
        return final_decoded_histograms, decoded_meanvar.view(batch_size, 32, 32, 2)


    # (b,32,32,nbins) , (b,32,32,2)
    def forward(self, histograms, meanvar):
        latent = self.encode(histograms, meanvar)
        final_decoded_histograms = self.decode(latent)
        return final_decoded_histograms, latent

class FullHistogramAutoencoder_B(nn.Module):
    def __init__(self, num_bins=256, n2d=16, dim_latent=4096, n_heads=4, n_layers=2):
        super(FullHistogramAutoencoder_B, self).__init__()

        self.num_bins = num_bins
        self.n2d = n2d
        self.dim_latent = dim_latent

        # Module for each single 2D histogram
        self.histogram_module = HistogramEncoderDecoder(num_bins=num_bins, n2d=n2d)

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 32 * 32, n2d))

        # Transformer Encoder
        self.transformer_dim = n2d
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.transformer_dim, nhead=n_heads, batch_first=True)
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Bottleneck
        #self.bottleneck = nn.Linear(32 * 32 * n2d, dim_latent)
        self.bottleneck = nn.Linear( int( 32 * 32 * n2d / dim_latent ) , 1)

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.transformer_dim, nhead=n_heads, batch_first=True)
        self.decoder_transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)

        # Expand latent space back
        self.expand_bottleneck = nn.Linear( int(dim_latent / (32*32) ) , n2d)

    def forward(self, histograms):
        batch_size = histograms.size(0)

        # Encode each 2D histogram independently
        encoded_histograms = []
        for i in range(32):
            for j in range(32):
                histogram = histograms[:, i, j, :]  # Shape: (batch_size, num_bins)
                encoded = self.histogram_module.encode(histogram)
                encoded_histograms.append(encoded)

        # Combine all encoded histograms
        encoded_histograms = torch.stack(encoded_histograms, dim=1)  # Shape: (batch_size, 32 * 32, n2d)

        # Add positional encoding
        encoded_histograms += self.positional_encoding

        # Transformer Encoder
        transformer_encoded = self.encoder_transformer(encoded_histograms)  # Shape: (batch_size, 32 * 32, n2d)

        reshaped_height =  int( (32*32*self.n2d) / self.dim_latent )
        transformer_encoded_reshaped = transformer_encoded.view(batch_size,self.dim_latent,-1)

        # Bottleneck
        #latent = self.bottleneck(transformer_encoded.view(batch_size, -1))  # Shape: (batch_size, dim_latent)
        latent = self.bottleneck(transformer_encoded_reshaped)

        # Expand latent space
        #expanded = self.expand_bottleneck(latent).view(batch_size, 32 * 32, n2d)  # Shape: (batch_size, 32 * 32, n2d)
        expanded = self.expand_bottleneck(latent.view(batch_size, 32 * 32, -1))  # Shape: (batch_size, 32 * 32, n2d)

        # Transformer Decoder
        transformer_decoded = self.decoder_transformer(expanded,
                                                       expanded)  # Shape: (batch_size, 32 * 32, n2d)

        # Decode each histogram independently
        final_decoded_histograms = []
        for idx in range(32 * 32):
            decoded_histogram = self.histogram_module.decode(
                transformer_decoded[:, idx, :])  # Shape: (batch_size, num_bins, num_bins)
            final_decoded_histograms.append(decoded_histogram)

        # Combine all decoded histograms
        final_decoded_histograms = torch.stack(final_decoded_histograms,
                                               dim=1)  # Shape: (batch_size, 32, 32, num_bins, num_bins)

        final_decoded_histograms = final_decoded_histograms.view(batch_size, 32, 32, self.num_bins)

        return final_decoded_histograms, latent



class MeanVarianceAutoencoder(nn.Module):
    def __init__(self, num_atoms = 32, dim_latent=256, dim_transformer_a=64, dim_transformer_b=96, n_heads=4, n_layers=2):
        super(MeanVarianceAutoencoder, self).__init__()

        self.num_atoms = num_atoms
        self.dim_latent = dim_latent
        self.dim_transformer_a = dim_transformer_a
        self.dim_transformer_b = dim_transformer_b

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 32, dim_transformer_a))

        self.ingress = nn.Linear(2*num_atoms,dim_transformer_a)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim_transformer_a, nhead=n_heads, batch_first=True)
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # bottleneck
        self.bottleneck = nn.Linear( int( (self.num_atoms * self.dim_transformer_a) / self.dim_latent), 1 )

        self.expand = nn.Linear( int(self.dim_latent/self.num_atoms), self.dim_transformer_b)

        # Second Transformer Encoder
        encoder_layer_b = nn.TransformerEncoderLayer(d_model=self.dim_transformer_b, nhead=n_heads, batch_first=True)
        self.decoder_transformer = nn.TransformerEncoder(encoder_layer_b, num_layers=n_layers)

        # output
        self.output = nn.Linear(self.dim_transformer_b, self.num_atoms*2)

    def encode(self, meanvar):
        batch_size = meanvar.size(0)

        input_a = self.ingress(  meanvar.view(batch_size,self.num_atoms,2*self.num_atoms) )
        input_a += self.positional_encoding

        transformer_out = self.encoder_transformer(input_a) # (b,num_atoms,dim_transformer)

        latent = self.bottleneck(transformer_out.view( batch_size , self.dim_latent , -1 ))

        return latent

    def decode(self, latent):
        batch_size = latent.size(0)
        expanded = self.expand( latent.view(batch_size, self.num_atoms,-1) )

        transformer_out = self.decoder_transformer(expanded)
        output_data = self.output(transformer_out)
        return output_data.view(batch_size,self.num_atoms,self.num_atoms,2)

    def forward(self, meanvar):
        latent = self.encode(meanvar)
        reconstructed = self.decode(latent)
        return reconstructed, latent




class GeometryPredictor(nn.Module):
    def __init__(self, num_atoms = 32, dim_structure_latent = 16, dim_transformer_a=256, n_heads=8, n_layers=2):
        super(GeometryPredictor, self).__init__()

        self.num_atoms = num_atoms
        self.dim_structure_latent = dim_structure_latent
        self.dim_transformer_a = dim_transformer_a

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 32, dim_transformer_a))

        self.ingress_structure = nn.Linear( 3*dim_structure_latent , dim_transformer_a )
        self.output_layer = nn.Linear(dim_transformer_a,2*num_atoms)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim_transformer_a, nhead=n_heads, batch_first=True)
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, structure_latent):
        batch_size = structure_latent.size(0)

        input_b = self.ingress_structure( structure_latent.view(batch_size, self.num_atoms , 3*self.dim_structure_latent ) )
        input_b += self.positional_encoding
        transformer_out = self.encoder_transformer(input_b) # (b,num_atoms,dim_transformer)
        output_pre = self.output_layer(transformer_out)
        output = softplus(output_pre)#torch.sigmoid(output_pre)
        return output.view(batch_size, self.num_atoms, self.num_atoms, 2)



class MeanVarianceAutoencoderWithStructureInput(nn.Module):
    #NOTE: d_latent fully is (b,3*num_atoms,d_latent) !!
    def __init__(self, num_atoms = 32, dim_structure_latent = 16, dim_latent=256, dim_transformer_a=64, dim_transformer_b=96, n_heads=4, n_layers=2):
        super(MeanVarianceAutoencoderWithStructureInput, self).__init__()

        self.num_atoms = num_atoms
        self.dim_structure_latent = dim_structure_latent
        self.dim_latent = dim_latent
        self.dim_transformer_a = dim_transformer_a
        self.dim_transformer_b = dim_transformer_b

        self.dim_latent_divided_by_num_atoms = (int) ( dim_latent / num_atoms )

        # Positional Encoding
        self.positional_encoding = nn.Parameter(torch.randn(1, 2*32, dim_transformer_a))

        self.ingress = nn.Linear(2*num_atoms,dim_transformer_a)
        self.ingress_structure = nn.Linear( 3*dim_structure_latent , dim_transformer_a )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim_transformer_a, nhead=n_heads, batch_first=True)
        self.encoder_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # bottleneck
        self.bottleneck = nn.Linear( int( (2*self.num_atoms * self.dim_transformer_a) / self.dim_latent), 1 )

        self.expand = nn.Linear( self.dim_latent_divided_by_num_atoms + 3*self.dim_structure_latent , self.dim_transformer_b)

        # Second Transformer Encoder
        encoder_layer_b = nn.TransformerEncoderLayer(d_model=self.dim_transformer_b, nhead=n_heads, batch_first=True)
        self.decoder_transformer = nn.TransformerEncoder(encoder_layer_b, num_layers=n_layers)

        # output
        self.output = nn.Linear(self.dim_transformer_b, self.num_atoms*2)

    def encode(self, meanvar, structure_latent): # structure size: (b,96,d1)
        batch_size = meanvar.size(0)

        input_a = self.ingress( meanvar.view(batch_size,self.num_atoms,2*self.num_atoms) )
        input_b = self.ingress_structure( structure_latent.view(batch_size, self.num_atoms , 3*self.dim_structure_latent ) )

        input_combined = torch.cat( (input_b,input_a) , dim=1 )


        input_combined += self.positional_encoding

        transformer_out = self.encoder_transformer(input_combined) # (b,num_atoms,dim_transformer)

        latent = self.bottleneck(transformer_out.view( batch_size , self.dim_latent , -1 ))

        return latent

    def decode(self, latent, structure_latent):
        batch_size = latent.size(0)

        input_assembly_a = latent.view(batch_size, self.num_atoms, self.dim_latent_divided_by_num_atoms)
        input_assembly_b = structure_latent.view(batch_size, self.num_atoms, 3*self.dim_structure_latent)  # dim
        input_assembly = torch.cat( (input_assembly_b,input_assembly_a) , dim=2 )
        expanded = self.expand( input_assembly )

        transformer_out = self.decoder_transformer(expanded)
        output_data = self.output(transformer_out)
        return output_data.view(batch_size,self.num_atoms,self.num_atoms,2)

    def forward(self, meanvar, structure_latent):
        latent = self.encode(meanvar, structure_latent)
        reconstructed = self.decode(latent, structure_latent)
        return reconstructed, latent



# Example Usage
if __name__ == "__main__":
    num_bins = 32
    n2d = 16
    dim_latent = 4096
    batch_size = 8

    # Mock data: (batch_size, 32, 32, num_bins, num_bins)
    histograms = torch.rand(batch_size, 32, 32, num_bins, num_bins)

    # Instantiate model
    model = FullHistogramAutoencoder(num_bins=num_bins, n2d=n2d, dim_latent=dim_latent)

    # Forward pass
    reconstructed, latent = model(histograms)
    print("Reconstructed shape:", reconstructed.shape)  # Should match input shape
    print("Latent shape:", latent.shape)  # Latent representation