import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader, random_split

import deepspace6.ds6
from deepspace6.configs.ds_constants import DeepSpaceConstants
from deepspace6.data.molecule_dataset import create_dataset_with_conformers, InMemoryDataset
from deepspace6.embeddings.structure_embeddings import AtomTypeEmbeddingPart, VertexDistanceMapEmbeddingPart, \
    BondInfoEmbeddingPart
from deepspace6.models.basic_autoencoder import TransformerAutoencoderWithIngress
from deepspace6.models.basic_geometry_model import GeometryModel
from deepspace6.models.basic_histogram_autoencoder import FullHistogramAutoencoder


def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        try:
            data = pickle.load(file)  # Load the entire pickle file
            print("Pickle file loaded successfully.")
            print(f"Data type: {type(data)}")  # Print the type of the loaded data
            return data
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            return None

# Workflow
class Workflow:
    def __init__(self, molecule_autoencoder, histogram_autoencoder, geometry_model):
        self.molecule_autoencoder = molecule_autoencoder
        self.histogram_autoencoder = histogram_autoencoder
        self.geometry_model = geometry_model
        self.device = "cuda"

    def load_data(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        smiles = data['smiles']
        histograms = data['histograms']
        return smiles, histograms

    def prepare_data(self, mol_dataloader, histogram_dataloader):
        device = self.device
        molecule_latents = []
        histogram_latents = []

        self.molecule_autoencoder.eval()
        self.histogram_autoencoder.eval()
        with torch.no_grad():
            # Encode molecular graph latent space
            for batch in mol_dataloader:
                atom_data = batch["atom_data"].to(device)
                bond_data = batch["bond_data"].to(device)

                # Forward pass
                outputs, latent = self.molecule_autoencoder(atom_data, bond_data)
                molecule_latents.append(latent.to("cpu"))


            for batch in histogram_dataloader:
                hist_data = batch["hist"]["histogram"].to(device)  # Move batch to GPU if available
                hist_data = hist_data.float()
                # Forward pass
                reconstructed, latent = self.histogram_autoencoder(hist_data)
                histogram_latents.append(latent.to("cpu"))

        return torch.cat(molecule_latents,dim=0), torch.squeeze( torch.cat( histogram_latents ,dim=0) , dim=(2) )

    def train(self, molecule_latents, histogram_latents, epochs=200, batch_size=256, learning_rate=0.0005):
        dataset = torch.utils.data.TensorDataset(molecule_latents, histogram_latents)
        #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # Create DataLoaders for training and validation

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        optimizer = torch.optim.Adam(self.geometry_model.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            self.geometry_model.train()
            train_loss = 0
            for batch in train_dataloader:
                input_latents, target_latents = batch
                input_latents, target_latents = input_latents.to('cuda'), target_latents.to('cuda')

                # Forward pass
                predicted_latents = self.geometry_model(input_latents)

                # Loss computation
                loss = loss_fn(predicted_latents, target_latents)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation loop
            self.geometry_model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_dataloader:
                    input_latents, target_latents = batch
                    input_latents, target_latents = input_latents.to('cuda'), target_latents.to('cuda')
                    predicted_latents = self.geometry_model(input_latents)
                    loss = loss_fn(predicted_latents, target_latents)
                    val_loss += loss.item()

            # Print epoch results
            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_dataloader):.4f}, Val Loss: {val_loss / len(val_dataloader):.4f}")

        print("Done!")
        torch.save(self.geometry_model.state_dict(), f"model_geom_A_ckpt_{epoch}.ckpt")

# Example workflow usage
if __name__ == "__main__":

    constants = DeepSpaceConstants(MAX_ATOMS=32, DIST_SCALING=1.0, device="cuda")

    device = 'cuda'
    num_bins = 64 # for histograms
    batch_size = 512


    atom_embedding_parts = [
        AtomTypeEmbeddingPart(constants),
        VertexDistanceMapEmbeddingPart(constants)
        #VertexDistanceEmbeddingPart(constants),
        #ApproximateDistanceEmbeddingPart(constants),
        #RingStatusEmbeddingPart(constants),
        #SymmetryRankEmbeddingPart(constants) #,
        # PharmacophoreFlagsEmbeddingPart(constants),
        # Add other embedding parts as needed
    ]
    bond_embedding_parts = [
        BondInfoEmbeddingPart(constants)
    ]
    print("Create datasets and cache in memory")



    if True:
        dataset = load_pickle('datasets/dataset_c_all_016k_01.pkl')
        dataset_inmem = load_pickle('datasets/dataset_c_all_inmem_016k_01.pkl')
        #dataset       = load_pickle('datasets/dataset_c_all_4k_01.pkl')
        #dataset_inmem = load_pickle('datasets/dataset_c_all_inmem_4k_01.pkl')

    dataset = dataset.helper # that's the dataset helper, we renamed / reorganized things..
    dataloader = DataLoader(dataset_inmem, batch_size=batch_size, shuffle=True)

    #dataset_val = create_dataset_with_conformers(smiles_val, atom_embedding_parts, bond_embedding_parts, constants)
    #dataset_val_inmem = InMemoryDataset(dataset_val)
    #dataloader_val = DataLoader(dataset_val_inmem, batch_size=batch_size, shuffle=True)

    # Instantiate models
    feature_dims_atoms = sum(pi.flattened_tensor_size() for pi in dataset.atom_embedding_parts)
    feature_dims_bonds = sum(pi.flattened_tensor_size() for pi in dataset.bond_embedding_parts)

    molecule_autoencoder = TransformerAutoencoderWithIngress(feature_dims=(feature_dims_atoms, feature_dims_bonds)).to(device)
    molecule_autoencoder.load_state_dict(torch.load(f"{deepspace6.ds6.DS6_ROOT_DIR}/workflows/checkpoints/model_ckpt_91.ckpt", weights_only=True))

    histogram_autoencoder = FullHistogramAutoencoder(num_bins=num_bins, n2d=8, dim_latent=8192,n_layers=1).to(device)
    histogram_autoencoder.load_state_dict(torch.load(f"{deepspace6.ds6.DS6_ROOT_DIR}/workflows/checkpoints/model_histo3d_latent8192_B_p2_ckpt_99.ckpt", weights_only=True))


    geometry_model = GeometryModel( input_size=(96,16) ).to(device)
    workflow = Workflow(molecule_autoencoder, histogram_autoencoder, geometry_model)

    # Load and prepare data
    #smiles, histograms = workflow.load_data("data.pkl")
    molecule_latents, histogram_latents = workflow.prepare_data(dataloader, dataloader)

    # Train the geometry model
    workflow.train(molecule_latents, histogram_latents, epochs=200)
