from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CyclicLR

from torch.utils.data import Dataset, DataLoader

import deepspace6
from deepspace6 import ds6
from deepspace6.configs.ds_constants import DeepSpaceConstants
from deepspace6.configs.ds_settings import DeepSpaceSettings
from deepspace6.embeddings.structure_embeddings import AtomTypeEmbeddingPart, VertexDistanceMapEmbeddingPart, \
    BondInfoEmbeddingPart
from deepspace6.models.basic_autoencoder import TransformerAutoencoderWithIngress
from deepspace6.models.basic_histogram_autoencoder import MeanVarianceAutoencoder, \
    MeanVarianceAutoencoderWithStructureInput, GeometryPredictor
from deepspace6.trainers.basic_molecule_encoder_trainer import LinearRampUpScheduler
from deepspace6.utils.ds6_utils import load_pickle, count_parameters
from deepspace6.utils.pytorch_utils import kl_divergence


class GeometryEncoderTrainer:
    def __init__(self, model: MeanVarianceAutoencoder):

        self.model = model

        # Create dataset and dataloader
        num_atoms = 32
        num_bins = 64
        # batch_size = 128
        batch_size = 1024
        device = "cuda"

    def train(self, dataset: Dataset, optimizer, device="cuda"):
        batch_size = 1024
        dataloader = DataLoader(dataset_inmem, batch_size=batch_size, shuffle=True)
        model = self.model.to(device)  # Use GPU if available

        loss_function = torch.nn.MSELoss()

        # Training loop
        num_epochs = 800
        for epoch in range(0, num_epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch in dataloader:
                batch_meanvar = batch[1].to(device)
                # !!! NORMALIZE MEAN AND VAR..!!!
                batch_meanvar[:,:,:,0] = batch_meanvar[:,:,:,0] / 10.0
                batch_meanvar[:,:,:,1] = batch_meanvar[:,:,:,1] * 10

                # Forward pass
                reconstructed_meanvar, latent = model(batch_meanvar)

                # Compute loss
                meanvar_loss = F.mse_loss(batch_meanvar, reconstructed_meanvar)

                # histogram_loss = torch.zeros(batch_size, num_atoms, num_atoms).to(device)

                print(f"lossMV: {meanvar_loss}")
                total_loss = meanvar_loss

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_loss += total_loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")
            if(epoch % 20 == 0):
                torch.save(model.state_dict(), f"checkpoints/model_meanvar_latent384_D_ckpt_{epoch}.ckpt")


class GeometryEncoderWithStructureLatentTrainer:
    def __init__(self, model: GeometryPredictor, model_structure: TransformerAutoencoderWithIngress, ds_constants: DeepSpaceConstants):

        self.model = model
        self.model_structure = model_structure
        self.ds_constants = ds_constants
        # Create dataset and dataloader
        num_atoms = 32
        num_bins = 64
        # batch_size = 128
        batch_size = 1024
        device = "cuda"

    def normalize_tensor(self, input_tensor):
        """
        Normalize a tensor containing means and variances.

        Parameters:
        - input_tensor: Tensor of shape [b, x, y, 2], where [:, :, :, 0] is the mean and [:, :, :, 1] is the variance.

        Returns:
        - normalized_tensor: Tensor of the same shape as input_tensor with normalized means and variances.
        """
        # Extract means and variances
        means = input_tensor[..., 0]
        variances = input_tensor[..., 1]

        # Normalize means to range [0, 1]
        means_normalized = means / self.ds_constants.GEOM_MAX_MEAN

        # Log-transform and normalize variances to range [0, 1]
        min_var = self.ds_constants.GEOM_MIN_VAR
        max_var = self.ds_constants.GEOM_MAX_VAR
        variances_normalized = (torch.log(variances) - torch.log(torch.tensor(min_var))) / (
                torch.log(torch.tensor(max_var)) - torch.log(torch.tensor(min_var))
        )

        # Combine back into a single tensor
        normalized_tensor = torch.stack((means_normalized, variances_normalized), dim=-1)

        return normalized_tensor

    def denormalize_tensor(self, normalized_tensor):
        """
        Denormalize a tensor containing normalized means and variances.

        Parameters:
        - normalized_tensor: Tensor of shape [b, x, y, 2], where [:, :, :, 0] is the normalized mean and [:, :, :, 1] is the normalized variance.

        Returns:
        - denormalized_tensor: Tensor of the same shape as normalized_tensor with denormalized means and variances.
        """
        # Extract normalized means and variances
        means_normalized = normalized_tensor[..., 0]
        variances_normalized = normalized_tensor[..., 1]

        # Denormalize means from range [0, 1] to [0, 25.0]
        means_denormalized = means_normalized * self.ds_constants.GEOM_MAX_MEAN

        # Denormalize variances from log-normalized range [0, 1] to [0.001, 15.0]
        min_var = self.ds_constants.GEOM_MIN_VAR
        max_var = self.ds_constants.GEOM_MAX_VAR

        variances_denormalized = torch.exp(
            variances_normalized * (torch.log(torch.tensor(max_var)) - torch.log(torch.tensor(min_var))) + torch.log(
                torch.tensor(min_var))
        )

        # Combine back into a single tensor
        denormalized_tensor = torch.stack((means_denormalized, variances_denormalized), dim=-1)

        return denormalized_tensor

    def compute_loss(self, batch_meanvar, predicted_meanvar, min_mean_not_zero = 0.00001):
        # this mask is needed for masking out trivial zero values
        distances_mask = torch.abs(batch_meanvar[:, :, :, 0] < min_mean_not_zero)

        batch_meanvar_normalized = self.normalize_tensor(batch_meanvar)

        # Apply the mask to mean and variance comparisons
        masked_mean_diff = (batch_meanvar_normalized[..., 0] - predicted_meanvar[..., 0]) * distances_mask
        masked_var_diff = (batch_meanvar_normalized[..., 1] - predicted_meanvar[..., 1]) * distances_mask

        # Compute MSE losses for mean and variance with masking
        meanvar_loss_a = 1.0 * torch.sum(masked_mean_diff ** 2) / distances_mask.sum()
        # meanvar_loss_b = 1 * torch.sum(masked_var_diff ** 2) / distances_mask.sum()

        # Total loss
        meanvar_loss = meanvar_loss_a #+ meanvar_loss_b

        # compute kl-divergence loss: (de-normalize..)
        #batch_meanvar2 = self.denormalize_tensor(batch_meanvar)

        predicted_meanvar_denormalized = self.denormalize_tensor(predicted_meanvar)
        loss_kldiv = kl_divergence(predicted_meanvar_denormalized, batch_meanvar, min_var=0.0001)

        return meanvar_loss, loss_kldiv



    def train(self, dataset_train_inmem: Dataset, dataset_val_inmem: Dataset, optimizer, learning_rate, device="cuda"):
        batch_size = 512#1024
        dataloader_train = DataLoader(dataset_train_inmem, batch_size=batch_size, shuffle=True)
        dataloader_val = DataLoader(dataset_val_inmem, batch_size=batch_size, shuffle=True)
        model = self.model.to(device)  # Use GPU if available
        model_ae = self.model_structure.to(device)

        model_ae.eval()
        loss_function = torch.nn.MSELoss()

        ramp_up_steps = 200
        # Create schedulers
        ramp_up_scheduler = LinearRampUpScheduler(optimizer, ramp_up_steps, learning_rate)
        cyclic_scheduler = CyclicLR(optimizer, cycle_momentum=False,
                                    base_lr=0.2 * learning_rate, max_lr=learning_rate,
                                    step_size_up=400, step_size_down=400)

        # Training loop
        num_epochs = 2000
        steps_counter = 0
        for epoch in range(0, num_epochs):
            self.model.train()
            epoch_loss = 0.0
            total_samples = 0

            for batch in dataloader_train:
                structure_input_a = batch["atom_data"].to(device)
                structure_input_b = batch["bond_data"].to(device)

                # compute latent
                structure_latent = model_ae(structure_input_a,structure_input_b)
                # forward
                predicted_meanvar = model(structure_latent[1])
                batch_meanvar = torch.stack( (batch["hist"]["mean"],batch["hist"]["variance"] ),3).to(device).float()
                meanvar_loss , loss_kldiv = self.compute_loss(batch_meanvar, predicted_meanvar)

                print(f"lossMV: {meanvar_loss}  klDiv-loss: {loss_kldiv}")
                total_loss = meanvar_loss + 0.01*loss_kldiv

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                # Update learning rate
                steps_counter = steps_counter + 1
                if steps_counter < ramp_up_steps:
                    ramp_up_scheduler.step()
                else:
                    cyclic_scheduler.step()
                # Print learning rate for demonstration
                print(f"Epoch {epoch + 1}, LR: {optimizer.param_groups[0]['lr']}")

                epoch_loss += total_loss.item()
                total_samples += len(batch["atom_data"])

            #print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader_train)}")
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / total_samples}")

            # Perform validation every epoch
            if epoch % 1 == 0:
                val_loss, eval_speed = self.validate(dataloader_val, device)
                print(f"Epoch {epoch + 1}/{num_epochs}, Validation Loss: {val_loss}, Eval Speed: {eval_speed:.2f} samples/sec")

            if(epoch % 20 == 0):
                torch.save(model.state_dict(), f"models/checkpoints/model_meanvar_polarisA_B_ckpt_{epoch}.ckpt")

    def validate(self, val_dataloader, device="cuda"):
        self.model.eval()
        val_loss = 0.0
        total_samples = 0
        start_time = time()

        with torch.no_grad():
            for batch in val_dataloader:
                structure_input_a = batch["atom_data"].to(device)
                structure_input_b = batch["bond_data"].to(device)
                structure_latent = self.model_structure(structure_input_a, structure_input_b)
                batch_meanvar = torch.stack((batch["hist"]["mean"], batch["hist"]["variance"]), 3).to(device).float()
                # Forward pass
                predicted_meanvar = self.model(structure_latent[1])
                meanvar_loss, loss_kldiv = self.compute_loss(batch_meanvar, predicted_meanvar)

                #val_loss += meanvar_loss.item() * len(batch["atom_data"])
                print(f"lossMV: {meanvar_loss}  klDiv-loss: {loss_kldiv}")
                #val_loss += meanvar_loss.item() + loss_kldiv
                total_loss = meanvar_loss + 0.01*loss_kldiv
                val_loss += total_loss.item()
                total_samples += len(batch["atom_data"])

        end_time = time()
        eval_speed = total_samples / (end_time - start_time)
        return val_loss / total_samples, eval_speed

def create_default_autoencoder_model(device="cuda"):
    constants = DeepSpaceConstants(MAX_ATOMS=32, DIST_SCALING=1.0, device="cuda")
    atom_embedding_parts = [
        AtomTypeEmbeddingPart(constants),
        VertexDistanceMapEmbeddingPart(constants)
        # VertexDistanceEmbeddingPart(constants),
        # ApproximateDistanceEmbeddingPart(constants),
        # RingStatusEmbeddingPart(constants),
        # SymmetryRankEmbeddingPart(constants) #,
        # PharmacophoreFlagsEmbeddingPart(constants),
        # Add other embedding parts as needed
    ]
    bond_embedding_parts = [
        BondInfoEmbeddingPart(constants)
    ]
    # Instantiate models
    feature_dims_atoms = sum(pi.flattened_tensor_size() for pi in atom_embedding_parts)
    feature_dims_bonds = sum(pi.flattened_tensor_size() for pi in bond_embedding_parts)

    molecule_autoencoder = TransformerAutoencoderWithIngress(feature_dims=(feature_dims_atoms, feature_dims_bonds)).to(
        device)
    molecule_autoencoder.load_state_dict(
        torch.load(f"{deepspace6.ds6.DS6_ROOT_DIR}/workflows/checkpoints/model_ckpt_91.ckpt", weights_only=True))
    return molecule_autoencoder

if __name__ == "__main__":

    model = GeometryPredictor()

    if False:
        dataset_inmem = load_pickle(f"{ds6.DS6_ROOT_DIR}/workflows/datasets/hist_data_16k_b.pkl")

    if True:
        dataset_inmem = load_pickle(f"{ds6.DS6_ROOT_DIR}/workflows/datasets/dataset_c_all_inmem_64k_01.pkl")


    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    #dataloader = DataLoader(dataset_inmem, batch_size=batch_size, shuffle=True)
    # Instantiate model
    # model = FullHistogramAutoencoder(num_bins=num_bins, n2d=16, dim_latent=8192)
    # model = FullHistogramAutoencoder(num_bins=num_bins, n2d=16, dim_latent=8192,n_layers=2)
    # model = FullHistogramAutoencoder(num_bins=num_bins, n2d=8, dim_latent=8192,n_layers=1)
    # model = FullHistogramAutoencoder(num_bins=num_bins, n2d=8, dim_latent=8192,n_layers=1)
    # model = FullHistogramAutoencoder(num_bins=num_bins, n2d=8, dim_latent=256, n_layers=1)
    #model = MeanVarianceAutoencoderWithStructureInput(dim_latent=128)



    model_autoencoder = create_default_autoencoder_model(device="cuda")

    #model_structure =
    #model.load_state_dict(torch.load(f"{ds6.DS6_ROOT_DIR}/trainers/checkpoints/model_meanvar_latent384_C_ckpt_799.ckpt", weights_only=True))

    # model.load_state_dict(torch.load(f"{ds6.DS6_ROOT_DIR}/workflows/model_histo3d_latent8192_C_ckpt_1.ckpt", weights_only=True))
    # model.load_state_dict(torch.load(f"{ds6.DS6_ROOT_DIR}/workflows/model_histo3d_latent8192_B_ckpt_99.ckpt", weights_only=True))
    # model.load_state_dict(torch.load(f"{ds6.DS6_ROOT_DIR}/workflows/checkpoints/model_histo3d_latent8192_B_p2_ckpt_99.ckpt", weights_only=True))
    count_parameters(model)
    #optimizer = optim.Adam(model.parameters(), lr=0.00025)
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.0015)

    trainer = GeometryEncoderWithStructureLatentTrainer(model,model_autoencoder)
    trainer.train(dataset_inmem, optimizer)



