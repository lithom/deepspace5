import time

import torch
from torch.utils.data import TensorDataset, DataLoader

from deepspace6.models.basic_geometry_model import GeometryModel
from deepspace6.models.basic_histogram_autoencoder import FullHistogramAutoencoder
from deepspace6.pipelines.molecule_encoder_pipeline import MoleculeEncoderPipeline


class MoleculeGeometryPipeline:
    def __init__(self, molecule_encoder_pipeline: MoleculeEncoderPipeline, geometry_model:GeometryModel, histogram_ae:FullHistogramAutoencoder, device):
        self.molecule_encoder_pipeline_dict = molecule_encoder_pipeline
        self.molecule_encoder_pipeline = molecule_encoder_pipeline['pipeline']
        self.geometry_model = geometry_model.to(device)
        self.histogram_ae   = histogram_ae.to(device)
        self.device = device

    def load_checkpoint_a(self, file_molecule_ae, file_geometry_model, file_histogram_ae):
        self.molecule_encoder_pipeline.load_checkpoint_a(file_molecule_ae)
        self.geometry_model.load_state_dict(torch.load(file_geometry_model,weights_only=True))
        self.histogram_ae.load_state_dict(torch.load(file_histogram_ae, weights_only=True))
        # checkpoint = torch.load(file, weights_only=True)
        # model.load_state_dict(checkpoint[']model_molecule_ae'])

    def save_checkpoint_a(self, file_molecule_ae, file_geometry_model, file_histogram_ae):
        self.molecule_encoder_pipeline.save_checkpoint_a(file_molecule_ae)
        torch.save(self.geometry_model.state_dict(), file_geometry_model)
        torch.save(self.histogram_ae.state_dict(), file_histogram_ae)
        #torch.save({'model_molecule_ae': self.molecule_ae.state_dict()}, file)

    def run(self, smiles_list, batch_size = 128):
        # Step 1: use molecule encoder pipeline
        mol_latents, timing_data_encoder = self.molecule_encoder_pipeline.run(smiles_list, batch_size=batch_size)

        # Step 2: Use geometry model
        # Create a TensorDataset wrapping your tensor
        dataset = TensorDataset(mol_latents)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        self.geometry_model.eval()
        geom_latents_raw = []

        time_start_torch_b = time.time()

        with torch.no_grad():
            # predict geometry
            for batch in dataloader:
                input_latents = batch[0].to(self.device)
                # Forward pass
                predicted_geom = self.geometry_model(input_latents)
                geom_latents_raw.append(predicted_geom.to("cpu"))

        geom_latents = torch.cat(geom_latents_raw)
        time_stop_torch_b = time.time()

        # Step 3: Decode to distance matrix
        geom_dataset = TensorDataset(geom_latents)
        geom_dataloader = DataLoader(geom_dataset, batch_size=batch_size, shuffle=False)
        self.histogram_ae.eval()
        hist_data_raw = []
        time_start_torch_c = time.time()
        with torch.no_grad():
            # predict geometry
            for batch in geom_dataloader:
                geom_latent = batch[0].to(self.device)
                reconstructed_hist = self.histogram_ae.decode(geom_latent)
                hist_data_raw.append(reconstructed_hist.to("cpu"))

        hist_data = torch.cat(hist_data_raw)
        time_stop_torch_c = time.time()

        timing_data = {
            'geom_model': (time_stop_torch_b-time_start_torch_b),
            'hist_ae_decode': (time_stop_torch_c - time_start_torch_c)
        }
        timing_data.update(timing_data_encoder)

        return hist_data, timing_data



