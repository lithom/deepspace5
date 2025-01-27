import torch
import time
from torch.utils.data import DataLoader

from deepspace6.data.molecule_dataset import MoleculeDatasetHelper, create_dataset
from deepspace6.models.basic_autoencoder import TransformerAutoencoderWithIngress


class MoleculeEncoderPipeline:

    def __init__(self, mol_autoencoder :TransformerAutoencoderWithIngress, molecule_dataset_helper: MoleculeDatasetHelper, device):

        self.molecule_dataset_helper = molecule_dataset_helper
        self.molecule_ae = mol_autoencoder.to(device)
        self.device = device


    def create_dataset(self, smiles_list):
        dataset = create_dataset(smiles_list, self.molecule_dataset_helper.atom_embedding_parts, self.molecule_dataset_helper.bond_embedding_parts, self.molecule_dataset_helper.constants, scramble_molecules=False)
        return dataset

    def load_checkpoint_a(self, file):
        self.molecule_ae.load_state_dict(torch.load(file, weights_only=True))
        # checkpoint = torch.load(file, weights_only=True)
        # model.load_state_dict(checkpoint[']model_molecule_ae'])

    def save_checkpoint_a(self, file):
        torch.save(self.molecule_ae.state_dict(), file)
        #torch.save({'model_molecule_ae': self.molecule_ae.state_dict()}, file)


    def run(self, smiles_list, batch_size=128):
        t_start_dataset = time.time()
        dataset, dataset_helper = self.create_dataset(smiles_list)
        mol_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        t_end_dataset = time.time()

        self.molecule_ae.eval()
        molecule_latents = []

        time_dataload = 0.0
        time_mol_ae_encode = 0.0

        t_start_dataload = time.time()

        with torch.no_grad():
            # Encode molecular graph latent space
            for batch in mol_dataloader:
                t_end_dataload = time.time()
                time_dataload += (t_end_dataload-t_start_dataload)

                t_start_torch = time.time()
                atom_data = batch["atom_data"].to(self.device)
                bond_data = batch["bond_data"].to(self.device)

                # Forward pass
                result, latent = self.molecule_ae(atom_data, bond_data)
                #latent = self.molecule_ae.encode(atom_data, bond_data)

                molecule_latents.append(latent.to("cpu"))
                t_end_torch = time.time()
                time_mol_ae_encode += (t_end_torch-t_start_torch)
                t_start_dataload = time.time()

        latents = torch.cat(molecule_latents)
        t_end_a = time.time();

        timing = {
            'dataloader_creation': (t_end_dataset-t_start_dataset),
            'dataload': time_dataload,
            'mol_ae_encode': time_mol_ae_encode
        }

        return latents, timing

