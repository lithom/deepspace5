import random

import torch
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader

from deepspace6.configs.base_experiment_config import BaseExperimentConfig, BaseTrainConfig
from deepspace6.configs.ds_constants import DeepSpaceConstants
from deepspace6.configs.ds_settings import DeepSpaceSettings
from deepspace6.data.molecule_dataset import MoleculeDatasetHelper, create_dataset, InMemoryDataset
from deepspace6.embeddings.structure_embeddings import AtomTypeEmbeddingPart, VertexDistanceMapEmbeddingPart, \
    BondInfoEmbeddingPart
from deepspace6.models.basic_autoencoder import TransformerAutoencoderWithIngress
from deepspace6.utils.ds6_utils import count_parameters
from deepspace6.utils.smiles_utils import filter_smiles




class LinearRampUpScheduler(_LRScheduler):
    def __init__(self, optimizer, ramp_up_steps, base_lr, last_epoch=-1):
        self.ramp_up_steps = ramp_up_steps
        self.base_lr = base_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch >= self.ramp_up_steps:
            return [self.base_lr for _ in self.optimizer.param_groups]
        scale = (self.last_epoch + 1) / self.ramp_up_steps
        return [scale * self.base_lr for _ in self.optimizer.param_groups]


class MoleculeEncoderTrainer:
    def __init__(self, molecule_dataset_helper: MoleculeDatasetHelper, train_config:BaseTrainConfig, ds_settings: DeepSpaceSettings):
        self.ds_settings = ds_settings
        self.molecule_dataset_helper = molecule_dataset_helper
        self.train_config = train_config
        #self.device = self.train_config.device

        # Scheduler parameters
        self.ramp_up_steps = 100
        self.cosine_total_steps = 1000
        self.cosine_base_lr = self.train_config.LEARNING_RATE

    def loss_fn(self, atom_data, bond_data, atom_output, bond_output, batch):
        dataset_helper = self.molecule_dataset_helper
        device = self.train_config.device
        atom_parts = dataset_helper.atom_embedding_parts
        bond_parts = dataset_helper.bond_embedding_parts

        # Compute losses for atom and bond embeddings
        atom_loss = torch.zeros(atom_data.size()[0], atom_data.size()[1]).to(device)
        bond_loss = torch.zeros(bond_data.size()[0], bond_data.size()[1]).to(device)

        for zi, part in enumerate(atom_parts):
            atom_mask = batch["atom_mask"][zi].to(device)
            part_name = part.__class__.__name__
            atom_target = dataset_helper.prepare_for_loss(atom_data, batch["atom_metadata"], part_name)
            atom_output_part = dataset_helper.prepare_for_loss(atom_output, batch["atom_metadata"], part_name)
            atom_loss += part.eval_loss(atom_target, atom_output_part, atom_mask)

        for zi, part in enumerate(bond_parts):
            bond_mask = batch["bond_mask"][zi].to(device)
            part_name = part.__class__.__name__
            bond_target = dataset_helper.prepare_for_loss(bond_data, batch["bond_metadata"], part_name)
            bond_output_part = dataset_helper.prepare_for_loss(bond_output, batch["bond_metadata"], part_name)
            bond_loss += part.eval_loss(bond_target, bond_output_part, bond_mask)

        # Total loss
        atom_loss_tot = torch.sum(atom_loss, dim=(0, 1))
        bond_loss_tot = torch.sum(bond_loss, dim=(0, 1))
        print(f" a={atom_loss_tot} b={bond_loss_tot} ")
        total_loss = atom_loss_tot + bond_loss_tot
        return total_loss

    # def train_model(model, dataloader, dataset_helper: MoleculeDatasetHelper, dataloader_val, atom_parts, bond_parts, optimizer, num_epochs=50, device='cuda'):
    def train_model(self, model, dataset, dataset_val,
                    optimizer, num_epochs=200, device=None):
        """
        Train a model using the MoleculeDataset.
        :param model: PyTorch model to train.
        :param dataloader: dataloader for MoleculeDataset.
        :param atom_parts: List of atom embedding parts.
        :param bond_parts: List of bond embedding parts.
        :param optimizer: PyTorch optimizer.
        :param num_epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        :param device: 'cpu' or 'cuda'.
        """
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        if(device is None):
            device = self.train_config.device
        dataset_helper = self.molecule_dataset_helper
        model.to(device)

        dataloader = DataLoader(dataset, batch_size=self.train_config.BATCH_SIZE, shuffle=True)
        dataloader_val = DataLoader(dataset_val, batch_size=self.train_config.BATCH_SIZE, shuffle=True)

        atom_parts = dataset_helper.atom_embedding_parts
        bond_parts = dataset_helper.bond_embedding_parts

        # Create schedulers
        ramp_up_scheduler = LinearRampUpScheduler(optimizer, self.ramp_up_steps, self.train_config.LEARNING_RATE)
        cyclic_scheduler = CyclicLR(optimizer, cycle_momentum=False,
                                    base_lr=0.2 * self.train_config.LEARNING_RATE, max_lr=self.train_config.LEARNING_RATE,
                                    step_size_up=400, step_size_down=400)

        ckpt_counter = 0
        steps_counter = 0
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0

            for batch in dataloader:
                atom_data = batch["atom_data"].to(device)
                bond_data = batch["bond_data"].to(device)

                optimizer.zero_grad()

                # Forward pass
                outputs, latent = model(atom_data, bond_data)  # Example: Predicting atom embeddings
                atom_output = outputs[0]
                bond_output = outputs[1]
                total_loss = self.loss_fn(atom_data, bond_data, atom_output, bond_output, batch)
                epoch_loss += total_loss.item()

                # Backward pass and optimization
                total_loss.backward()
                # Clip gradients
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.train_config.GRAD_CLIP)

                optimizer.step()

                # Update learning rate
                steps_counter = steps_counter+1
                if steps_counter < self.ramp_up_steps:
                    ramp_up_scheduler.step()
                else:
                    cyclic_scheduler.step()
                # Print learning rate for demonstration
                print(f"Epoch {epoch + 1}, LR: {optimizer.param_groups[0]['lr']}")

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

            # Validation phase
            # val_loss, avg_forward_time = validate(model, dataloader_val,dataset_val,atom_parts,bond_parts, device=device)
            val_loss, avg_forward_time = self.validate(model, dataloader_val, device=device)

            # print(f"Epoch {epoch + 1}/{settings.NUM_EPOCHS} - Validation Loss: {val_loss:.4f}")
            print(f"Average Forward Pass Time: {avg_forward_time:.4f} seconds, Val Loss: {val_loss}")

            # Save the model checkpoint if validation loss improves
            if val_loss < best_val_loss:
                print("Validation loss improved, saving model...")
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"{self.ds_settings.CHECKPOINTS_DIR}/model_B_ckpt_{ckpt_counter}.ckpt")
                ckpt_counter = ckpt_counter + 1

    def validate(self, model, val_loader, device=None):
        """
        Validate the model and measure forward pass time.

        :param model: The PyTorch model.
        :param val_loader: DataLoader for validation data.
        :param loss_fn: Loss function.
        :param settings: Settings object with validation configurations.
        :return: Tuple (validation loss, average forward pass time).
        """
        if(device is None):
            device = self.train_config.device
        dataset_helper = self.molecule_dataset_helper

        model.eval()
        model.to(device)
        val_loss = 0.0
        total_time = 0.0

        with torch.no_grad():
            for batch in val_loader:
                atom_data = batch["atom_data"].to(device)
                bond_data = batch["bond_data"].to(device)
                # Forward pass

                # Measure forward pass time
                start_time = torch.cuda.Event(enable_timing=True)
                end_time = torch.cuda.Event(enable_timing=True)

                start_time.record()
                outputs, latent = model(atom_data, bond_data)  # Example: Predicting atom embeddings
                end_time.record()

                atom_output = outputs[0]
                bond_output = outputs[1]

                torch.cuda.synchronize()
                total_time += start_time.elapsed_time(end_time) / 1000.0  # Convert ms to seconds

                total_loss = self.loss_fn(atom_data, bond_data, atom_output, bond_output, batch)
                val_loss += total_loss.item()

        val_loss /= len(val_loader)
        avg_forward_time = total_time / len(val_loader)

        return val_loss, avg_forward_time

    #def run_experiment(self, experiment_config : BaseExperimentConfig):
    #    experiment_config.create_data_config()

if __name__ == "__main__":
    # Access file paths
    train_data_path = DeepSpaceSettings.TRAIN_DATA_FILE
    model_checkpoint = DeepSpaceSettings.CHECKPOINT_FILE

    #smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CC[N+](C)(C)C"]


    #with open('C:\datasets\chembl_size90_input_smiles.csv', 'r') as file:
    #    lines = file.readlines()

    smiles_file = "C:\datasets\chembl_size90_input_smiles.csv"
    input_smiles = filter_smiles(smiles_file,40000,32,return_scrambled=0)

    random.shuffle(input_smiles)
    smiles = input_smiles[0:1600]
    smiles_val = input_smiles[1601:2000]

    constants = DeepSpaceConstants(MAX_ATOMS=32, DIST_SCALING=1.0, device="cuda")
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

    dataset, dataset_helper = create_dataset(smiles, atom_embedding_parts, bond_embedding_parts, constants)
    dataset_inmem = InMemoryDataset(dataset)
    dataloader = DataLoader(dataset_inmem, batch_size=batch_size, shuffle=True)

    dataset_val, dataset_helper_val = create_dataset(smiles_val, atom_embedding_parts, bond_embedding_parts, constants)
    dataset_val_inmem = InMemoryDataset(dataset_val)
    dataloader_val = DataLoader(dataset_val_inmem, batch_size=batch_size, shuffle=True)

    print("Create datasets and cache in memory [DONE]")

    feature_dims_atoms = sum( pi.flattened_tensor_size() for pi in dataset.atom_embedding_parts )
    feature_dims_bonds = sum( pi.flattened_tensor_size() for pi in dataset.bond_embedding_parts )

    model = TransformerAutoencoderWithIngress(feature_dims=(feature_dims_atoms,feature_dims_bonds)).to('cuda')
    count_parameters(model)

    if True:
        model.load_state_dict(torch.load(f"checkpoints/model_ckpt_91.ckpt", weights_only=True))

    #optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # train_model(model, dataloader, dataset_helper, dataloader_val, dataset_val, atom_embedding_parts, bond_embedding_parts, optimizer, num_epochs=2000)
    train_model(model, dataloader, dataset_helper, dataloader_val, optimizer, num_epochs=2000)