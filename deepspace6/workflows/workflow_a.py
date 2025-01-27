import torch
from torch.utils.data import DataLoader
import random

from deepspace6.configs.ds_constants import DeepSpaceConstants
from deepspace6.embeddings.basic_embeddings import SymmetryRankEmbeddingPart, PharmacophoreFlagsEmbeddingPart, VertexDistanceEmbeddingPart, ApproximateDistanceEmbeddingPart,RingStatusEmbeddingPart
from deepspace6.embeddings.structure_embeddings import AtomTypeEmbeddingPart, BondInfoEmbeddingPart, VertexDistanceMapEmbeddingPart
from deepspace6.data.molecule_dataset import MoleculeDataset, create_dataset, InMemoryDataset
from deepspace6.models.basic_autoencoder import TransformerAutoencoderWithIngress
from deepspace6.utils.pytorch_utils import count_parameters
from deepspace6.utils.smiles_utils import filter_smiles
from deepspace6.workflows.training_a import train_model

from deepspace6.configs.ds_settings import DeepSpaceSettings




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





