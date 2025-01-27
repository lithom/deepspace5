import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader

from deepspace6.configs.ds_constants import DeepSpaceConstants
from deepspace6.data.molecule_dataset import create_dataset_with_conformers, InMemoryDataset
from deepspace6.embeddings.structure_embeddings import AtomTypeEmbeddingPart, VertexDistanceMapEmbeddingPart, \
    BondInfoEmbeddingPart
from deepspace6.utils.smiles_utils import filter_smiles


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

if __name__ == "__main__":

    constants = DeepSpaceConstants(MAX_ATOMS=32, DIST_SCALING=1.0, device="cuda")
    batch_size = 512
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

    if True:
        smiles_rand = []
        #file_path = 'C:\dev\deepspace5\data\datasets\dataset_s32_a1.pickle'  # Replace with your pickle file path
        #data = load_pickle(file_path)
        # for zi in range(64000):
        #    smiles_rand.append( data["Samples"][zi]["smiles_rand"] )

        smiles_file = "C:\datasets\chembl_size90_input_smiles.csv"
        input_smiles = filter_smiles(smiles_file, 64000, 8,32, return_scrambled=0)
        smiles_rand = input_smiles
        dataset_with_confis, dataset_helper = create_dataset_with_conformers(smiles_rand, atom_embedding_parts, bond_embedding_parts, constants, num_scrambled=1)
        dataset_with_confis_inmem = InMemoryDataset(dataset_with_confis)
        with open('datasets/dataset_c_all_64k_01.pkl', 'wb') as f:
            pickle.dump(dataset_with_confis, f)
        with open('datasets/dataset_c_all_inmem_64k_01.pkl', 'wb') as f:
            pickle.dump(dataset_with_confis_inmem, f)
        print("mkay")
