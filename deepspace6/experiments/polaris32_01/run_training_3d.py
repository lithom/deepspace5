import pickle

import torch
from torch import optim

from deepspace6.data.molecule_dataset import create_dataset_with_conformers, InMemoryDataset
from deepspace6.experiments.polaris32_01.polaris32_01_config import Polaris32ExperimentConfig
from deepspace6.models.basic_autoencoder import TransformerAutoencoderWithIngress
from deepspace6.models.basic_histogram_autoencoder import GeometryPredictor
from deepspace6.trainers.basic_geometry_encoder_trainer import GeometryEncoderWithStructureLatentTrainer
from deepspace6.utils.ds6_utils import load_pickle, count_parameters
from deepspace6.utils.smiles_utils import filter_smiles

if __name__ == "__main__":

    experiment = Polaris32ExperimentConfig()
    model_config = experiment.create_model_config()
    dataset_helper = model_config.molecule_dataset_helper

    model = GeometryPredictor(n_layers=3)
    count_parameters(model)

    if True:
        model.load_state_dict(torch.load("C:\dev\deepspace5\deepspace6\experiments\polaris32_01\models\checkpoints\model_meanvar_polarisA_B_ckpt_140.ckpt",weights_only=True))

    if False:
        #smiles_file = "C:\datasets\chembl_size90_input_smiles.csv"
        #input_smiles = filter_smiles(smiles_file, 4000, 8, 32, return_scrambled=0)
        #smiles_rand = input_smiles
        smiles_file_train = "C:/dev/deepspace5/deepspace6/utils/smilesSetB.txt"
        smiles_file_val = "C:/dev/deepspace5/deepspace6/utils/smilesSetA.txt"

        input_smiles_train = filter_smiles(smiles_file_train, 512000, 8, 32, return_original=False, return_scrambled=2)
        input_smiles_val = filter_smiles(smiles_file_val, 16000, 8, 32, return_original=False, return_scrambled=1)

        dataset_train_with_confis, dataset_helper = create_dataset_with_conformers(input_smiles_train, dataset_helper.atom_embedding_parts,
                                                                             dataset_helper.bond_embedding_parts, model_config.ds_constants,
                                                                             num_scrambled=1)
        dataset_val_with_confis, dataset_helper = create_dataset_with_conformers(input_smiles_val,
                                                                                   dataset_helper.atom_embedding_parts,
                                                                                   dataset_helper.bond_embedding_parts,
                                                                                   model_config.ds_constants,
                                                                                   num_scrambled=1)


        dataset_train_with_confis_inmem = InMemoryDataset(dataset_train_with_confis)
        dataset_val_with_confis_inmem = InMemoryDataset(dataset_val_with_confis)

        with open('data/dataset_polarisB_train_03.pkl', 'wb') as f:
            pickle.dump(dataset_train_with_confis_inmem, f)
        with open('data/dataset_polarisB_val_03.pkl', 'wb') as f:
            pickle.dump(dataset_val_with_confis_inmem, f)

    if True:
        #dataset_inmem = load_pickle("data/dataset_polarisA_all_inmem_4k_01.pkl")
        dataset_train_inmem = load_pickle('data/dataset_polarisB_train_03.pkl') #load_pickle("C:\dev\deepspace5\deepspace6/workflows/datasets/dataset_c_all_inmem_64k_01.pkl")
        dataset_val_inmem = load_pickle('data/dataset_polarisB_val_03.pkl')

    # create autoencoder model
    feature_dims_atoms = sum(pi.flattened_tensor_size() for pi in dataset_helper.atom_embedding_parts)
    feature_dims_bonds = sum(pi.flattened_tensor_size() for pi in dataset_helper.bond_embedding_parts)
    model_autoencoder = TransformerAutoencoderWithIngress(feature_dims=(feature_dims_atoms, feature_dims_bonds)).to('cuda')
    model_autoencoder.load_state_dict(
        #torch.load("C:\dev\deepspace5\deepspace6\experiments\polaris32_01\models\checkpoints\model_ckpt_1.ckpt",
        #           weights_only=True))
        torch.load("C:\dev\deepspace5\deepspace6\experiments\polaris32_01\models\checkpoints\model_B_ckpt_7.ckpt",
                    weights_only=True))

    # optimizer = optim.Adam(model.parameters(), lr=0.00025)
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.0015)

    trainer = GeometryEncoderWithStructureLatentTrainer(model, model_autoencoder, model_config.ds_constants)
    trainer.train(dataset_train_inmem, dataset_val_inmem, optimizer, 0.00075)
