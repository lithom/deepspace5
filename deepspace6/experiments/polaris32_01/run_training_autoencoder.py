import random

import torch

from deepspace6.data.molecule_dataset import create_dataset, InMemoryDataset
from deepspace6.experiments.polaris32_01.polaris32_01_config import Polaris32ExperimentConfig
from deepspace6.models.basic_autoencoder import TransformerAutoencoderWithIngress
from deepspace6.trainers.basic_molecule_encoder_trainer import MoleculeEncoderTrainer
from deepspace6.utils.ds6_utils import count_parameters
from deepspace6.utils.smiles_utils import filter_smiles, filter_smiles_for_train_and_val, evaluate_set_distances

if __name__ == "__main__":
    experiment = Polaris32ExperimentConfig()

    model_config = experiment.create_model_config()
    data_config = experiment.create_data_config()
    train_config = experiment.create_train_config()

    # prepare data..
    dataset_helper = []
    dataset = []
    dataset_val = []

    # if data_config.INPUT_TYPE == "smiles":
    #     smiles_file = data_config.FILE_TRAIN
    #     num_train = 256000#64000
    #     num_val = 16000
    #     # NOTE: for molecule autoencoder, we scramble
    #     input_smiles = filter_smiles(smiles_file, num_train+num_val, 8,32, return_original=False,return_scrambled=4)
    #     random.shuffle(input_smiles)
    #
    #     dataset, dataset_helper = create_dataset(input_smiles[0:num_train],
    #                                              model_config.molecule_dataset_helper.atom_embedding_parts,
    #                                              model_config.molecule_dataset_helper.bond_embedding_parts,
    #                                              model_config.ds_constants,
    #                                              scramble_molecules=False)
    #     dataset_val, dataset_helper_val = create_dataset(input_smiles[num_train:num_val],
    #                                              model_config.molecule_dataset_helper.atom_embedding_parts,
    #                                              model_config.molecule_dataset_helper.bond_embedding_parts,
    #                                              model_config.ds_constants,
    #                                              scramble_molecules=False)
    #     dataset_inmem = InMemoryDataset(dataset)



    smiles_file_train = "C:/dev/deepspace5/deepspace6/utils/smilesSetB.txt"
    smiles_file_val = "C:/dev/deepspace5/deepspace6/utils/smilesSetA.txt"
    smiles_file_large = "C:\datasets\chembl_size90_input_smiles.csv"

    if(False):
        input_smiles_train = filter_smiles(smiles_file_train,32000, 8, 32, return_original=False, return_scrambled=4)
        input_smiles_val   = filter_smiles(smiles_file_val, 8000, 8, 32, return_original=False,return_scrambled=2)
    if(True):
        #input_smiles_all = filter_smiles(smiles_file_train,1000000, 8, 32, return_original=False, return_scrambled=4)
        input_smiles_train, input_smiles_val = filter_smiles_for_train_and_val(smiles_file_large, 128000,ratio_val=0.15, return_original=False, return_scrambled=3)
        #distance_stats = evaluate_set_distances(input_smiles_train,input_smiles_val)


    dataset, dataset_helper = create_dataset(input_smiles_train,
                                             model_config.molecule_dataset_helper.atom_embedding_parts,
                                             model_config.molecule_dataset_helper.bond_embedding_parts,
                                             model_config.ds_constants,
                                             scramble_molecules=False)
    dataset_val, dataset_helper_val = create_dataset(input_smiles_val,
                                             model_config.molecule_dataset_helper.atom_embedding_parts,
                                             model_config.molecule_dataset_helper.bond_embedding_parts,
                                             model_config.ds_constants,
                                             scramble_molecules=False)

    dataset_inmem_train = InMemoryDataset(dataset)
    dataset_inmem_val = InMemoryDataset(dataset_val)




    # create model:
    feature_dims_atoms = sum( pi.flattened_tensor_size() for pi in dataset.atom_embedding_parts )
    feature_dims_bonds = sum( pi.flattened_tensor_size() for pi in dataset.bond_embedding_parts )
    model = TransformerAutoencoderWithIngress(feature_dims=(feature_dims_atoms,feature_dims_bonds)).to('cuda')
    model.load_state_dict(torch.load("C:\dev\deepspace5\deepspace6\experiments\polaris32_01\models\checkpoints\model_ckpt_1.ckpt",weights_only=True))
    count_parameters(model)



    # create optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.LEARNING_RATE)

    # train:
    trainer = MoleculeEncoderTrainer(dataset_helper, train_config, ds_settings=model_config.ds_settings)
    trainer.train_model(model,dataset_inmem_train, dataset_inmem_val,optimizer,train_config.NUM_EPOCHS)

    print("mkay")








