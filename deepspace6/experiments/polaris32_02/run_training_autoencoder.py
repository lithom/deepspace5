import random

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from deepspace6.data.molecule_dataset import create_dataset, InMemoryDataset, HDF5Dataset
from deepspace6.experiments.polaris32_02.polaris32_02_config import Polaris32_02_ExperimentConfig
from deepspace6.models.basic_autoencoder import TransformerAutoencoderWithIngress
from deepspace6.trainers.basic_molecule_encoder_lightning_trainer import MoleculeLightningModule
from deepspace6.trainers.basic_molecule_encoder_trainer import MoleculeEncoderTrainer
from deepspace6.utils.ds6_utils import count_parameters
from deepspace6.utils.smiles_utils import filter_smiles, filter_smiles_for_train_and_val, evaluate_set_distances

import torch.multiprocessing as mp



def main():
    experiment = Polaris32_02_ExperimentConfig()

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
        #input_smiles_train, input_smiles_val = filter_smiles_for_train_and_val(smiles_file_large, 128000,ratio_val=0.15, return_original=False, return_scrambled=3)
        input_smiles_train, input_smiles_val = filter_smiles_for_train_and_val(smiles_file_large, 32000,
                                                                               ratio_val=0.15, return_original=False,
                                                                               return_scrambled=3)
        #distance_stats = evaluate_set_distances(input_smiles_train,input_smiles_val)


    dataset_train, dataset_helper = create_dataset(input_smiles_train,
                                             model_config.molecule_dataset_helper.atom_embedding_parts,
                                             model_config.molecule_dataset_helper.bond_embedding_parts,
                                             model_config.ds_constants,
                                             scramble_molecules=False)
    dataset_val, dataset_helper_val = create_dataset(input_smiles_val,
                                             model_config.molecule_dataset_helper.atom_embedding_parts,
                                             model_config.molecule_dataset_helper.bond_embedding_parts,
                                             model_config.ds_constants,
                                             scramble_molecules=False)

    dataset_inmem_train = InMemoryDataset(dataset_train)
    dataset_inmem_val = InMemoryDataset(dataset_val)

    #dataset_hdf5_train = HDF5Dataset(dataset_train,hdf5_path="dataset_train.h5")
    #dataset_hdf5_val   = HDF5Dataset(dataset_val,hdf5_path="dataset_val.h5")


    # create model:
    feature_dims_atoms = sum( pi.flattened_tensor_size() for pi in dataset_train.atom_embedding_parts )
    feature_dims_bonds = sum( pi.flattened_tensor_size() for pi in dataset_train.bond_embedding_parts )
    model = TransformerAutoencoderWithIngress( combined_dim=320, n_heads=8, latent_dim=16,n_layers=2,feature_dims=(feature_dims_atoms,feature_dims_bonds)).to('cuda')
    #model.load_state_dict(torch.load("C:\dev\deepspace5\deepspace6\experiments\polaris32_01\models\checkpoints\model_ckpt_1.ckpt",weights_only=True))
    count_parameters(model)



    # create optimizer:
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.LEARNING_RATE)

    # train:
    if(False) :
        trainer = MoleculeEncoderTrainer(dataset_helper, train_config, ds_settings=model_config.ds_settings)
        trainer.train_model(model,dataset_inmem_train, dataset_inmem_val,optimizer,train_config.NUM_EPOCHS)

    if(True):
        # ✅ Define checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            dirpath="checkpoints/",  # Folder where checkpoints are saved
            filename="molecule_model-{epoch:02d}-{val_loss:.4f}",  # Filename format
            monitor="val_loss",  # Metric to monitor
            save_top_k=3,  # Save top 3 best models
            mode="min",  # "min" because lower val_loss is better
            save_weights_only=False,  # Save optimizer state as well
        )

        logger = TensorBoardLogger("tb_logs", name="molecule_model")
        trainer = Trainer(logger=logger, max_epochs=8000, gradient_clip_val=1.0, accelerator="gpu", devices=1,
                          callbacks=[checkpoint_callback] )

        model_module = MoleculeLightningModule(model, dataset_helper, train_config, model_config.ds_settings)

        dataloader_train = DataLoader(dataset_inmem_train, batch_size=train_config.BATCH_SIZE, shuffle=True)
        dataloader_val = DataLoader(dataset_inmem_val, batch_size=train_config.BATCH_SIZE, shuffle=False)
        #dataloader_train = DataLoader(dataset_hdf5_train, batch_size=train_config.BATCH_SIZE, shuffle=True, num_workers=4)
        #dataloader_val = DataLoader(dataset_hdf5_val, batch_size=train_config.BATCH_SIZE, shuffle=False, num_workers=4)
        #dataloader_train = DataLoader(dataset_train, batch_size=train_config.BATCH_SIZE, shuffle=True, num_workers=4)
        #dataloader_val = DataLoader(dataset_val, batch_size=train_config.BATCH_SIZE, shuffle=False, num_workers=4)

        trainer.fit(model_module, dataloader_train, dataloader_val, ckpt_path="C:\dev\deepspace5\deepspace6\experiments\polaris32_02\checkpoints\molecule_model-epoch=219-val_loss=3539.8447.ckpt")
        #trainer.fit(model_module, dataloader_train, dataloader_val)
    print("mkay")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Ensures correct multiprocessing behavior
    main()







