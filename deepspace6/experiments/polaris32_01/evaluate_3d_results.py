import torch

from deepspace6.data.molecule_dataset import create_dataset_with_conformers, InMemoryDataset
from deepspace6.experiments.polaris32_01.polaris32_01_config import Polaris32ExperimentConfig
from deepspace6.models.basic_autoencoder import TransformerAutoencoderWithIngress
from deepspace6.models.basic_histogram_autoencoder import GeometryPredictor
from deepspace6.utils.pytorch_utils import count_parameters
from deepspace6.utils.smiles_utils import filter_smiles_for_train_and_val

if __name__ == "__main__":
    checkpoint_encoder = "C:\dev\deepspace5\deepspace6\experiments\polaris32_01\models\checkpoints\model_B_ckpt_7.ckpt"
    checkpoint_meanvar = "C:\dev\deepspace5\deepspace6\experiments\polaris32_01\models\checkpoints\model_meanvar_polarisA_B_ckpt_440.ckpt"

    experiment = Polaris32ExperimentConfig()

    model_config = experiment.create_model_config()
    data_config = experiment.create_data_config()
    train_config = experiment.create_train_config()


    smiles_file_train = "C:/dev/deepspace5/deepspace6/utils/smilesSetB.txt"
    smiles_file_val = "C:/dev/deepspace5/deepspace6/utils/smilesSetA.txt"
    smiles_file_large = "C:\datasets\chembl_size90_input_smiles.csv"


    input_smiles_train, input_smiles_val = filter_smiles_for_train_and_val(smiles_file_large, 400,ratio_val=0, return_original=False, return_scrambled=1)

    dataset, dataset_helper = create_dataset_with_conformers(input_smiles_train,
                                             model_config.molecule_dataset_helper.atom_embedding_parts,
                                             model_config.molecule_dataset_helper.bond_embedding_parts,
                                             model_config.ds_constants,
                                             scramble_molecules=False)

    dataset_inmem_train = InMemoryDataset(dataset)



    # create model:
    feature_dims_atoms = sum( pi.flattened_tensor_size() for pi in dataset.atom_embedding_parts )
    feature_dims_bonds = sum( pi.flattened_tensor_size() for pi in dataset.bond_embedding_parts )
    model_ae = TransformerAutoencoderWithIngress(feature_dims=(feature_dims_atoms,feature_dims_bonds)).to('cuda')
    model_ae.load_state_dict(torch.load("C:\dev\deepspace5\deepspace6\experiments\polaris32_01\models\checkpoints\model_ckpt_1.ckpt", weights_only=True))
    count_parameters(model_ae)

    # create 3d model:
    model_3d = GeometryPredictor(n_layers=3)
    count_parameters(model_3d)
    if True:
        model_3d.load_state_dict(torch.load("C:\dev\deepspace5\deepspace6\experiments\polaris32_01\models\checkpoints\model_meanvar_polarisA_B_ckpt_140.ckpt",weights_only=True))











