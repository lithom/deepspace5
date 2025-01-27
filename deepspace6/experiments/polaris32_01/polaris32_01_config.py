from pathlib import Path

from deepspace6.configs.base_experiment_config import BaseExperimentConfig, BaseModelConfig, BaseDataConfig, \
    BaseTrainConfig
from deepspace6.configs.ds_constants import DeepSpaceConstants
from deepspace6.configs.ds_settings import DeepSpaceSettings
from deepspace6.data.molecule_dataset import MoleculeDatasetHelper
from deepspace6.embeddings.structure_embeddings import AtomTypeEmbeddingPart, VertexDistanceMapEmbeddingPart, \
    BondInfoEmbeddingPart


# Example derived class
class Polaris32ExperimentConfig(BaseExperimentConfig):
    def __init__(self):
        super().__init__()

    def create_model_config(self):
        ds_constants = DeepSpaceConstants(MAX_ATOMS=32, DIST_SCALING=1.0, device="cuda")
        ds_settings = DeepSpaceSettings(Path(__file__).resolve().parent)
        ds_settings.create_directories()
        atom_embedding_parts = [
            AtomTypeEmbeddingPart(ds_constants),
            VertexDistanceMapEmbeddingPart(ds_constants)
            # VertexDistanceEmbeddingPart(constants),
            # ApproximateDistanceEmbeddingPart(constants),
            # RingStatusEmbeddingPart(constants),
            # SymmetryRankEmbeddingPart(constants) #,
            # PharmacophoreFlagsEmbeddingPart(constants),
            # Add other embedding parts as needed
        ]
        bond_embedding_parts = [
            BondInfoEmbeddingPart(ds_constants)
        ]
        molecule_dataset_helper = MoleculeDatasetHelper(atom_embedding_parts, bond_embedding_parts, ds_constants)

        return BaseModelConfig(
            molecule_dataset_helper,
            ds_settings=ds_settings,
            ds_constants=ds_constants
        )

    def create_data_config(self):
        return BaseDataConfig(file_train="C:\datasets\chembl_size90_input_smiles.csv")

    def create_train_config(self):
        return BaseTrainConfig()