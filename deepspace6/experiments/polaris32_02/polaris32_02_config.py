from pathlib import Path

from deepspace6.configs.base_experiment_config import BaseExperimentConfig, BaseModelConfig, BaseDataConfig, \
    BaseTrainConfig
from deepspace6.configs.ds_constants import DeepSpaceConstants
from deepspace6.configs.ds_settings import DeepSpaceSettings
from deepspace6.data.molecule_dataset import MoleculeDatasetHelper
from deepspace6.embeddings.basic_embeddings_2 import ApproximateDistanceEmbeddingPart2, RingStatusEmbeddingPart2, \
    SymmetryRankEmbeddingPart2, PharmacophoreFlagsEmbeddingPart2, HybridizationEmbeddingPart, \
    LocalGraphFeaturesEmbeddingPart
from deepspace6.embeddings.structure_embeddings import AtomTypeEmbeddingPart, VertexDistanceMapEmbeddingPart, \
    BondInfoEmbeddingPart


# Example derived class
class Polaris32_02_ExperimentConfig(BaseExperimentConfig):
    def __init__(self):
        super().__init__()

    def create_model_config(self):
        ds_constants = DeepSpaceConstants(MAX_ATOMS=32, DIST_SCALING=1.0, MAX_GRAPH_DIST_EXACT=4,device="cuda")
        ds_settings = DeepSpaceSettings(Path(__file__).resolve().parent)
        ds_settings.create_directories()
        atom_embedding_parts = [
            AtomTypeEmbeddingPart(ds_constants),
            VertexDistanceMapEmbeddingPart(ds_constants),
            ApproximateDistanceEmbeddingPart2(ds_constants),
            RingStatusEmbeddingPart2(ds_constants),
            SymmetryRankEmbeddingPart2(ds_constants),
            #PharmacophoreFlagsEmbeddingPart2(ds_constants),
            HybridizationEmbeddingPart(ds_constants),
            LocalGraphFeaturesEmbeddingPart(ds_constants)
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
        return BaseTrainConfig(learning_rate=0.00025)