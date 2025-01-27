from deepspace6.configs.ds_constants import DeepSpaceConstants
from deepspace6.data.molecule_dataset import MoleculeDatasetHelper
from deepspace6.embeddings.structure_embeddings import AtomTypeEmbeddingPart, VertexDistanceMapEmbeddingPart, \
    BondInfoEmbeddingPart
from deepspace6.models.basic_autoencoder import TransformerAutoencoderWithIngress
from deepspace6.models.basic_geometry_model import GeometryModel
from deepspace6.models.basic_histogram_autoencoder import FullHistogramAutoencoder
from deepspace6.pipelines.molecule_encoder_pipeline import MoleculeEncoderPipeline
from deepspace6.pipelines.molecule_geometry_pipeline import MoleculeGeometryPipeline
import os
from pathlib import Path

DS6_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root

def create_pipeline_polaris_01(device="cuda", checkpoint='' ):
    constants = DeepSpaceConstants(MAX_ATOMS=32, DIST_SCALING=1.0, device=device)
    atom_embedding_parts = [
        AtomTypeEmbeddingPart(constants),
        VertexDistanceMapEmbeddingPart(constants)
    ]
    bond_embedding_parts = [
        BondInfoEmbeddingPart(constants)
    ]
    dataset_helper = MoleculeDatasetHelper(atom_embedding_parts, bond_embedding_parts, constants)
    feature_dims_atoms = sum( pi.flattened_tensor_size() for pi in dataset_helper.atom_embedding_parts )
    feature_dims_bonds = sum( pi.flattened_tensor_size() for pi in dataset_helper.bond_embedding_parts )
    transformer_ae = TransformerAutoencoderWithIngress(feature_dims=(feature_dims_atoms,feature_dims_bonds)).to('cuda')


    return {
        'constants': constants,
        'model_polaris_ae': transformer_ae,
        'dataset_helper': dataset_helper,
        'pipeline': MoleculeEncoderPipeline(transformer_ae,dataset_helper,device)
    }

def create_pipeline_hyperflex_01(device="cuda"):

    polaris = create_pipeline_polaris_01()
    geometry_model = GeometryModel()
    hist_ae = FullHistogramAutoencoder(num_bins=64, n2d=8, dim_latent=8192,n_layers=1)
    hyperflex = MoleculeGeometryPipeline(polaris,geometry_model,hist_ae,device)

    return {
        'model_polaris_ae': polaris['model_polaris_ae'],
        'model_geometry': geometry_model,
        'model_hyperflex_ae': hist_ae,
        'dataset_helper': polaris['dataset_helper'],
        'pipeline_polaris': polaris,
        'pipeline_hyperflex': hyperflex
    }


