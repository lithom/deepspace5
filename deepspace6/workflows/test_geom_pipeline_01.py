import time
import torch
#import cProfile
#import pstats

import rdkit.Chem as Chem

import deepspace6.ds6 as ds6
from deepspace5.datagenerator.dataset_generator import generate_conformers_and_compute_statistics

from deepspace6.pipelines.molecule_encoder_pipeline import MoleculeEncoderPipeline
from deepspace6.pipelines.molecule_geometry_pipeline import MoleculeGeometryPipeline
from deepspace6.utils.smiles_utils import filter_smiles
from deepspace6.utils.visualization_utils import compare_histograms

if __name__ == "__main__":

    num_structures = 100

    smiles_file = "C:\datasets\chembl_size90_input_smiles.csv"
    input_smiles = filter_smiles(smiles_file, num_structures, 8, 32 )
    hyperflex_01 = ds6.create_pipeline_hyperflex_01()
    geom_pipeline = hyperflex_01['pipeline_hyperflex']

    geom_pipeline.load_checkpoint_a(ds6.DS6_ROOT_DIR+'/workflows/checkpoints/model_ckpt_91.ckpt',
                                    ds6.DS6_ROOT_DIR+'/workflows/checkpoints/model_geom_A_ckpt_199.ckpt',
                                    ds6.DS6_ROOT_DIR+'/workflows/checkpoints/model_histo3d_latent8192_B_p2_ckpt_99.ckpt')

    print("Go!")
    t0 = time.time()
    hist_data = geom_pipeline.run(input_smiles, batch_size=4096)
    #cProfile.run('geom_pipeline.run(input_smiles, batch_size=512)','profile_output')
    t1 = time.time()
    print("Done!")

    print(f'time: {(t1-t0)} , mol/s: { num_structures * (1.0/(t1-t0)) } , time per mol: {(t1-t0)/num_structures} s')

    test_idx = 14
    coords_list, distance_stats = generate_conformers_and_compute_statistics(Chem.MolFromSmiles(input_smiles[test_idx]),32,num_bins=64,max_dist=32)

    print("mkay")
    tensor_test = torch.tensor(distance_stats['histogram']).float()
    for zi in range(32):
        for zj in range(32):
            if( sum(tensor_test[zi,zj,:]) >= 0.00001 ):
                tensor_test[zi,zj,:] = tensor_test[zi,zj,:]  / sum(tensor_test[zi,zj,:])
            else:
                tensor_test[zi,zj,0] = 1.0

    tensor_out, latent = geom_pipeline.histogram_ae(tensor_test.unsqueeze(0).to("cuda"))
    tensor_out_2 = geom_pipeline.histogram_ae.decode(latent)
    latent_n1 = latent + 0.1*torch.randn(1,8192,1).to("cuda")
    latent_n2 = latent + 0.25 * torch.randn(1, 8192, 1).to("cuda")
    tensor_out_3 = geom_pipeline.histogram_ae.decode(latent_n1 )
    tensor_out_4 = geom_pipeline.histogram_ae.decode(latent_n2)

    print("mkay")

    #compare_histograms(torch.tensor(distance_stats['histogram']), torch.exp(tensor_out[0,:,:,:].detach()), num_pairs=4)

    compare_histograms(torch.exp(tensor_out_3[0, :, :, :].detach()), torch.exp(tensor_out[0, :, :, :].detach()),num_pairs=4)

    compare_histograms(torch.tensor(distance_stats['histogram']), torch.tensor(distance_stats['histogram']),num_pairs=8)
    compare_histograms(torch.tensor(distance_stats['histogram']),torch.exp(hist_data[0][test_idx,:,:,:]))
    # Print a sorted summary
    #p = pstats.Stats('profile_output')
    #p.strip_dirs().sort_stats('time').print_stats()

