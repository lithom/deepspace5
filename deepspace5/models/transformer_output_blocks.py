import torch
from torch import nn
from torch.nn import Transformer
import torch.nn.functional as F
#from leet_deep.canonicalsmiles import LeetTransformer
import math
import numpy as np

from deepspace5.architecture.outputhead import OutputHeadConfiguration
from deepspace5.models.dataset_utils import create_atom_mask
from deepspace5.models.loss_utils import compute_class_counts


class TransformerOutputHeadA(nn.Module):
    def __init__(self, dim, internal_dim_1, output_dim, num_layers=1, nhead=2, dim_feedforward=256):
        super(TransformerOutputHeadA, self).__init__()
        self.dim = dim
        self.internal_dim_1 = internal_dim_1
        self.output_dim = output_dim
        #self.output_dim_total = output_dim
        self.output_dim_total = np.prod(output_dim)

        # First linear layer to transform dimension to internal_dim_1
        self.linear1 = nn.Linear(self.dim, self.internal_dim_1)

        # Transformer block
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.internal_dim_1,
                nhead=nhead,  # Adjust the number of heads as needed
                batch_first=True,
                dim_feedforward=dim_feedforward
            ),
            num_layers=num_layers  # Number of layers in the transformer encoder
        )

        # Second linear layer to scale the output to output_dim
        self.linear2 = nn.Linear(self.internal_dim_1, self.output_dim_total)

    def forward(self, x):
        # x expected to be of shape (batch_size, 32, dim)
        x = self.linear1(x)  # Transform to (batch_size, 32, internal_dim_1)
        x = self.transformer(x)  # Process with transformer
        x = self.linear2(x)  # Scale to (batch_size, 32, output_dim)
        if(isinstance(self.output_dim, list)):
            x = torch.reshape(x , (*x.shape[:-1], *self.output_dim))
        return x


#@register_output_head("ExampleOutputHead")
class ConfigTransformerOutputHeadA(OutputHeadConfiguration):
    def __init__(self):
        super().__init__()
        self.head_name = None
        self.global_data = None
        self.config = None
        self.module = None
        self.iteration_data = []

    def load_config(self, base_model_dim, config):
        # Configuration could include specific parameters for the head, like dimensions
        self.base_model_dim = base_model_dim
        self.dim_internal = config.get('dim_internal',32)
        self.output_dim = config.get('output_dim', 10)  # Default to 10 if not specified
        self.num_layers = config.get('num_layers',2)
        self.nhead = config.get('nhead',8)
        self.dim_feedforward = config.get('dim_feedforward',512)
        self.datafile   = config.get('datafile')
        self.datafile_dataset = config.get('datafile_dataset')
        self.config = config
        self.module = TransformerOutputHeadA(self.base_model_dim,self.dim_internal,self.output_dim,self.num_layers,nhead=self.nhead,dim_feedforward=self.dim_feedforward)
        self.device = None

    def create_data_sample(self, idx: int):
        # Assume 'labels' is a key in the loaded data dictionary
        labels = self.global_data[self.datafile][self.datafile_dataset]
        return {
            "in": None,  # No specific input data required by this head
            "out": torch.tensor(labels, dtype=torch.int64)
        }

    def set_global_data(self, head_name, device, global_data):
        self.head_name = head_name
        self.device = device
        self.global_data = global_data

    def create_module(self):
        return self.module

    def get_data_length(self):
        return len(self.global_data[self.datafile]['Samples'])

    def get_data_sample(self, idx):
        data_i = torch.tensor(self.global_data[self.datafile]['Samples'][idx][self.datafile_dataset],dtype=torch.float32)
        #num_atoms = self.global_data[self.datafile]['Samples'][idx]['mol_properties_a']['numAtoms'] #self.global_data[self.datafile]['Samples'][idx]['NumAtoms']
        num_atoms = len(self.global_data[self.datafile]['Samples'][idx]['atom_properties'])  # self.global_data[self.datafile]['Samples'][idx]['NumAtoms']
        mask_i = create_atom_mask(data_i.shape, [num_atoms]*len(self.config['masked_dimensions']) , self.config['masked_dimensions'] )
        return { 'data': data_i, 'mask': mask_i }


    def compute_loss(self, output, target):
        mask = target['mask'].to(self.device)
        return torch.nn.functional.mse_loss(output*mask,target['data'].to(self.device)*mask)



class ConfigTransformerOutputHeadA_ForAdjacency(OutputHeadConfiguration):
    def __init__(self):
        super().__init__()
        self.head_name = None
        self.global_data = None
        self.config = None
        self.module = None

    def load_config(self, base_model_dim, config):
        # Configuration could include specific parameters for the head, like dimensions
        self.base_model_dim = base_model_dim
        self.dim_internal = config.get('dim_internal',32)
        self.output_dim = [32,4,2]
        self.num_layers = config.get('num_layers',2)
        self.nhead = config.get('nhead',8)
        self.dim_feedforward = config.get('dim_feedforward',512)
        self.datafile   = config.get('datafile')
        #self.datafile_dataset = config.get('datafile_dataset')
        self.config = config
        self.module = TransformerOutputHeadA(self.base_model_dim,self.dim_internal,self.output_dim,self.num_layers,nhead=self.nhead,dim_feedforward=self.dim_feedforward)
        self.device = None

    def create_data_sample(self, idx: int):
        # Assume 'labels' is a key in the loaded data dictionary
        labels = self.global_data[self.datafile][self.datafile_dataset]
        return {
            "in": None,  # No specific input data required by this head
            "out": torch.tensor(labels, dtype=torch.int64)
        }

    def set_global_data(self, head_name, device, global_data):
        self.head_name = head_name
        self.device = device
        self.global_data = global_data

    def create_module(self):
        return self.module

    def get_data_length(self):
        return len(self.global_data[self.datafile]['Samples'])

    def get_data_sample(self, idx):
        data_i = torch.tensor(self.global_data[self.datafile]['Samples'][idx]["adj_tensor"],dtype=torch.int64)
        #data_i = torch.reshape( data_i , (data_i.size(0), data_i.size(1), -1))
        #num_atoms = self.global_data[self.datafile]['Samples'][idx]['mol_properties_a']['numAtoms'] #self.global_data[self.datafile]['Samples'][idx]['NumAtoms']
        num_atoms = len(self.global_data[self.datafile]['Samples'][idx]['atom_properties'])  # self.global_data[self.datafile]['Samples'][idx]['NumAtoms']
        mask_i = create_atom_mask(data_i.shape[0:2], [num_atoms]*2 , [0,1] )
        mask_i = mask_i.unsqueeze(-1).repeat(1,1,data_i.shape[2])
        mask_i = mask_i.unsqueeze(-1).repeat(1,1,1,2)
        # flatten
        #data_i = torch.reshape( data_i , (data_i.size(0), -1) )
        #mask_i = torch.reshape( mask_i , (mask_i.size(0), -1) )
        return { 'data': data_i, 'mask': mask_i }


    def start_iteration(self):
        self.iteration_data = []


    def end_iteration(self, output_level: int):
        if(output_level > 0):
            if( len(self.iteration_data)>0 ):
                #combined_data = {''}
                for xi in self.iteration_data:
                    print(xi)



    def compute_loss(self, output, target):
        mask = target['mask'].to(self.device)
        output_masked = output * mask
        output_masked_permuted = output_masked.permute(0,4,1,2,3)
        target_data = target['data'].to(self.device)
        # set all masked values to new class..
        mask_scaled = mask[:,:,:,:,0]
        num_classes = output_masked_permuted.size()[1]
        target_data[mask_scaled==0] = num_classes
        output_masked_permuted_extended = torch.cat((output_masked_permuted, torch.zeros( output_masked_permuted.size()[0] , 1, 32, 32, 4).to(self.device) ), dim=1)

        #result = torch.nn.functional.cross_entropy(output_masked_permuted,target_data)
        # weights: there are about ten times more class 0 than class 1..
        weights = torch.tensor([0.1, 1.0, 0.0]).to(output_masked_permuted_extended.device)
        result = torch.nn.functional.cross_entropy(output_masked_permuted_extended[:,:,:,:,0:1], target_data[:,:,:,0:1], ignore_index=2 , weight = weights)

        # for logging:
        #class_counts = compute_class_counts(self.device, output_masked_permuted_extended[:,:,:,:,0], target_data[:,:,:,0])
        class_counts = compute_class_counts(self.device, output_masked_permuted_extended[:, :, :, :, 0:1], target_data[:, :, :, 0:1])
        self.iteration_data.append(class_counts)

        return result