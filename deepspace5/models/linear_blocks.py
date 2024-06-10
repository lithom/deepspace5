import torch
from torch import nn, tensor

from deepspace5.architecture.outputhead import OutputHeadConfiguration
from deepspace5.models.dataset_utils import create_atom_mask, convert_element_to_int
import torch.nn.functional as F



class LinearOutputHeadA(nn.Module):
    def __init__(self, seq_length, dim, internal_dim_1, internal_dim_2, output_dim, num_layers=1):
        super(LinearOutputHeadA, self).__init__()
        self.dim = dim
        self.internal_dim_1 = internal_dim_1
        self.internal_dim_2 = internal_dim_2
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.num_layers = num_layers

        # First linear layer to transform dimension to internal_dim_1
        if(self.internal_dim_1>0):
            self.linA1 = nn.Linear(self.dim,self.internal_dim_1)
            self.linA2 = nn.Linear(self.internal_dim_1*seq_length, self.internal_dim_1*seq_length)
        # Second linear layer to transform to internal_dim_2
        self.linB = nn.Linear(self.dim + self.internal_dim_1,internal_dim_2)
        # Then create the processing layers:
        if(self.num_layers>0):
            layers = []
            # Add hidden layers with ReLU activation
            for zi in range(num_layers):
                layers.append(nn.Linear(internal_dim_2, internal_dim_2))
                layers.append(nn.ReLU())
            # Sequentially stack the layers
            self.linLayers = nn.Sequential(*layers)
        else:
            self.linLayers = None
        # Third linear layer to scale the output to output_dim
        self.linOutput = nn.Linear(self.internal_dim_2, self.output_dim)

    def forward(self, x):
        # x expected to be of shape (batch_size, 32, dim)
        if(self.internal_dim_1 > 0):
            xa = self.linA1(x)
            xa = xa.view(xa.size(0), -1)
            xa = self.linA2(xa)
            xa = xa.view(xa.size(0),self.seq_length,-1)
            x = torch.cat( (x,xa) , 2 )
        else:
            x = x
        x = self.linB(x)  # Transform to (batch_size, 32, internal_dim_1)
        if(self.num_layers>0):
            x = self.linLayers(x)  # Process with transformer
        x = self.linOutput(x)  # Scale to (batch_size, 32, output_dim)
        return x


#@register_output_head("ExampleOutputHead")
class ExampleLinearOutputHead(OutputHeadConfiguration):
    def __init__(self):
        super().__init__()
        self.head_name = None
        self.global_data = None
        self.config = None
        self.module = None

    def load_config(self, base_model_dim, config):
        # Configuration could include specific parameters for the head, like dimensions
        self.output_dim = config.get('output_dim', 10)  # Default to 10 if not specified
        self.datafile   = config.get('datafile')
        self.datafile_dataset = config.get('datafile_dataset')
        self.config = config
        self.module = LinearOutputHeadA(config['seq_length'])


    def create_data_sample(self, idx: int):
        # Assume 'labels' is a key in the loaded data dictionary
        labels = self.global_data[self.datafile][self.datafile_dataset]
        return {
            "in": None,  # No specific input data required by this head
            "out": torch.tensor(labels, dtype=torch.int64)
        }

    def set_global_data(self, head_name, global_data):
        self.head_name = head_name
        self.global_data = global_data

    def create_module(self):
        return self.module

    def get_data_length(self):
        len(self.global_data[self.datafile]['Samples'])

    def get_data_sample(self, idx):
        data_i = torch.tensor(self.global_data[self.datafile]['Samples'][idx][self.datafile_dataset],dtype=torch.float32)
        # TODO: implement..
        num_atoms = self.global_data[self.datafile]['Samples'][idx]['mol_properties_a']['numAtoms'] #self.global_data[self.datafile]['Samples'][idx]['NumAtoms']
        mask_i = create_atom_mask(data_i.shape, [num_atoms]*len(self.config['masked_dimensions']) , self.config['masked_dimensions'] )
        return { 'data': data_i, 'mask': mask_i }


    def compute_loss(self, output, target):
        return torch.nn.functional.mse_loss(output,target)


class AtomPropertiesLinearOutputHeadA(nn.Module):
    def __init__(self, seq_length, dim, internal_dim_1, internal_dim_2, num_output_layers):
        super(AtomPropertiesLinearOutputHeadA, self).__init__()
        self.module_element = LinearOutputHeadA(seq_length,dim,internal_dim_1,internal_dim_2,10,num_output_layers)
        self.module_small_ring = LinearOutputHeadA(seq_length, dim, internal_dim_1, internal_dim_2, 2, num_output_layers)
        self.module_ba = LinearOutputHeadA(seq_length,dim,internal_dim_1,internal_dim_2,4,num_output_layers)
        self.module_b1 = LinearOutputHeadA(seq_length, dim, internal_dim_1, internal_dim_2, 5, num_output_layers)
        self.module_b2 = LinearOutputHeadA(seq_length, dim, internal_dim_1, internal_dim_2, 5, num_output_layers)
        self.module_b3 = LinearOutputHeadA(seq_length, dim, internal_dim_1, internal_dim_2, 3, num_output_layers)
        self.module_h  = LinearOutputHeadA(seq_length, dim, internal_dim_1, internal_dim_2, 5, num_output_layers)

    def forward(self, x):
        x_element = self.module_element(x)
        x_sr = self.module_small_ring(x)
        x_ba = self.module_ba(x)
        x_b1 = self.module_b1(x)
        x_b2 = self.module_b2(x)
        x_b3 = self.module_b3(x)
        x_h  = self.module_h(x)
        return {"element":x_element,
                "ba":x_ba,
                "sr":x_sr,
                "b1":x_b1,
                "b2":x_b2,
                "b3":x_b3,
                "h":x_h}
class LinearAtomPropertiesOutputHeadConfig(OutputHeadConfiguration):
    def __init__(self):
        super().__init__()
        self.head_name = None
        self.global_data = None
        self.config = None
        self.module = None
        self.device = None

    def load_config(self, base_model_dim, config):
        # Configuration could include specific parameters for the head, like dimensions
        #self.output_dim = config.get('output_dim', 10)  # Default to 10 if not specified
        self.datafile   = config.get('datafile')
        self.datafile_dataset = config.get('datafile_dataset')
        self.config = config
        self.module = AtomPropertiesLinearOutputHeadA(config['seq_length'],base_model_dim,0,4,1)

    def set_global_data(self, head_name, device, global_data):
        self.head_name = head_name
        self.global_data = global_data
        self.device = device

    def create_module(self):
        return self.module

    def get_data_length(self):
        return len(self.global_data[self.datafile]['Samples'])

    def get_data_sample(self, idx):
        element_strings = [xi['element'] for xi in self.global_data[self.datafile]['Samples'][idx]["atom_properties"]]
        num_atoms = len(element_strings)
        seq_length = self.config["seq_length"]
        data_element    = torch.tensor(  convert_element_to_int( element_strings, seq_length ) ,dtype=torch.int64)
        data_small_ring = torch.tensor( [xi['is_in_small_ring'] for xi in self.global_data[self.datafile]['Samples'][idx]["atom_properties"]] + [0] * (seq_length-num_atoms)  ,dtype=torch.int64)
        data_ba = torch.tensor( [xi['num_aromatic_bonds'] for xi in self.global_data[self.datafile]['Samples'][idx]["atom_properties"]] + [0] * (seq_length-num_atoms)  ,dtype=torch.int64)
        data_b1 = torch.tensor( [xi['num_single_bonds'] for xi in self.global_data[self.datafile]['Samples'][idx]["atom_properties"]] + [0] * (seq_length-num_atoms)  ,dtype=torch.int64)
        data_b2 = torch.tensor( [xi['num_double_bonds'] for xi in self.global_data[self.datafile]['Samples'][idx]["atom_properties"]] + [0] * (seq_length-num_atoms)  ,dtype=torch.int64)
        data_b3 = torch.tensor( [xi['num_triple_bonds'] for xi in self.global_data[self.datafile]['Samples'][idx]["atom_properties"]] + [0] * (seq_length-num_atoms)  ,dtype=torch.int64)
        data_h  = torch.tensor( [xi['num_hydrogen_atoms'] for xi in self.global_data[self.datafile]['Samples'][idx]["atom_properties"]] + [0] * (seq_length-num_atoms)  ,dtype=torch.int64)

        #if( torch.any(data_ba > 3) or torch.any(data_b1 > 4) or torch.any(data_b2 > 4) or torch.any(data_b3 > 2) or torch.any(data_h > 4)):
        #    print("woah..")


        return { 'element': data_element, 'sr': data_small_ring,
                 "ba": data_ba,"b1": data_b1,"b2": data_b2,"b3": data_b3,"h":data_h}

    def compute_loss(self, output, target):
        le = F.cross_entropy(output['element'].permute(0,2,1),target['element'].to(self.device))
        lsr = F.cross_entropy(output['sr'].permute(0,2,1), target['sr'].to(self.device))
        lba = F.cross_entropy(output['ba'].permute(0,2,1), target['ba'].to(self.device))
        lb1 = F.cross_entropy(output['b1'].permute(0,2,1), torch.clamp( target['b1'].to(self.device) , min=0, max=4 ))
        lb2 = F.cross_entropy(output['b2'].permute(0,2,1), target['b2'].to(self.device))
        lb3 = F.cross_entropy(output['b3'].permute(0,2,1), target['b3'].to(self.device))
        lh  = F.cross_entropy(output['h'].permute(0,2,1), target['h'].to(self.device))
        return le+lsr+lba+lb1+lb2+lb3+lh





