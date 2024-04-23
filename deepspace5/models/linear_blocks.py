import torch
from torch import float32

from deepspace5.architecture.outputhead import OutputHeadConfiguration
from deepspace5.models.dataset_utils import create_atom_mask


#@register_output_head("ExampleOutputHead")
class ExampleLinearOutputHead(OutputHeadConfiguration):
    def __init__(self):
        super().__init__()
        self.head_name = None
        self.global_data = None
        self.config = None

    def load_config(self, config):
        # Configuration could include specific parameters for the head, like dimensions
        self.output_dim = config.get('output_dim', 10)  # Default to 10 if not specified
        self.datafile   = config.get('datafile')
        self.datafile_dataset = config.get('datafile_dataset')
        self.config = config

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
        pass

    def get_data_length(self):
        len(self.global_data[self.datafile]['Samples'])

    def get_data_sample(self, idx):
        data_i = torch.tensor(self.global_data[self.datafile]['Samples'][idx][self.datafile_dataset],dtype=torch.float32)
        # TODO: implement..
        num_atoms = self.global_data[self.datafile]['Samples'][idx]['mol_properties_a']['numAtoms'] #self.global_data[self.datafile]['Samples'][idx]['NumAtoms']
        mask_i = create_atom_mask(data_i.shape, [num_atoms]*len(self.config['masked_dimensions']) , self.config['masked_dimensions'] )
        return { 'data': data_i, 'mask': mask_i }


    def compute_loss(self, output, target):
        pass

