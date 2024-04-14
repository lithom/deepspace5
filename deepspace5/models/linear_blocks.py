import torch

from deepspace5.architecture.outputhead import OutputHeadConfiguration


#@register_output_head("ExampleOutputHead")
class ExampleLinearOutputHead(OutputHeadConfiguration):
    def __init__(self):
        super().__init__()

    def load_config(self, config):
        # Configuration could include specific parameters for the head, like dimensions
        self.output_dim = config.get('output_dim', 10)  # Default to 10 if not specified

    def create_data_sample(self, data, idx: int):
        # Assume 'labels' is a key in the loaded data dictionary
        labels = data['MainChemistryData']['labels']
        return {
            "in": None,  # No specific input data required by this head
            "out": torch.tensor(labels, dtype=torch.int64)
        }

    def create_module(self):
        pass

    def get_data_length(self):
        pass

    def get_data_sample(self, idx):
        pass

    def compute_loss(self, output, target):
        pass

