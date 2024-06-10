from torch.utils.data import Dataset, DataLoader



class CombinedDataset(Dataset):
    def __init__(self, datasets, base_model_conf, output_heads):
        """
        Initialize with multiple OutputHeadConfiguration instances.
        """
        self.datasets = datasets
        self.base_model_conf = base_model_conf
        self.output_heads = output_heads
        self.length = min([head['config'].get_data_length() for head in output_heads])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Fetch and combine samples from all output heads for the given index.
        """
        sample = {}
        sample['base'] = {'in': self.base_model_conf.create_input_data_sample(idx)}
        for output_head_i in self.output_heads:
            sample[output_head_i['config'].head_name] = output_head_i['config'].get_data_sample(idx)

        return sample