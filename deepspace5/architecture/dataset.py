from torch.utils.data import Dataset, DataLoader



class CombinedDataset(Dataset):
    def __init__(self, datasets, base_model_conf, output_heads):
        """
        Initialize with multiple OutputHeadConfiguration instances.
        """
        self.datasets = datasets
        self.base_model_conf = base_model_conf
        self.output_heads = output_heads
        self.length = min(head.get_length() for head in output_heads)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Fetch and combine samples from all output heads for the given index.
        """
        sample = {}
        sample['base'] = {'in': self.base_model_conf.create_sample(idx, self.datasets)}
        for name, output_head_conf in self.datasets.items():
            sample[name] = output_head_conf.create_sample(idx, self.datasets)

        return sample