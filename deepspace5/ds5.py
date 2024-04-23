import json
import torch
import pickle
import sys
from torch.utils.data import DataLoader, Dataset

from deepspace5.architecture.dataset import CombinedDataset
from deepspace5.models.linear_blocks import ExampleLinearOutputHead
from deepspace5.models.transformer_blocks import ExampleTransformerBaseModel

output_head_registry = {}
base_model_registry = {}

def init_base_model_registry():
    base_model_registry['ExampleTransformerBaseModel'] = ExampleTransformerBaseModel

def init_output_head_registry():
    output_head_registry['ExampleLinearOutputHead'] = ExampleLinearOutputHead


def register_output_head(name):
    def register(cls):
        output_head_registry[name] = cls
        return cls
    return register

def register_base_model(name):
    def register(cls):
        base_model_registry[name] = cls
        return cls
    return register


def load_output_heads_from_config(config):
    output_heads = []
    for head_name, head_config in config["OutputHeads"].items():
        if head_name in output_head_registry:
            head_cls = output_head_registry[head_name]
            output_head = head_cls()
            output_head.loadConfig(head_config)
            output_heads.append(output_head)
        else:
            raise ValueError(f"Unknown output head: {head_name}")
    return output_heads

def load_base_model_from_config(config):
    output_heads = []
    base_model_name = config["BaseModelConfig"]["type"]
    base_model_config = config["BaseModelConfig"]["config"]
    if base_model_name in base_model_registry:
        model_cls = base_model_registry[base_model_name]
        base_model = model_cls()
        base_model.loadConfig(base_model_config)
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    return output_heads



def load_global_data(data_config):
    """
    Load data files specified in the configuration.
    Args:
        data_config (dict): Dictionary specifying filenames under data identifiers.
    Returns:
        dict: Dictionary with data identifiers as keys and loaded data as values.
    """
    data = {}
    for data_id, info in data_config.items():
        file_path = info['file']
        with open(file_path, 'rb') as f:
            data[data_id] = pickle.load(f)
    return data


def main_train(config_path):

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    init_base_model_registry()
    init_output_head_registry()

    # Load base model
    base_model_config = config['BaseModelConfig']
    base_model_class = base_model_registry[base_model_config['type']]
    base_model = base_model_class()
    base_model.load_base_model_from_config(base_model_config)

    # Load global data
    data = load_global_data(config['Data'])

    # Set data in base model
    base_model.set_global_data(data)

    # Initialize and load output heads
    output_heads = []
    for head_name, head_config in config['OutputHeadConfig'].items():
        head_class = output_head_registry[head_config['type']]
        head = head_class()
        head.load_config(head_config)
        head.set_global_data(head_name,data)
        output_heads.append(head)

    # Create datasets
    dataset = CombinedDataset(data,base_model,output_heads)

    # Create DataLoader:
    batch_size = 2
    shuffle = True
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    # Create optimizer
    optimizer = []

    # Example training loop
    for xi in data_loader:
        input_batch = xi
        output_batch = xi
        model_output = base_model.create_model()(input_batch)
        losses = [head.compute_loss(model_output, output_batch[head_idx]['out']) for head_idx, head in enumerate(output_heads)]
        total_loss = sum(losses)
        total_loss.backward()  # Assume an optimizer is defined and omitted for brevity
        optimizer.step()
        optimizer.zero_grad()

if __name__ == "__main__":
    mode = sys.argv[1].lower()

    with open("C:\\dev\\deepspace5\\data\\datasets\\dataset_s32_a1.pickle", 'rb') as f:
        data_a = pickle.load(f)
    # Initialize an empty set to store distinct characters
    distinct_characters = set()
    # Iterate through each string in the dataset
    samples = data_a['Samples']
    for val in samples:
        # Update the set with the characters from the current string
        distinct_characters.update(set(val["smiles_rand"]))

    print("Distinct characters: ", distinct_characters)
    print(f"\nNumber: {len(distinct_characters)}")

    if mode == "train":
        config_path = sys.argv[2] if len(sys.argv) > 1 else 'config.json'
        main_train(config_path)
