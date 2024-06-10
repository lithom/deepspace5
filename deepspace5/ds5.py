import json
import torch
import pickle
import sys
import time
from torch.utils.data import DataLoader, Dataset, random_split

from deepspace5.architecture.dataset import CombinedDataset
from deepspace5.ds5optim import create_optimizer
from deepspace5.ds5persistence import save_model_checkpoint, initialize_model_from_checkpoint
from deepspace5.models.linear_blocks import ExampleLinearOutputHead, LinearAtomPropertiesOutputHeadConfig
from deepspace5.models.transformer_blocks import ExampleTransformerBaseModel
from deepspace5.models.transformer_output_blocks import ConfigTransformerOutputHeadA, ConfigTransformerOutputHeadA_ForAdjacency

output_head_registry = {}
base_model_registry = {}

def init_base_model_registry():
    base_model_registry['ExampleTransformerBaseModel'] = ExampleTransformerBaseModel

def init_output_head_registry():
    output_head_registry['ExampleLinearOutputHead'] = ExampleLinearOutputHead
    output_head_registry['TransformerOutputHeadA'] = ConfigTransformerOutputHeadA
    output_head_registry['LinearAtomPropertiesOutputHeadA'] = LinearAtomPropertiesOutputHeadConfig
    output_head_registry['TransformerOutputHeadA_ForAdjacency'] = ConfigTransformerOutputHeadA_ForAdjacency


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

# Count the number of parameters for a specific module
def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def main_train(config_path):

    device = torch.device("cuda")

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
    base_model_dim = base_model_config['model_dim']

    # Load global data
    data = load_global_data(config['Data'])

    # Set data in base model
    base_model.set_global_data(data)

    # Create the Module
    base_model_module = base_model.create_module().to(device)

    # Initialize and load output heads
    output_heads = []
    for head_name, head_config in config['OutputHeadConfig'].items():
        head_class = output_head_registry[head_config['type']]
        head = head_class()
        head.load_config(base_model_dim, head_config)
        head.set_global_data(head_name,device,data)
        head_full = {'module': head.create_module().to(device),
                            'config': head}
        output_heads.append(head_full)
        #output_head_modules.append(head.create_model())

    # Load model parameters
    if( "ModelParameters" in config ):
        if( config["ModelParameters"] ):
            initialize_model_from_checkpoint(config["ModelParameters"],base_model_module, output_heads)



    # Create datasets
    dataset = CombinedDataset(data,base_model,output_heads)

    # Create DataLoader:
    total_size = len(dataset)
    val_size = int(config["Training"]["validation_split"] * total_size)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    batch_size = config["Training"]["batch_size"]
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Create optimizer
    optimizer, scheduler = create_optimizer(config['Optimizer'])

    # Add Module Parameters for optimization:
    if base_model_config['train']:
        optimizer.add_param_group({'params': base_model_module.parameters()})
        print(f'Add optim parameters: base model ({count_parameters(base_model_module)})')
    for head_i in output_heads:
        head_config = head_i['config']
        if( head_config.config['train'] ):
            optimizer.add_param_group({'params': head_i['module'].parameters()})
            print(f'Add optim parameters: {head_i['config'].head_name} ({count_parameters(head_i['module'])})')
    #optimizer = optimizer.to(device)

    # Main training loop (old)
    if False:
        for epoch in range(200):
            for xi in data_loader:
                input_batch = xi
                output_batch = xi
                tensor_base_in = input_batch['base']['in'].to(device)
                base_model_output = base_model_module(tensor_base_in, tensor_base_in)
                losses = {}
                for head_idx, head_i in enumerate(output_heads):
                    head_name = head_i['config'].head_name
                    output_i = head_i['module'](base_model_output)
                    loss_i = head_i['config'].compute_loss(output_i,output_batch[head_name])
                    losses[head_i['config'].head_name] = loss_i
                total_loss = sum(losses.values())
                total_loss.backward()  # Assume an optimizer is defined and omitted for brevity
                print(f'loss: {total_loss}')
                optimizer.step()
                optimizer.zero_grad()

    # Main training loop
    for epoch in range(200):
        # Training
        head_losses = {head['config'].head_name: 0.0 for head in output_heads}  # Initialize per-head loss tracking
        batch_count = 0

        for head_idx, head_i in enumerate(output_heads):
            head_i['config'].start_iteration()

        for xi in dataloader_train:
            input_batch = xi
            output_batch = xi
            tensor_base_in = input_batch['base']['in'].to(device)
            base_model_output = base_model_module(tensor_base_in, tensor_base_in)

            losses = {}
            for head_idx, head_i in enumerate(output_heads):
                head_name = head_i['config'].head_name
                output_i = head_i['module'](base_model_output)
                loss_i = head_i['config'].compute_loss(output_i, output_batch[head_name])
                losses[head_name] = loss_i
                head_losses[head_name] += loss_i.item()  # Sum losses for each head

            total_loss = sum(losses.values())
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            batch_count += 1

        # Print the average losses for each head after each epoch
        print(f"Epoch {epoch + 1} training losses:")
        for head_name, loss_sum in head_losses.items():
            print(f"  {head_name}: {loss_sum / batch_count}")

        for head_idx, head_i in enumerate(output_heads):
            head_i['config'].start_iteration()

        # Validation
        with torch.no_grad():
            val_head_losses = {head['config'].head_name: 0.0 for head in
                               output_heads}  # Initialize per-head loss tracking for validation
            val_batch_count = 0
            total_time = 0
            for xi in dataloader_val:
                start_time = time.time()
                input_batch = xi
                output_batch = xi
                tensor_base_in = input_batch['base']['in'].to(device)
                base_model_output = base_model_module(tensor_base_in, tensor_base_in)

                val_losses = {}
                for head_idx, head_i in enumerate(output_heads):
                    head_name = head_i['config'].head_name
                    output_i = head_i['module'](base_model_output)
                    loss_i = head_i['config'].compute_loss(output_i, output_batch[head_name])
                    val_losses[head_name] = loss_i
                    val_head_losses[head_name] += loss_i.item()

                val_batch_count += 1
                total_time += time.time() - start_time

            # Print validation results
            print(f"Epoch {epoch + 1} validation losses and timing:")
            for head_name, loss_sum in val_head_losses.items():
                print(f"  {head_name}: {loss_sum / val_batch_count}")
            print(f"  Average evaluation time per sample: {total_time / (val_batch_count*batch_size) :.6f} seconds")

        # Save the Model:
        print("Save model..")
        save_model_checkpoint(config, base_model_module, output_heads, epoch)


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
