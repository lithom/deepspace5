import os

import torch


def save_model_checkpoint(config, base_model_module, output_heads, epoch):
    model_state = {
        'base_model': base_model_module.state_dict(),
        'output_heads': {head['config'].head_name: head['module'].state_dict() for head in output_heads}
    }
    # Define the file path for saving the model parameters
    epoch_model_path = os.path.join(config["Training"]["output"]["path_model"], f'model_epoch_{epoch + 1}.pth')
    # Save the model state dictionary
    torch.save(model_state, epoch_model_path)
    print(f"Model parameters saved to {epoch_model_path} after epoch {epoch + 1}")

def initialize_model_from_checkpoint(path_to_checkpoint, base_model_module, output_heads):
    """
    Initialize model parameters from a checkpoint file if available.

    Parameters:
        path_to_checkpoint (str): Path to the checkpoint file.
        base_model_module (torch.nn.Module): The base model module to be loaded.
        output_heads (list of dicts): List of output head modules and their configurations.
    """
    try:
        # Load the saved model state dictionary
        model_state = torch.load(path_to_checkpoint, map_location=torch.device('cpu'))

        # Load base model parameters if available
        if 'base_model' in model_state:
            base_model_module.load_state_dict(model_state['base_model'])
            print("Base model parameters loaded.")

        # Load parameters for each output head if available
        for head in output_heads:
            head_name = head['config'].head_name
            if head_name in model_state['output_heads']:
                head['module'].load_state_dict(model_state['output_heads'][head_name])
                print(f"Parameters for head '{head_name}' loaded.")
            else:
                print(f"No parameters found for head '{head_name}' in the checkpoint.")

    except FileNotFoundError:
        print(f"No checkpoint file found at {path_to_checkpoint}. Starting from scratch.")
    except Exception as e:
        print(f"Error loading the checkpoint: {e}")