import pickle
from pathlib import Path

from prettytable import PrettyTable


def find_project_root(filename="ds6.py"):
    current = Path(__file__).resolve()
    while current != current.parent:  # Stop at the filesystem root
        if (current / filename).exists():
            return current
        current = current.parent
    raise FileNotFoundError(f"Could not find project root containing {filename}")




def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        try:
            data = pickle.load(file)  # Load the entire pickle file
            print("Pickle file loaded successfully.")
            print(f"Data type: {type(data)}")  # Print the type of the loaded data
            return data
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            return None