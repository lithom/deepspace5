import random

import torch


def pad_string_randomly(s, target_length=64, pad_char='y'):
    current_length = len(s)
    if current_length >= target_length:
        return s  # Return the original string if it is already long enough

    total_padding = target_length - current_length
    left_padding = random.randint(0, total_padding)  # Randomly decide how much padding goes on the left
    right_padding = total_padding - left_padding  # The rest goes on the right

    # Create the padded string
    padded_string = (pad_char * left_padding) + s + (pad_char * right_padding)
    return padded_string



element_to_class = {
    'C': 0,
    'N': 1,
    'O': 2,
    'F': 3,
    'P': 4,
    'S': 5,
    'Cl': 6,
    'Br': 7,
    'I': 8,
}

# Function to convert organic atom element identifier string to integer class
def convert_element_to_int(elements, padded_length):
    # Convert strings to integers
    integers = [element_to_class.get(element, 9) for element in elements]
    # Pad the resulting list with zeros to size 32
    padded_integers = integers + [0] * (padded_length - len(integers))
    return padded_integers


def create_atom_mask(tensor_shape, mask_lengths, masked_dimensions):
    # Create a tensor filled with ones
    mask = (torch.ones(tensor_shape,dtype=torch.int64))

    # Iterate over the specified masked dimensions
    for dim in masked_dimensions:
        # Create a mask for the current dimension
        dim_mask = torch.zeros(tensor_shape[dim])
        dim_mask[:mask_lengths[dim]] = 1

        # Expand the mask to match the tensor shape
        mask = mask * dim_mask.unsqueeze(dim)

    return mask
