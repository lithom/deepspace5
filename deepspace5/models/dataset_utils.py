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
