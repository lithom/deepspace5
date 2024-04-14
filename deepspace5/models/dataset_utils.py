import random

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
