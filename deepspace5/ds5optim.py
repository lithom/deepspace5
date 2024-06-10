import torch
from torch import optim
from torch.optim.lr_scheduler import LambdaLR, CyclicLR


def create_optimizer(config):
    optimizer_type = config['optimizer_type']
    learning_rate = config['learning_rate']
    optimizer = None
    if optimizer_type == "AdamW":
        optimizer = optim.AdamW([torch.empty(0)], lr=learning_rate)
    else:
        raise ValueError("Unsupported optimizer type: {}".format(optimizer_type))

    # Extract parameters from config
    learning_rate = config["learning_rate"]
    warmup_steps = config["warmup_steps"]
    cycle_length = config["cycle_length"]
    # Define min and max learning rates
    max_lr = learning_rate
    min_lr = learning_rate / 5.0  # Minimum learning rate is 5 times lower than the maximum
    # Linear warmup scheduler
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: min((epoch+1)/warmup_steps, 1))
    # Cyclic learning rate scheduler
    cyclic_scheduler = CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=cycle_length, mode='triangular', cycle_momentum=False)
    # Combined scheduler
    def combined_scheduler(epoch):
        if epoch < warmup_steps:
            return warmup_scheduler.get_lr()[0]
        else:
            return cyclic_scheduler.get_lr()[0]
    scheduler = LambdaLR(optimizer, lr_lambda=combined_scheduler)

    return optimizer, scheduler
