import torch
from prettytable import PrettyTable

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



def kl_divergence(predicted, target, min_var=1e-4):
    """
    Compute the KL divergence between two normal distributions.

    Parameters:
    - predicted: Tensor of shape [b, x, y, 2], where [:, :, :, 0] is the mean and [:, :, :, 1] is the variance.
    - target: Tensor of shape [b, x, y, 2], where [:, :, :, 0] is the mean and [:, :, :, 1] is the variance.

    Returns:
    - kl_loss: Scalar tensor representing the KL divergence.
    """

    # Extract means and variances
    mu_pred, var_pred = predicted[..., 0], predicted[..., 1]
    mu_target, var_target = target[..., 0], target[..., 1]

    # Ensure variances are positive (for numerical stability)
    var_pred = torch.clamp(var_pred, min=min_var)
    var_target = torch.clamp(var_target, min=min_var)

    # Create a mask to exclude distributions with mean zero
    mask = (torch.abs(mu_target) > 0.001)

    # Apply the mask to exclude mean-zero distributions
    mu_pred = mu_pred[mask]
    var_pred = var_pred[mask]
    mu_target = mu_target[mask]
    var_target = var_target[mask]

    # Compute KL divergence per element
    kl_element = (
        torch.log(var_target / var_pred) +
        (var_pred + (mu_pred - mu_target) ** 2) / (2 * var_target) -
        0.5
    )

    # Compute the mean KL loss over remaining elements
    kl_loss = kl_element.mean()  # Mean over valid elements

    return kl_loss