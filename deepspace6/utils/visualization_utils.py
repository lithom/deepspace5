import torch
import matplotlib.pyplot as plt


def compare_histograms(tensor1, tensor2, num_pairs=4):
    """
    Compare histograms from two PyTorch tensors (32,32,64).

    Args:
        tensor1: PyTorch tensor of shape (32,32,64)
        tensor2: PyTorch tensor of shape (32,32,64)
        num_pairs: Number of random pairs to visualize for comparison
    """
    if tensor1.shape != tensor2.shape:
        raise ValueError("Both tensors must have the same shape")

    num_bins = tensor1.shape[2]
    bin_edges = torch.linspace(0, 1, num_bins)  # Adjust bin scaling if needed

    fig, axes = plt.subplots(num_pairs, num_pairs, figsize=(15, 15))
    fig.suptitle("Histogram Comparison Between Two Tensors")

    for i in range(num_pairs):
        for j in range(num_pairs):
            ax = axes[i, j]
            hist1 = tensor1[i+2, j+8].cpu().numpy()
            hist2 = tensor2[i+2, j+8].cpu().numpy()

            hist1 /= sum(abs(hist1[:]))
            hist2 /= sum(abs(hist2[:]))

            ax.plot(bin_edges, hist1, label="Tensor 1", alpha=0.7)
            ax.plot(bin_edges, hist2, label="Tensor 2", alpha=0.7)
            ax.fill_between(bin_edges, hist1, hist2, alpha=0.3)
            ax.set_title(f"Hist {i + 1},{j + 1}")
            ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

if __name__ == "__main__":
    # Example usage with random data
    tensor1 = torch.zeros(32, 32, 64)
    tensor1[:,:,10:20] = 1.0
    tensor1[:,:,20:25] = 0.25
    tensor2 = torch.zeros(32, 32, 64)
    tensor2[:,:,15:30] = 0.75
    tensor2[:,:,30:35] = 0.5

    compare_histograms(tensor1, tensor2)