import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


class HistogramDataset(torch.utils.data.Dataset):
    def __init__(self, conformations, max_distance=32.0, num_bins=256):
        """
        Args:
            conformations (list of np.ndarray): Each molecule contains a list of conformations.
            max_distance (float): Maximum distance for histogram generation.
            num_bins (int): Number of bins for the 1D histograms.
        """
        self.conformations = conformations
        self.max_distance = max_distance
        self.num_bins = num_bins
        self.bin_edges = np.linspace(0, max_distance, num_bins + 1)

    def __len__(self):
        return len(self.conformations)

    def __getitem__(self, idx):
        conformations = self.conformations[idx]
        num_atoms = conformations[0].shape[0]
        pairwise_distances = []

        # Collect all distances
        for conformation in conformations:
            distances = np.linalg.norm(conformation[:, None, :] - conformation[None, :, :], axis=-1)
            pairwise_distances.append(distances)

        pairwise_distances = np.array(pairwise_distances).reshape(-1, num_atoms, num_atoms)  # Flatten conformations
        histograms = np.zeros((num_atoms, num_atoms, self.num_bins))
        mean_variance = np.zeros((num_atoms, num_atoms, 2))

        # Compute histograms for each atom pair
        for i in range(num_atoms):
            for j in range(num_atoms):
                all_distances = pairwise_distances[:, i, j].flatten()
                hist, _ = np.histogram(all_distances, bins=self.bin_edges)
                histograms[i, j] = hist
                # Compute mean and variance
                mean_variance[i, j, 0] = np.mean(all_distances)
                mean_variance[i, j, 1] = np.var(all_distances)

        histograms /= len(conformations)  # Normalize by number of conformations
        return (
            torch.tensor(histograms, dtype=torch.float32),
            torch.tensor(mean_variance, dtype=torch.float32)
        )