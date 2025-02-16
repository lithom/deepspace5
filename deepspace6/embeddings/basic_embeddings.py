import math

import torch
import torch.nn as nn

from abc import ABC, abstractmethod
from deepspace6.configs.ds_constants import DeepSpaceConstants


class DSAbstractEmbeddingPart(ABC):
    def __init__(self, constants):
        self.constants = constants  # deepspaceConstants, includes MAX_ATOMS, device config, etc.

    @abstractmethod
    def create_vector_input(self, mol, order):
        """
        Generate an input tensor and a mask for a given molecule based on a specific atom order.
        Generate an n x MAX_ATOMS PyTorch tensor for a given molecule based on a specific atom order and the
        corresponding mask.
        :param mol: RDKit molecule object
        :param order: List of length MAX_ATOMS, containing indices for atom order or -1 for no atom.
        :return: Tuple (data tensor, mask tensor) , both tensors of shape (N, MAX_ATOMS)
        """
        pass

    def create_vector_target(self, mol):
        """
        Generate an n x MAX_ATOMS PyTorch tensor for the canonical target representation of a molecule.
        :param mol: RDKit molecule object
        :return: Tensor of shape (N, MAX_ATOMS)
        """
        return self.create_vector_input(mol,range(0,mol.GetNumAtoms))

    @abstractmethod
    def tensor_size(self):
        """
        Return the size (shape) of the tensor generated per atom for this part.
        :return: Tuple of integers
        """
        pass

    def flattened_tensor_size(self):
        return math.prod( self.tensor_size()[1:])

    @abstractmethod
    def eval_loss(self, target, output, mask, lightning_module=None, is_val=False, global_index=None):
        """
        Evaluate the loss for this embedding part.
        :param target: Target tensor of shape (batch_size, N, MAX_ATOMS)
        :param output: Output tensor of shape (batch_size, N, MAX_ATOMS)
        :param mask: Mask tensor of shape (batch_size, MAX_ATOMS), where 1 indicates valid atoms.
        :return: Tensor of shape (batch_size,) with loss values for each batch element
        """
        pass


# Implementation for vertex distance embedding
class VertexDistanceEmbeddingPart(DSAbstractEmbeddingPart):
    def __init__(self, constants):
        super().__init__(constants)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')  # Binary loss for distance classes

    def tensor_size(self):
        return [self.constants.MAX_GRAPH_DIST_EXACT, self.constants.MAX_ATOMS]

    def create_vector_input(self, mol, order):
        """
        Generate a tensor of vertex distances based on a specific atom order.
        :param mol: RDKit molecule object
        :param order: List of length MAX_ATOMS, containing indices for atom order or -1 for no atom.
        :return: Tensor of shape (MAX_GRAPH_DIST_EXACT, MAX_ATOMS, MAX_ATOMS)
        """
        MAX_ATOMS = self.constants.MAX_ATOMS
        MAX_GRAPH_DIST_EXACT = self.constants.MAX_GRAPH_DIST_EXACT

        # Initialize tensor
        distance_tensor = torch.zeros((MAX_GRAPH_DIST_EXACT, MAX_ATOMS, MAX_ATOMS), dtype=torch.float32)

        for i, atom_idx in enumerate(order):
            if atom_idx == -1 or atom_idx >= mol.GetNumAtoms():
                continue

            distances = self._compute_distances(mol, atom_idx)

            for j, dist in enumerate(distances):
                if order[j] == -1 or j >= MAX_ATOMS:
                    continue
                if dist <= MAX_GRAPH_DIST_EXACT and dist > 0:
                    distance_tensor[int(dist) - 1, i, j] = 1.0

        return distance_tensor

    def create_vector_target(self, mol):
        """
        Generate a tensor of vertex distances in canonical order.
        :param mol: RDKit molecule object
        :return: Tensor of shape (MAX_GRAPH_DIST_EXACT, MAX_ATOMS, MAX_ATOMS)
        """
        MAX_ATOMS = self.constants.MAX_ATOMS
        MAX_GRAPH_DIST_EXACT = self.constants.MAX_GRAPH_DIST_EXACT

        # Initialize tensor
        distance_tensor = torch.zeros((MAX_GRAPH_DIST_EXACT, MAX_ATOMS, MAX_ATOMS), dtype=torch.float32)

        for i, atom in enumerate(mol.GetAtoms()):
            if i >= MAX_ATOMS:
                break

            distances = self._compute_distances(mol, i)

            for j, dist in enumerate(distances):
                if j >= MAX_ATOMS:
                    continue
                if dist <= MAX_GRAPH_DIST_EXACT:
                    distance_tensor[int(dist) - 1, i, j] = 1.0

        return distance_tensor

    def eval_loss(self, target, output, mask):
        """
        Compute loss for vertex distance predictions.
        :param target: Target tensor of shape (batch_size, MAX_GRAPH_DIST_EXACT, MAX_ATOMS, MAX_ATOMS)
        :param output: Output tensor of shape (batch_size, MAX_GRAPH_DIST_EXACT, MAX_ATOMS, MAX_ATOMS)
        :param mask: Mask tensor of shape (batch_size, MAX_ATOMS)
        :return: Loss tensor of shape (batch_size,)
        """
        # Broadcast mask to (batch_size, 1, MAX_ATOMS, MAX_ATOMS)
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = mask * mask.transpose(-1, -2)  # Combine masks for atom pairs

        # Compute binary loss
        loss = self.criterion(output, target)

        # Apply mask
        loss = loss * mask

        # Sum and normalize loss
        loss = loss.sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp(min=1)  # Avoid division by zero
        return loss

    def _compute_distances(self, mol, atom_idx):
        """
        Compute graph distances from the given atom to all others in the molecule.
        :param mol: RDKit molecule object
        :param atom_idx: Index of the source atom
        :return: List of graph distances to each atom
        """
        from rdkit.Chem.rdmolops import GetDistanceMatrix

        dist_matrix = GetDistanceMatrix(mol)
        return dist_matrix[atom_idx].tolist()



# Implementation for approximate distance embedding
class ApproximateDistanceEmbeddingPart(DSAbstractEmbeddingPart):
    def __init__(self, constants):
        super().__init__(constants)
        self.DIST_SCALING = constants.DIST_SCALING
        self.criterion = nn.MSELoss(reduction='none')  # Element-wise mean squared error

    def tensor_size(self):
        return [self.constants.MAX_ATOMS]

    def create_vector_input(self, mol, order):
        """
        Generate a tensor of approximate distances based on a specific atom order.
        :param mol: RDKit molecule object
        :param order: List of length MAX_ATOMS, containing indices for atom order or -1 for no atom.
        :return: Tensor of shape (MAX_ATOMS, MAX_ATOMS)
        """
        MAX_ATOMS = self.constants.MAX_ATOMS

        # Initialize tensor
        distance_tensor = torch.zeros((MAX_ATOMS, MAX_ATOMS), dtype=torch.float32)

        distances = self._compute_distances(mol)

        for i, atom_idx_i in enumerate(order):
            if atom_idx_i == -1 or atom_idx_i >= mol.GetNumAtoms():
                continue

            for j, atom_idx_j in enumerate(order):
                if atom_idx_j == -1 or atom_idx_j >= mol.GetNumAtoms():
                    continue

                dist = distances[atom_idx_i][atom_idx_j]
                distance_tensor[i, j] = self._scale_distance(dist)

        return distance_tensor

    def create_vector_target(self, mol):
        """
        Generate a tensor of approximate distances in canonical order.
        :param mol: RDKit molecule object
        :return: Tensor of shape (MAX_ATOMS, MAX_ATOMS)
        """
        MAX_ATOMS = self.constants.MAX_ATOMS

        # Initialize tensor
        distance_tensor = torch.zeros((MAX_ATOMS, MAX_ATOMS), dtype=torch.float32)

        distances = self._compute_distances(mol)

        for i, atom_i in enumerate(mol.GetAtoms()):
            if i >= MAX_ATOMS:
                break

            for j, atom_j in enumerate(mol.GetAtoms()):
                if j >= MAX_ATOMS:
                    break

                dist = distances[i][j]
                distance_tensor[i, j] = self._scale_distance(dist)

        return distance_tensor

    def eval_loss(self, target, output, mask):
        """
        Compute loss for approximate distance predictions.
        :param target: Target tensor of shape (batch_size, MAX_ATOMS, MAX_ATOMS)
        :param output: Output tensor of shape (batch_size, MAX_ATOMS, MAX_ATOMS)
        :param mask: Mask tensor of shape (batch_size, MAX_ATOMS)
        :return: Loss tensor of shape (batch_size,)
        """
        # Broadcast mask to (batch_size, MAX_ATOMS, MAX_ATOMS)
        mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # Combine masks for atom pairs

        # Compute element-wise MSE loss
        loss = self.criterion(output, target)

        # Apply mask
        loss = loss * mask

        # Sum and normalize loss
        loss = loss.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1)  # Avoid division by zero
        return loss

    def _compute_distances(self, mol):
        """
        Compute graph distances between all atom pairs in the molecule.
        :param mol: RDKit molecule object
        :return: 2D list of distances
        """
        from rdkit.Chem.rdmolops import GetDistanceMatrix

        dist_matrix = GetDistanceMatrix(mol)
        return dist_matrix

    def _scale_distance(self, dist):
        """
        Scale the distance to the (-1, 1) interval using DIST_SCALING.
        :param dist: Original distance
        :return: Scaled distance
        """
        return (dist / self.constants.MAX_ATOMS) * 2 - 1

# Implementation for ring status embedding
class RingStatusEmbeddingPart(DSAbstractEmbeddingPart):
    def __init__(self, constants):
        super().__init__(constants)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')  # Element-wise binary loss

    def tensor_size(self):
        return [2]

    def create_vector_input(self, mol, order):
        """
        Generate a tensor of ring status flags based on a specific atom order.
        :param mol: RDKit molecule object
        :param order: List of length MAX_ATOMS, containing indices for atom order or -1 for no atom.
        :return: Tensor of shape (2, MAX_ATOMS)
        """
        MAX_ATOMS = self.constants.MAX_ATOMS

        # Initialize tensor
        ring_status_tensor = torch.zeros((2, MAX_ATOMS), dtype=torch.float32)

        for i, atom_idx in enumerate(order):
            if atom_idx == -1 or atom_idx >= mol.GetNumAtoms():
                continue

            atom = mol.GetAtomWithIdx(atom_idx)
            is_in_small_ring  = False
            is_in_large_ring = False
            if(atom.IsInRing):
                for xi in range(3,self.constants.MAX_SMALL_RING+1):
                    is_in_small_ring = atom.IsInRingSize(xi)
                    if(is_in_small_ring):
                        break
                for xi in range(self.constants.MAX_SMALL_RING+1, self.constants.MAX_LARGE_RING_TEST):
                    is_in_large_ring = atom.IsInRingSize(xi)
                    if (is_in_large_ring):
                        break

            ring_status_tensor[0, i] = 1.0 if is_in_small_ring else 0.0
            ring_status_tensor[1, i] = 1.0 if is_in_large_ring else 0.0

        return ring_status_tensor

    def create_vector_target(self, mol):
        """
        Generate a tensor of ring status flags in canonical order.
        :param mol: RDKit molecule object
        :return: Tensor of shape (2, MAX_ATOMS)
        """
        MAX_ATOMS = self.constants.MAX_ATOMS

        # Initialize tensor
        ring_status_tensor = torch.zeros((2, MAX_ATOMS), dtype=torch.float32)

        for i, atom in enumerate(mol.GetAtoms()):
            if i >= MAX_ATOMS:
                break

            # atom = mol.GetAtomWithIdx(atom_idx)
            is_in_small_ring = False
            is_in_large_ring = False
            if (atom.IsInRing):
                for xi in range(3, self.constants.MAX_SMALL_RING + 1):
                    is_in_small_ring = atom.IsInRingSize(xi)
                    if (is_in_small_ring):
                        break
                for xi in range(self.constants.MAX_SMALL_RING + 1, self.constants.MAX_LARGE_RING_TEST):
                    is_in_large_ring = atom.IsInRingSize(xi)
                    if (is_in_large_ring):
                        break

            ring_status_tensor[0, i] = 1.0 if is_in_small_ring else 0.0
            ring_status_tensor[1, i] = 1.0 if is_in_large_ring else 0.0

        return ring_status_tensor

    def eval_loss(self, target, output, mask):
        """
        Compute loss for ring status predictions.
        :param target: Target tensor of shape (batch_size, 2, MAX_ATOMS)
        :param output: Output tensor of shape (batch_size, 2, MAX_ATOMS)
        :param mask: Mask tensor of shape (batch_size, MAX_ATOMS)
        :return: Loss tensor of shape (batch_size,)
        """
        # Broadcast mask to (batch_size, 2, MAX_ATOMS)
        mask = mask.unsqueeze(1)  # Expand mask for both ring flags

        # Compute binary loss
        loss = self.criterion(output, target)

        # Apply mask
        loss = loss * mask

        # Sum and normalize loss
        loss = loss.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1)  # Avoid division by zero
        return loss





# Implementation for symmetry rank embedding
class SymmetryRankEmbeddingPart(DSAbstractEmbeddingPart):
    def __init__(self, constants):
        super().__init__(constants)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')  # Element-wise binary loss

    def tensor_size(self):
        return [self.constants.MAX_ATOMS]

    def create_vector_input(self, mol, order):
        """
        Generate a tensor of symmetry rank flags based on a specific atom order.
        :param mol: RDKit molecule object
        :param order: List of length MAX_ATOMS, containing indices for atom order or -1 for no atom.
        :return: Tensor of shape (MAX_ATOMS, MAX_ATOMS)
        """
        MAX_ATOMS = self.constants.MAX_ATOMS

        # Initialize tensor
        symmetry_rank_tensor = torch.zeros((MAX_ATOMS, MAX_ATOMS), dtype=torch.float32)

        for i, atom_idx in enumerate(order):
            if atom_idx == -1 or atom_idx >= mol.GetNumAtoms():
                continue

            atom = mol.GetAtomWithIdx(atom_idx)
            rank = int(atom.GetProp("_CIPRank")) if atom.HasProp("_CIPRank") else -1

            if 0 <= rank < MAX_ATOMS:
                symmetry_rank_tensor[rank, i] = 1.0

        return symmetry_rank_tensor

    def create_vector_target(self, mol):
        """
        Generate a tensor of symmetry rank flags in canonical order.
        :param mol: RDKit molecule object
        :return: Tensor of shape (MAX_ATOMS, MAX_ATOMS)
        """
        MAX_ATOMS = self.constants.MAX_ATOMS

        # Initialize tensor
        symmetry_rank_tensor = torch.zeros((MAX_ATOMS, MAX_ATOMS), dtype=torch.float32)

        for i, atom in enumerate(mol.GetAtoms()):
            if i >= MAX_ATOMS:
                break

            rank = int(atom.GetProp("_CIPRank")) if atom.HasProp("_CIPRank") else -1

            if 0 <= rank < MAX_ATOMS:
                symmetry_rank_tensor[rank, i] = 1.0

        return symmetry_rank_tensor

    def eval_loss(self, target, output, mask):
        """
        Compute loss for symmetry rank predictions.
        :param target: Target tensor of shape (batch_size, MAX_ATOMS, MAX_ATOMS)
        :param output: Output tensor of shape (batch_size, MAX_ATOMS, MAX_ATOMS)
        :param mask: Mask tensor of shape (batch_size, MAX_ATOMS)
        :return: Loss tensor of shape (batch_size,)
        """
        # Broadcast mask to (batch_size, MAX_ATOMS, MAX_ATOMS)
        mask = mask.unsqueeze(1) * mask.unsqueeze(2)  # Combine masks for atom pairs

        # Compute binary loss
        loss = self.criterion(output, target)

        # Apply mask
        loss = loss * mask

        # Sum and normalize loss
        loss = loss.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1)  # Avoid division by zero
        return loss

# Implementation for pharmacophore flags embedding
class PharmacophoreFlagsEmbeddingPart(DSAbstractEmbeddingPart):
    def __init__(self, constants):
        super().__init__(constants)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')  # Element-wise binary loss

    def create_vector_input(self, mol, order):
        """
        Generate a tensor of pharmacophore flags based on a specific atom order.
        :param mol: RDKit molecule object
        :param order: List of length MAX_ATOMS, containing indices for atom order or -1 for no atom.
        :return: Tensor of shape (5, MAX_ATOMS)
        """
        MAX_ATOMS = self.constants.MAX_ATOMS

        # Initialize tensor
        pharmacophore_tensor = torch.zeros((5, MAX_ATOMS), dtype=torch.float32)

        for i, atom_idx in enumerate(order):
            if atom_idx == -1 or atom_idx >= mol.GetNumAtoms():
                continue

            atom = mol.GetAtomWithIdx(atom_idx)
            pharmacophore_tensor[0, i] = 1.0 if atom.GetBoolProp("donor") else 0.0
            pharmacophore_tensor[1, i] = 1.0 if atom.GetBoolProp("acceptor") else 0.0
            pharmacophore_tensor[2, i] = 1.0 if atom.GetBoolProp("aromatic") else 0.0
            pharmacophore_tensor[3, i] = 1.0 if atom.GetBoolProp("pos_charge") else 0.0
            pharmacophore_tensor[4, i] = 1.0 if atom.GetBoolProp("neg_charge") else 0.0

        return pharmacophore_tensor

    def create_vector_target(self, mol):
        """
        Generate a tensor of pharmacophore flags in canonical order.
        :param mol: RDKit molecule object
        :return: Tensor of shape (5, MAX_ATOMS)
        """
        MAX_ATOMS = self.constants.MAX_ATOMS

        # Initialize tensor
        pharmacophore_tensor = torch.zeros((5, MAX_ATOMS), dtype=torch.float32)

        for i, atom in enumerate(mol.GetAtoms()):
            if i >= MAX_ATOMS:
                break

            pharmacophore_tensor[0, i] = 1.0 if atom.GetBoolProp("donor") else 0.0
            pharmacophore_tensor[1, i] = 1.0 if atom.GetBoolProp("acceptor") else 0.0
            pharmacophore_tensor[2, i] = 1.0 if atom.GetBoolProp("aromatic") else 0.0
            pharmacophore_tensor[3, i] = 1.0 if atom.GetBoolProp("pos_charge") else 0.0
            pharmacophore_tensor[4, i] = 1.0 if atom.GetBoolProp("neg_charge") else 0.0

        return pharmacophore_tensor

    def eval_loss(self, target, output, mask):
        """
        Compute loss for pharmacophore flags predictions.
        :param target: Target tensor of shape (batch_size, 5, MAX_ATOMS)
        :param output: Output tensor of shape (batch_size, 5, MAX_ATOMS)
        :param mask: Mask tensor of shape (batch_size, MAX_ATOMS)
        :return: Loss tensor of shape (batch_size,)
        """
        # Broadcast mask to (batch_size, 5, MAX_ATOMS)
        mask = mask.unsqueeze(1)  # Expand mask for all pharmacophore flags

        # Compute binary loss
        loss = self.criterion(output, target)

        # Apply mask
        loss = loss * mask

        # Sum and normalize loss
        loss = loss.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1)  # Avoid division by zero
        return loss



def evaluate_loss(flattened_output, flattened_target, mask, embedding_parts, offsets):
    """
    Evaluate the loss for all embedding parts.
    :param flattened_output: Flattened output tensor.
    :param flattened_target: Flattened target tensor.
    :param mask: Mask tensor.
    :param embedding_parts: List of DSAbstractEmbeddingPart objects.
    :param offsets: Offset mapping for slicing.
    :return: Total loss.
    """
    total_loss = 0.0

    for part in embedding_parts:
        start, end = offsets[part]
        part_output = flattened_output[:, start:end].view(-1, part.vector_size(), mask.size(1))
        part_target = flattened_target[:, start:end].view(-1, part.vector_size(), mask.size(1))

        part_loss = part.eval_loss(part_target, part_output, mask)
        total_loss += part_loss.sum()

    return total_loss



# Example usage
if __name__ == "__main__":
    constants = DeepSpaceConstants(
        MAX_ATOMS=100,
        DIST_SCALING=1.0,
        device="cuda"
    )

    pharmacophore_flags_part = PharmacophoreFlagsEmbeddingPart(constants)

    # Simulate target, output, and mask tensors
    batch_size = 4
    target = torch.rand((batch_size, 5, constants.MAX_ATOMS))
    output = torch.rand((batch_size, 5, constants.MAX_ATOMS))
    mask = torch.randint(0, 2, (batch_size, constants.MAX_ATOMS)).float()

    loss = pharmacophore_flags_part.eval_loss(target, output, mask)
    print("Loss per batch:", loss)
