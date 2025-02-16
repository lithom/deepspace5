import torch
import torch.nn as nn
import numpy as np
from torch.nn import Linear
from torch.nn.functional import softplus

from deepspace6.embeddings.basic_embeddings import DSAbstractEmbeddingPart


class ApproximateDistanceEmbeddingPart2(DSAbstractEmbeddingPart):
    def __init__(self, constants):
        super().__init__(constants)
        self.DIST_SCALING = constants.DIST_SCALING  # (Not currently used; consider using it in _scale_distance() if needed)
        self.criterion = nn.MSELoss(reduction='none')  # Element-wise mean squared error

    def tensor_size(self):
        # The output is a distance matrix of shape (MAX_ATOMS, MAX_ATOMS)
        return (self.constants.MAX_ATOMS, self.constants.MAX_ATOMS)

    def create_vector_input(self, mol, order):
        """
        Generate a tensor of approximate distances based on a specific atom order.
        :param mol: RDKit molecule object
        :param order: List of length MAX_ATOMS, containing indices for atom order or -1 for no atom.
        :return: Tensor of shape (MAX_ATOMS, MAX_ATOMS)
        """
        MAX_ATOMS = self.constants.MAX_ATOMS
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

        mask = torch.ones(self.constants.MAX_ATOMS,self.constants.MAX_ATOMS)
        return distance_tensor, mask

    def create_vector_target(self, mol):
        """
        Generate a tensor of approximate distances in canonical order.
        :param mol: RDKit molecule object
        :return: Tensor of shape (MAX_ATOMS, MAX_ATOMS)
        """
        MAX_ATOMS = self.constants.MAX_ATOMS
        distance_tensor = torch.zeros((MAX_ATOMS, MAX_ATOMS), dtype=torch.float32)
        distances = self._compute_distances(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            if i >= MAX_ATOMS:
                break
            for j, _ in enumerate(mol.GetAtoms()):
                if j >= MAX_ATOMS:
                    break
                dist = distances[i][j]
                distance_tensor[i, j] = self._scale_distance(dist)
        return distance_tensor

    def eval_loss(self, target, output, mask, lightning_module=None, is_val=False, global_index=None):
        """
        Compute loss for approximate distance predictions.
        :param target: Target tensor of shape (batch_size, MAX_ATOMS, MAX_ATOMS)
        :param output: Output tensor of shape (batch_size, MAX_ATOMS, MAX_ATOMS)
        :param mask: Mask tensor of shape (batch_size, MAX_ATOMS)
        :return: Loss tensor of shape (batch_size,)
        """
        # Broadcast mask to (batch_size, MAX_ATOMS, MAX_ATOMS)
        #mask = mask.unsqueeze(1) * mask.unsqueeze(2)
        loss = self.criterion(output, target)
        loss = loss * mask
        loss = loss.sum(dim=(2)) / mask.sum(dim=(2)).clamp(min=1)
        return loss

    def _compute_distances(self, mol):
        """
        Compute graph distances between all atom pairs in the molecule.
        :param mol: RDKit molecule object
        :return: 2D list (or numpy array) of distances
        """
        from rdkit.Chem.rdmolops import GetDistanceMatrix
        return GetDistanceMatrix(mol)

    def _scale_distance(self, dist):
        """
        Scale the distance to the (-1, 1) interval.
        Currently scales using MAX_ATOMS; if desired, you can incorporate self.DIST_SCALING.
        :param dist: Original distance
        :return: Scaled distance
        """
        return (dist / self.constants.MAX_ATOMS) * 2 - 1





class RingStatusEmbeddingPart2(DSAbstractEmbeddingPart):
    def __init__(self, constants):
        super().__init__(constants)
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')  # Element-wise binary loss

    def tensor_size(self):
        # The output is a tensor of shape (2, MAX_ATOMS)
        return (self.constants.MAX_ATOMS , 2)

    def create_vector_input(self, mol, order):
        """
        Generate a tensor of ring status flags based on a specific atom order.
        :param mol: RDKit molecule object
        :param order: List of length MAX_ATOMS (indices for atom order or -1 for no atom)
        :return: Tensor of shape (2, MAX_ATOMS)
        """
        MAX_ATOMS = self.constants.MAX_ATOMS
        ring_status_tensor = torch.zeros((2, MAX_ATOMS), dtype=torch.float32)
        for i, atom_idx in enumerate(order):
            if atom_idx == -1 or atom_idx >= mol.GetNumAtoms():
                continue
            atom = mol.GetAtomWithIdx(atom_idx)
            is_in_small_ring = False
            is_in_large_ring = False
            if atom.IsInRing():  # Note the method call
                for xi in range(3, self.constants.MAX_SMALL_RING + 1):
                    is_in_small_ring = atom.IsInRingSize(xi)
                    if is_in_small_ring:
                        break
                for xi in range(self.constants.MAX_SMALL_RING + 1, self.constants.MAX_LARGE_RING_TEST):
                    is_in_large_ring = atom.IsInRingSize(xi)
                    if is_in_large_ring:
                        break
            ring_status_tensor[0, i] = 1.0 if is_in_small_ring else 0.0
            ring_status_tensor[1, i] = 1.0 if is_in_large_ring else 0.0

        mask = torch.ones(MAX_ATOMS,2)
        ring_status_tensor = ring_status_tensor.permute(1,0)
        return ring_status_tensor , mask

    def create_vector_target(self, mol):
        """
        Generate a tensor of ring status flags in canonical order.
        :param mol: RDKit molecule object
        :return: Tensor of shape (2, MAX_ATOMS)
        """
        MAX_ATOMS = self.constants.MAX_ATOMS
        ring_status_tensor = torch.zeros((2, MAX_ATOMS), dtype=torch.float32)
        for i, atom in enumerate(mol.GetAtoms()):
            if i >= MAX_ATOMS:
                break
            is_in_small_ring = False
            is_in_large_ring = False
            if atom.IsInRing():
                for xi in range(3, self.constants.MAX_SMALL_RING + 1):
                    is_in_small_ring = atom.IsInRingSize(xi)
                    if is_in_small_ring:
                        break
                for xi in range(self.constants.MAX_SMALL_RING + 1, self.constants.MAX_LARGE_RING_TEST):
                    is_in_large_ring = atom.IsInRingSize(xi)
                    if is_in_large_ring:
                        break
            ring_status_tensor[0, i] = 1.0 if is_in_small_ring else 0.0
            ring_status_tensor[1, i] = 1.0 if is_in_large_ring else 0.0
        return ring_status_tensor

    def eval_loss(self, target, output, mask, lightning_module=None, is_val=False, global_index=None):
        """
        Compute loss for ring status predictions.
        :param target: Target tensor of shape (batch_size, 2, MAX_ATOMS)
        :param output: Output tensor of shape (batch_size, 2, MAX_ATOMS)
        :param mask: Mask tensor of shape (batch_size, MAX_ATOMS)
        :return: Loss tensor of shape (batch_size,)
        """
        #mask = mask.unsqueeze(1)  # Expand mask for both ring flags
        loss = self.criterion(output, target)
        loss = loss * mask
        loss = loss.sum(dim=(2)) / mask.sum(dim=(2)).clamp(min=1)
        return loss





import torch
import torch.nn.functional as F

class SymmetryRankEmbeddingPart2(DSAbstractEmbeddingPart):
    def __init__(self, constants):
        super().__init__(constants)
        # No fixed loss module; we will use F.kl_div in eval_loss.

    def tensor_size(self):
        # Output shape will be (MAX_ATOMS, MAX_ATOMS) where for each atom (first dim)
        # we output a probability distribution over MAX_ATOMS rank categories.
        return (self.constants.MAX_ATOMS, self.constants.MAX_ATOMS)

    def create_vector_input(self, mol, order):
        """
        Generate a one-hot distribution tensor for symmetry rank based on a given atom order.
        Returns a tuple: (tensor, mask) both of shape (MAX_ATOMS, MAX_ATOMS)
        The tensor is organized as (atom_position, rank_category).
        """
        MAX_ATOMS = self.constants.MAX_ATOMS
        # Initialize one-hot tensor and mask.
        sym_tensor = torch.zeros((MAX_ATOMS, MAX_ATOMS), dtype=torch.float32)
        mask = torch.zeros((MAX_ATOMS, MAX_ATOMS), dtype=torch.float32)
        for pos, atom_idx in enumerate(order):
            if atom_idx == -1 or atom_idx >= mol.GetNumAtoms():
                continue
            atom = mol.GetAtomWithIdx(atom_idx)
            rank = int(atom.GetProp("_CIPRank")) if atom.HasProp("_CIPRank") else -1
            if 0 <= rank < MAX_ATOMS:
                # For the atom at position 'pos', set the one-hot entry at index 'rank'
                sym_tensor[pos, rank] = 1.0
                mask[pos, :] = 1.0  # Mark entire category vector as valid.
        return sym_tensor, mask

    def create_vector_target(self, mol):
        """
        Generate the target tensor (one-hot distributions) for symmetry rank in canonical order.
        Returns a tuple: (tensor, mask) with shape (MAX_ATOMS, MAX_ATOMS)
        """
        MAX_ATOMS = self.constants.MAX_ATOMS
        sym_tensor = torch.zeros((MAX_ATOMS, MAX_ATOMS), dtype=torch.float32)
        mask = torch.zeros((MAX_ATOMS, MAX_ATOMS), dtype=torch.float32)
        for pos, atom in enumerate(mol.GetAtoms()):
            if pos >= MAX_ATOMS:
                break
            rank = int(atom.GetProp("_CIPRank")) if atom.HasProp("_CIPRank") else -1
            if 0 <= rank < MAX_ATOMS:
                sym_tensor[pos, rank] = 1.0
                mask[pos, :] = 1.0
        return sym_tensor, mask

    def eval_loss(self, target, output, mask, lightning_module=None, is_val=False, global_index=None):
        """
        Compute KL divergence loss for symmetry rank predictions.
        Both target and output have shape (batch, MAX_ATOMS, MAX_ATOMS).
        We average over all dimensions except the first non-batch (atom) dimension,
        yielding an average per atom-position loss.
        """
        # Convert output to log-probabilities along the rank categories (dim=2)
        log_probs = F.log_softmax(output, dim=2)
        # Compute elementwise KL divergence. (Note: target is assumed to be a one-hot distribution.)
        kl = F.kl_div(log_probs, target, reduction='none')
        # Sum over the rank category dimension to get per-atom losses.
        loss_per_atom = kl.sum(dim=2)
        return 10.0 * loss_per_atom
        # Use the mask (which is of shape (batch, MAX_ATOMS, MAX_ATOMS)) to get a per-atom mask.
        # Since the mask is identical across the category dimension, we take, e.g., the first channel.
        # mask_per_atom = mask[:, :, 0]
        # Average over valid atom positions.
        #loss_per_sample = (loss_per_atom * mask_per_atom).sum(dim=1) / mask_per_atom.sum(dim=1).clamp(min=1)
        #return loss_per_sample





class PharmacophoreFlagsEmbeddingPart2(DSAbstractEmbeddingPart):
    def __init__(self, constants):
        super().__init__(constants)
        self.criterion = torch.nn.BCEWithLogitsLoss(reduction='none')  # Element-wise binary loss

    def tensor_size(self):
        # We want output shape: (MAX_ATOMS, 5) where MAX_ATOMS is the first non-batch dimension.
        return (self.constants.MAX_ATOMS, 5)

    def create_vector_input(self, mol, order):
        """
        Generate a tensor of pharmacophore flags for a given atom order.
        Returns (tensor, mask), each of shape (MAX_ATOMS, 5).
        """
        MAX_ATOMS = self.constants.MAX_ATOMS
        # Original tensor computed as (5, MAX_ATOMS); we transpose it.
        pharm_tensor = torch.zeros((5, MAX_ATOMS), dtype=torch.float32)
        mask = torch.zeros((5, MAX_ATOMS), dtype=torch.float32)
        for i, atom_idx in enumerate(order):
            if atom_idx == -1 or atom_idx >= mol.GetNumAtoms():
                continue
            atom = mol.GetAtomWithIdx(atom_idx)
            pharm_tensor[0, i] = 1.0 if atom.GetBoolProp("donor") else 0.0
            pharm_tensor[1, i] = 1.0 if atom.GetBoolProp("acceptor") else 0.0
            pharm_tensor[2, i] = 1.0 if atom.GetBoolProp("aromatic") else 0.0
            pharm_tensor[3, i] = 1.0 if atom.GetBoolProp("pos_charge") else 0.0
            pharm_tensor[4, i] = 1.0 if atom.GetBoolProp("neg_charge") else 0.0
            mask[:, i] = 1.0
        # Transpose to shape (MAX_ATOMS, 5)
        return pharm_tensor.transpose(0, 1), mask.transpose(0, 1)

    def create_vector_target(self, mol):
        """
        Generate the target tensor of pharmacophore flags in canonical order.
        Returns (tensor, mask), each of shape (MAX_ATOMS, 5).
        """
        MAX_ATOMS = self.constants.MAX_ATOMS
        pharm_tensor = torch.zeros((5, MAX_ATOMS), dtype=torch.float32)
        mask = torch.zeros((5, MAX_ATOMS), dtype=torch.float32)
        for i, atom in enumerate(mol.GetAtoms()):
            if i >= MAX_ATOMS:
                break
            pharm_tensor[0, i] = 1.0 if atom.GetBoolProp("donor") else 0.0
            pharm_tensor[1, i] = 1.0 if atom.GetBoolProp("acceptor") else 0.0
            pharm_tensor[2, i] = 1.0 if atom.GetBoolProp("aromatic") else 0.0
            pharm_tensor[3, i] = 1.0 if atom.GetBoolProp("pos_charge") else 0.0
            pharm_tensor[4, i] = 1.0 if atom.GetBoolProp("neg_charge") else 0.0
            mask[:, i] = 1.0
        return pharm_tensor.transpose(0, 1), mask.transpose(0, 1)

    def eval_loss(self, target, output, mask, lightning_module=None, is_val=False, global_index=None):
        """
        Compute loss for pharmacophore flag predictions.
        Both target and output have shape (batch, MAX_ATOMS, 5).
        The loss is averaged over all dimensions except the atom-position dimension,
        returning an average loss per atom position.
        """
        loss = self.criterion(output, target)  # shape: (batch, MAX_ATOMS, 5)
        # Average loss over the channel dimension (5) to get per-atom loss.
        loss_per_atom = loss.mean(dim=2)  # shape: (batch, MAX_ATOMS)
        # Create a per-atom mask by assuming the mask is identical across channels.
        mask_per_atom = mask[:, :, 0]  # shape: (batch, MAX_ATOMS)
        # Average loss over valid atom positions.
        loss_per_sample = (loss_per_atom * mask_per_atom).sum(dim=1) / mask_per_atom.sum(dim=1).clamp(min=1)
        return loss_per_sample





from rdkit.Chem import rdchem
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridizationEmbeddingPart(DSAbstractEmbeddingPart):
    def __init__(self, constants):
        super().__init__(constants)
        # Define a mapping from RDKit hybridization to indices.
        # Allowed types: SP, SP2, SP3; any other type goes to index 3.
        self.hybridization_map = {
            rdchem.HybridizationType.SP: 0,
            rdchem.HybridizationType.SP2: 1,
            rdchem.HybridizationType.SP3: 2,
        }
        self.num_classes = len(self.hybridization_map) + 1  # extra class for "other"/unknown
        # Use cross-entropy loss (reduction 'none' to allow per-atom averaging)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def tensor_size(self):
        # Output tensor will have shape (MAX_ATOMS, num_classes)
        return (self.constants.MAX_ATOMS, self.num_classes)

    def create_vector_input(self, mol, order):
        """
        Compute a one-hot encoded hybridization state for atoms in the given order.
        Returns:
            tensor: shape (MAX_ATOMS, num_classes)
            mask: same shape, with 1.0 for valid atom positions and 0.0 otherwise.
        """
        MAX_ATOMS = self.constants.MAX_ATOMS
        hybrid_tensor = torch.zeros((MAX_ATOMS, self.num_classes), dtype=torch.float32)
        mask = torch.zeros((MAX_ATOMS, self.num_classes), dtype=torch.float32)
        for pos, atom_idx in enumerate(order):
            if atom_idx == -1 or atom_idx >= mol.GetNumAtoms():
                continue
            atom = mol.GetAtomWithIdx(atom_idx)
            hyb = atom.GetHybridization()
            class_idx = self.hybridization_map.get(hyb, self.num_classes - 1)
            hybrid_tensor[pos, class_idx] = 1.0
            mask[pos, :] = 1.0  # entire one-hot vector is valid
        return hybrid_tensor, mask

    def create_vector_target(self, mol):
        """
        Compute target one-hot hybridization for atoms in canonical order.
        Returns:
            tensor: shape (MAX_ATOMS, num_classes)
            mask: same shape.
        """
        MAX_ATOMS = self.constants.MAX_ATOMS
        hybrid_tensor = torch.zeros((MAX_ATOMS, self.num_classes), dtype=torch.float32)
        mask = torch.zeros((MAX_ATOMS, self.num_classes), dtype=torch.float32)
        num_atoms = mol.GetNumAtoms()
        for pos in range(min(num_atoms, MAX_ATOMS)):
            atom = mol.GetAtomWithIdx(pos)
            hyb = atom.GetHybridization()
            class_idx = self.hybridization_map.get(hyb, self.num_classes - 1)
            hybrid_tensor[pos, class_idx] = 1.0
            mask[pos, :] = 1.0
        return hybrid_tensor, mask

    def eval_loss(self, target, output, mask, lightning_module=None, is_val=False, global_index=None):
        """
        Compute cross-entropy loss for hybridization predictions.
        target and output: shape (batch, MAX_ATOMS, num_classes)
        The loss is computed per atom (averaging over the class dimension) and then
        averaged over valid atom positions.
        """
        batch, MAX_ATOMS, _ = output.shape
        # Convert one-hot target to label indices.
        true_labels = target.argmax(dim=2)  # shape (batch, MAX_ATOMS)
        # Reshape for cross entropy: (batch*MAX_ATOMS, num_classes) vs (batch*MAX_ATOMS,)
        loss = self.criterion(output.view(-1, self.num_classes), true_labels.view(-1))
        loss = loss.view(batch, MAX_ATOMS)
        return loss
        # Create a per-atom mask (assume mask is identical across class dimension)
        #mask_per_atom = mask[:, :, 0]
        #loss_per_sample = (loss * mask_per_atom).sum(dim=1) / mask_per_atom.sum(dim=1).clamp(min=1)
        #return loss_per_sample



import torch
import torch.nn as nn

class LocalGraphFeaturesEmbeddingPart(DSAbstractEmbeddingPart):
    def __init__(self, constants):
        super().__init__(constants)
        self.criterion = nn.MSELoss(reduction='none')
        self.num_features = 2  # degree and closeness centrality

    def tensor_size(self):
        # Output shape: (MAX_ATOMS, 2)
        return (self.constants.MAX_ATOMS, self.num_features)

    def create_vector_input(self, mol, order):
        """
        Compute graph topology features (degree and closeness centrality) for a given atom order.
        Returns:
            features: tensor of shape (MAX_ATOMS, 2)
            mask: tensor of shape (MAX_ATOMS, 2)
        """
        MAX_ATOMS = self.constants.MAX_ATOMS
        features = torch.zeros((MAX_ATOMS, self.num_features), dtype=torch.float32)
        mask = torch.zeros((MAX_ATOMS, self.num_features), dtype=torch.float32)
        from rdkit.Chem.rdmolops import GetDistanceMatrix
        dist_matrix = GetDistanceMatrix(mol)
        num_atoms = mol.GetNumAtoms()
        # Pre-compute closeness centrality for atoms in canonical order.
        closeness = [0.0] * num_atoms
        for i in range(num_atoms):
            dsum = dist_matrix[i].sum()
            closeness[i] = (num_atoms - 1) / dsum if dsum > 0 else 0.0
        for pos, atom_idx in enumerate(order):
            if atom_idx == -1 or atom_idx >= num_atoms:
                continue
            atom = mol.GetAtomWithIdx(atom_idx)
            degree = float(len(atom.GetNeighbors())) / 4.0
            features[pos, 0] = degree
            features[pos, 1] = closeness[atom_idx]
            mask[pos, :] = 1.0
        return features, mask

    def create_vector_target(self, mol):
        """
        Compute target features (degree and closeness centrality) in canonical order.
        Returns:
            features: tensor of shape (MAX_ATOMS, 2)
            mask: tensor of shape (MAX_ATOMS, 2)
        """
        MAX_ATOMS = self.constants.MAX_ATOMS
        features = torch.zeros((MAX_ATOMS, self.num_features), dtype=torch.float32)
        mask = torch.zeros((MAX_ATOMS, self.num_features), dtype=torch.float32)
        from rdkit.Chem.rdmolops import GetDistanceMatrix
        dist_matrix = GetDistanceMatrix(mol)
        num_atoms = mol.GetNumAtoms()
        closeness = [0.0] * num_atoms
        for i in range(num_atoms):
            dsum = dist_matrix[i].sum()
            closeness[i] = (num_atoms - 1) / dsum if dsum > 0 else 0.0
        for i, atom in enumerate(mol.GetAtoms()):
            if i >= MAX_ATOMS:
                break
            degree = float(len(atom.GetNeighbors())) / 4.0
            features[i, 0] = degree
            features[i, 1] = closeness[i]
            mask[i, :] = 1.0
        return features, mask

    def eval_loss(self, target, output, mask, lightning_module=None, is_val=False, global_index=None):
        """
        Compute MSE loss for graph topology features.
        target, output, mask: shape (batch, MAX_ATOMS, 2).
        The loss is averaged over the feature dimension for each atom, then averaged over valid atom positions.
        """
        loss = self.criterion(output, target)  # shape: (batch, MAX_ATOMS, 2)
        loss_per_atom = loss.mean(dim=2)  # shape: (batch, MAX_ATOMS)
        return loss_per_atom
        #mask_per_atom = mask[:, :, 0]  # assume same across features
        #loss_per_sample = (loss_per_atom * mask_per_atom).sum(dim=1) / mask_per_atom.sum(dim=1).clamp(min=1)
        #return loss_per_sample

