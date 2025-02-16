import torch
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import GetDistanceMatrix

from abc import ABC, abstractmethod
from deepspace6.configs.ds_constants import DeepSpaceConstants
from deepspace6.embeddings.basic_embeddings import DSAbstractEmbeddingPart



# Updated AtomTypeEmbeddingPart
class AtomTypeEmbeddingPart(DSAbstractEmbeddingPart):
    def __init__(self, constants: DeepSpaceConstants, ratio_c=0.75, ratio_chiral=0.05):
        super().__init__(constants)
        device = constants.device
        self.atom_types = constants.atom_types  # List of supported atom types (e.g., ["C", "N", "O", ...])
        self.ratio_c = ratio_c  # Ratio of carbon atoms
        self.ratio_chiral = ratio_chiral  # Ratio of chiral atoms

        # Loss weights
        self.atom_types_with_noatom = ["XX"]+self.atom_types[:]
        self.atom_type_weights = torch.tensor(
            [1 / ratio_c if (t == "C" or t == "XX") else 1 / ((1 - ratio_c) / (len(self.atom_types_with_noatom) - 1)) for t in self.atom_types_with_noatom],
            dtype=torch.float32
        ).to(device)
        self.chiral_weight = torch.tensor([1 / (1 - ratio_chiral), 1 / ratio_chiral], dtype=torch.float32).to(device)

        self.cross_entropy = nn.CrossEntropyLoss(weight=self.atom_type_weights, reduction='none').to(device)
        self.binary_loss = nn.BCEWithLogitsLoss(reduction='none').to(device)

    def create_vector_input(self, mol, order):
        """
        Generate a tensor of atom types and stereochemistry flags.
        :param mol: RDKit molecule object
        :param order: List of length MAX_ATOMS, containing indices for atom order or -1 for no atom.
        :return: Tensor of shape (MAX_ATOMS, (len(atom_types) + 2))
        """
        MAX_ATOMS = self.constants.MAX_ATOMS
        atom_tensor = torch.zeros(MAX_ATOMS, (len(self.atom_types_with_noatom) + 2), dtype=torch.float32)
        mask_tensor = torch.zeros(MAX_ATOMS, (len(self.atom_types_with_noatom) + 2), dtype=torch.float32)

        for i, atom_idx in enumerate(order):
            if atom_idx == -1 or atom_idx >= mol.GetNumAtoms():
                continue

            atom = mol.GetAtomWithIdx(atom_idx)
            mask_tensor[i,:] = 1.0 # we also want to have the "no atom" information included and dont mask it..

            # Encode atom type
            atom_symbol = atom.GetSymbol()
            if atom_symbol in self.atom_types:
                atom_tensor[i, self.atom_types.index(atom_symbol) + 1 ] = 1.0 # +1!!
            else:
                atom_tensor[i, 0] = 1.0

            # Encode stereochemistry
            if atom.HasProp("_CIPCode"):
                atom_tensor[i, len(self.atom_types) +1] = 1.0  # Chiral flag
                cip_code = atom.GetProp("_CIPCode")
                if cip_code == "R":
                    atom_tensor[i, len(self.atom_types) + 2] = 0.0  # R flag
                elif cip_code == "S":
                    atom_tensor[i, len(self.atom_types) + 2] = 1.0  # S flag

        return (atom_tensor, mask_tensor)

    def create_vector_target(self, mol):
        """
        Generate the target tensor by calling create_vector_input with canonical order.
        :param mol: RDKit molecule object
        :return: Tensor of shape ((len(atom_types) + 2), MAX_ATOMS)
        """
        canonical_order = list(range(mol.GetNumAtoms())) + [-1] * (self.constants.MAX_ATOMS - mol.GetNumAtoms())
        return self.create_vector_input(mol, canonical_order)

    def eval_loss(self, target, output, mask, lightning_module=None, is_val=False, global_index=None):
        """
        Compute loss for atom type and stereochemistry predictions.
        :param target: Target tensor of shape (batch_size, len(atom_types) + 2, MAX_ATOMS)
        :param output: Output tensor of shape (batch_size, len(atom_types) + 2, MAX_ATOMS)
        :param mask: Mask tensor of shape (batch_size, MAX_ATOMS)
        :return: Loss tensor of shape (batch_size,)
        """
        #mask = mask.unsqueeze(1)  # Expand mask to match channel dimensions

        # Atom type loss (multiclass cross entropy)
        atom_type_target = target[:, : , :len(self.atom_types_with_noatom)].argmax(dim=2)  # Convert one-hot to class indices
        atom_type_output = output[:, :, :len(self.atom_types_with_noatom)].permute(0,2,1)  # For CrossEntropyLoss (N, C, D)
        atom_type_loss = self.cross_entropy(atom_type_output, atom_type_target)
        atom_type_loss = atom_type_loss #* mask

        # Chiral loss (binary cross entropy)
        chiral_target = target[:, :, len(self.atom_types_with_noatom):]
        chiral_output = output[:, :, len(self.atom_types_with_noatom):]
        chiral_loss = self.binary_loss(chiral_output, chiral_target)
        chiral_loss = chiral_loss #* mask.unsqueeze(1)  # Expand mask for two chiral flags

        if(lightning_module):
            total_atom_type_loss = atom_type_loss.sum(dim=1)
            total_chiral_loss = chiral_loss.sum(dim=(1, 2))
            # ✅ Log losses inside the Lightning module
            train_or_val = "val" if is_val else "train"
            lightning_module.log(train_or_val+"/atom_type_A_loss", total_atom_type_loss.mean(), on_step=False, on_epoch=True)
            lightning_module.log(train_or_val+"/atom_type_chiral_loss", total_chiral_loss.mean(), on_step=False, on_epoch=True)

        # Combine losses
        total_loss = atom_type_loss + chiral_loss.sum(dim=2) #(atom_type_loss.sum(dim=1) + chiral_loss.sum(dim=(1, 2))) / mask.sum(dim=1).clamp(min=1)
        return total_loss

    def tensor_size(self):
        """
        Return the tensor size for this embedding part.
        :return: Tuple (len(atom_types) +1 + 2,) , the first +1 is to explicitly label "no atom"
        """
        return (self.constants.MAX_ATOMS ,len(self.atom_types_with_noatom) + 2)


# Updated BondInfoEmbeddingPart
class BondInfoEmbeddingPart(DSAbstractEmbeddingPart):
    def __init__(self, constants : DeepSpaceConstants):
        super().__init__(constants)
        self.max_bonds = constants.MAX_BONDS

        self.bond_type_indices = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
        # self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.binary_loss_adj = nn.BCEWithLogitsLoss(reduction='none', pos_weight = torch.tensor(constants.ADJACENCY_POS_WEIGHT).to(constants.device) )
        self.binary_loss = nn.BCEWithLogitsLoss(reduction='none')

    def create_vector_input(self, mol, bond_order):
        """
        Generate a tensor of bond information and a mask.
        :param mol: RDKit molecule object
        :param bond_order: List of length MAX_BONDS, containing indices for bond order or -1 for no bond.
        :return: Tuple (data tensor, mask tensor)
        """
        MAX_BONDS = self.max_bonds
        MAX_ATOMS = self.constants.MAX_ATOMS
        bond_tensor = torch.zeros((MAX_BONDS, 2 * MAX_ATOMS + len(self.bond_type_indices) + 2), dtype=torch.float32)
        mask_tensor = torch.zeros((MAX_BONDS, 2 * MAX_ATOMS + len(self.bond_type_indices) + 2), dtype=torch.float32)

        bonds = list(mol.GetBonds())


        for i, bond_idx in enumerate(bond_order):
            if bond_idx == -1 or bond_idx >= len(bonds):
                continue

            mask_tensor[i,:] = 1.0

            bond = bonds[bond_idx]
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

            # Ensure begin_idx < end_idx for canonical order
            begin_idx, end_idx = sorted((begin_idx, end_idx))

            # Encode vertex indices as one-hot
            bond_tensor[i, begin_idx] = 1.0
            bond_tensor[i, MAX_ATOMS + end_idx] = 1.0

            # Encode bond type
            bond_type = bond.GetBondType()
            if bond_type.name in self.bond_type_indices:
                bond_tensor[i, 2 * MAX_ATOMS + self.bond_type_indices[bond_type.name]] = 1.0

            # Encode stereo information
            if bond_type == Chem.BondType.DOUBLE:
                bond_tensor[i, -2] = bond.HasProp("_Stereo")
                if bond.HasProp("_Stereo") and bond.GetStereo().name in ["E", "Z"]:
                    bond_tensor[i, -1] = 1.0 if bond.GetStereo().name == "E" else -1.0

        return bond_tensor, mask_tensor

    def eval_loss(self, target, output, mask, lightning_module=None, is_val=False, global_index=None):
        """
        Compute loss for bond information predictions.
        :param target: Target tensor of shape (batch_size, 2 * MAX_ATOMS + len(bond_types) + 2, MAX_BONDS)
        :param output: Output tensor of shape (batch_size, 2 * MAX_ATOMS + len(bond_types) + 2, MAX_BONDS)
        :param mask: Mask tensor.
        :return: Loss tensor of shape (batch_size,)
        """
        #mask = mask.unsqueeze(1)

        # Vertex index loss (bce)
        vertex_target = target[:,:, :2 * self.constants.MAX_ATOMS]
        vertex_output = output[:,:, :2 * self.constants.MAX_ATOMS]
        vertex_loss = self.binary_loss_adj(vertex_output, vertex_target) # self.cross_entropy(vertex_output.view(-1, self.constants.MAX_ATOMS), vertex_target.argmax(dim=1))

        # Bond type and stereo loss (binary cross entropy)
        bond_target = target[:,:, 2 * self.constants.MAX_ATOMS:]
        bond_output = output[:,:, 2 * self.constants.MAX_ATOMS:]
        bond_loss = self.binary_loss(bond_output, bond_target)

        # Combine losses
        combined_loss_masked = torch.cat( (vertex_loss,bond_loss),2) # here we could apply the mask, but we don't..

        if(lightning_module):
            total_vertex_loss = torch.sum( torch.flatten(vertex_loss) )
            total_bond_loss = torch.sum(torch.flatten(bond_loss))
            train_or_val = "val" if is_val else "train"
            lightning_module.log(train_or_val+"/bond_info_vertex_loss", total_vertex_loss, on_step=False, on_epoch=True)
            lightning_module.log(train_or_val+"/bond_info_bond_loss", total_bond_loss, on_step=False, on_epoch=True)

        # loss = (vertex_loss + bond_loss.sum(dim=(1, 2))) * mask
        return combined_loss_masked.sum(dim=2) # / mask.sum(dim=1).clamp(min=1)

    def tensor_size(self):
        """
        Return the tensor size for this embedding part.
        :return: Tuple (2 * MAX_ATOMS + len(bond_types) + 2,)
        """
        return (self.constants.MAX_BONDS, 2 * self.constants.MAX_ATOMS + len(self.bond_type_indices) + 2,)






def normalize_logits(M_sym, eps=1e-6):
    """
    Normalize the symmetrized matrix so it has zero mean and unit variance.
    Ensures the logits are well-distributed for BCEWithLogitsLoss.
    """
    mean = M_sym.mean(dim=(-2, -1), keepdim=True)
    std = M_sym.std(dim=(-2, -1), keepdim=True) + eps  # Prevent division by zero
    return (M_sym - mean) / std


def clip_logits(M_sym, min_val=-5, max_val=5):
    """
    Clip logits to prevent extreme values that could saturate sigmoid in BCEWithLogitsLoss.
    """
    return torch.clamp(M_sym, min_val, max_val)

def preprocess_logits(M_sym):
    """
    Full preprocessing pipeline for adjacency matrix logits before BCEWithLogitsLoss.
    """
    M_sym = normalize_logits(M_sym)  # Step 1: Normalize (zero mean, unit variance)
    M_sym = clip_logits(M_sym)  # Step 2: Prevent extreme values
    return M_sym




class VertexDistanceMapEmbeddingPart(DSAbstractEmbeddingPart):
    def __init__(self, constants):
        super().__init__(constants)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(20.0)) # much more zeros than ones..
        # Class probabilities for weighting
        max_atoms = self.constants.MAX_ATOMS
        #self.class_weights = torch.tensor([1 / (1 - 4 / max_atoms), 1 / (4 / max_atoms)], dtype=torch.float32)

    def create_vector_input(self, mol, order_a):
        """
        Generate a tensor of vertex distance maps and a mask.
        :param mol: RDKit molecule object.
        :param order: List of length MAX_ATOMS, containing indices for atom order or -1 for no atom.
        :return: Tuple (data tensor, mask tensor)
        NOTE: this was done by chatgpt, for a couple of checks it yielded the same results as the old function. In case
              that something breaks, maybe go back to old / slow version :)
        """

        order = torch.tensor(order_a, dtype=torch.int64)

        max_atoms = self.constants.MAX_ATOMS
        max_dist = self.constants.MAX_GRAPH_DIST_EXACT

        # Initialize the distance map tensor and mask
        distance_map = torch.zeros((max_atoms, max_dist, max_atoms), dtype=torch.float32)
        mask = torch.zeros((max_atoms, max_dist, max_atoms), dtype=torch.float32)

        # Convert distance matrix to tensor
        distance_matrix = torch.tensor(GetDistanceMatrix(mol), dtype=torch.int32)
        num_atoms = distance_matrix.size(0)

        # Create tensors for indexing
        valid_order = order.clone().detach() >= 0 # torch.tensor(order) >= 0 # note these are constants, not parameters
        #valid_indices = order.clone().detach()[valid_order] #torch.tensor(order)[valid_order]

        # Create masks for valid atoms
        mask[valid_order] = 1.0

        # Generate a grid of indices
        grid_i, grid_j = torch.meshgrid(torch.arange(max_atoms), torch.arange(max_atoms), indexing='ij')

        # Mask for valid pairs
        valid_pairs = (order[grid_i] >= 0) & (order[grid_j] >= 0) & (order[grid_i] < num_atoms) & (
                    order[grid_j] < num_atoms)
        mask[grid_i[valid_pairs], :, grid_j[valid_pairs]] = 1.0

        # Gather the distances only for valid pairs
        oi = order.clone().detach()[grid_i[valid_pairs]]
        oj = order.clone().detach()[grid_j[valid_pairs]]
        dist_values = distance_matrix[oi, oj]

        # Apply distance map for valid pairs
        valid_dist = (dist_values > 0) & (dist_values <= max_dist)
        distance_map[
            grid_i[valid_pairs][valid_dist], dist_values[valid_dist] - 1, grid_j[valid_pairs][valid_dist]] = 1.0

        return distance_map, mask

    def create_vector_input_old_slow(self, mol, order):
        """
        Generate a tensor of vertex distance maps and a mask.
        :param mol: RDKit molecule object.
        :param order: List of length MAX_ATOMS, containing indices for atom order or -1 for no atom.
        :return: Tuple (data tensor, mask tensor)
        """


        max_atoms = self.constants.MAX_ATOMS
        max_dist = self.constants.MAX_GRAPH_DIST_EXACT

        # Initialize the distance map tensor and mask
        distance_map = torch.zeros((max_atoms, max_dist, max_atoms), dtype=torch.float32)
        #mask = torch.zeros((max_atoms,), dtype=torch.float32)
        mask = torch.zeros((max_atoms, max_dist, max_atoms), dtype=torch.float32)

        # Get the distance matrix from RDKit
        distance_matrix = torch.tensor(GetDistanceMatrix(mol), dtype=torch.int32)
        num_atoms = distance_matrix.size(0)

        # Populate the distance map based on the distance matrix
        for i in range(max_atoms):
            for j in range(max_atoms):
                oi = order[i]
                oj = order[j]
                if(oi>=0 and oj>=0 and oi < num_atoms and oj < num_atoms):
                    mask[i,:,j] = 1.0
                    dist = distance_matrix[oi, oj]
                    if dist > 0 and dist <= max_dist:
                        distance_map[i, dist - 1, j] = 1.0
                    #else:
                    #    if (dist >= max_dist):
                    #        distance_map[i, max_dist - 1, j] = 1.0
                #else: nothing, keeps masked out..



        # Apply the mask
        for i, atom_idx in enumerate(order):
            if atom_idx != -1 and atom_idx < num_atoms:
                mask[i] = 1.0

        #dm2, mask2 = self.create_vector_input_vectorized(mol,order)
        print("mkay")

        return distance_map, mask

    def enforce_symmetry_vectorized(self, data):
        """
        Enforces symmetry for each slice [b, :, i, :] by computing M @ M^T.

        :param data: Tensor of shape (batch_size, x, i, y)
        :return: Symmetric tensor of the same shape
        """
        batch_size, x, num_i, y = data.shape
        assert x == y, "Matrix must be square (x == y) to enforce symmetry"

        # Permute so the 32x32 matrices are the last two dimensions
        data_permuted = data.permute(0, 2, 1, 3)  # Shape (batch_size, i, x, y)

        # Compute M @ M^T
        symmetric_data = torch.matmul(data_permuted, data_permuted.transpose(-1, -2))

        # Permute back to original shape
        return symmetric_data.permute(0, 2, 1, 3)  # Back to (batch_size, x, i, y)

    def enforce_symmetry_avg(self, data):
        # Permute so that the square matrices are last two dimensions
        data_permuted = data.permute(0, 2, 1, 3)  # (batch, i, x, y)
        symmetric_data = 0.5 * (data_permuted + data_permuted.transpose(-1, -2))
        return symmetric_data.permute(0, 2, 1, 3)

    def eval_loss(self, target, output, mask, lightning_module=None, is_val=False, global_index=None):
        """
        Compute the loss for vertex distance maps.
        :param target: Target tensor of shape (batch_size, max_atoms, max_dist, max_atoms).
        :param output: Output tensor of shape (batch_size, max_atoms, max_dist, max_atoms).
        :param mask: Mask tensor of shape (batch_size, max_atoms).
        :return: Loss tensor of shape (batch_size,).
        """
        max_atoms = self.constants.MAX_ATOMS
        #mask = mask.unsqueeze(1).unsqueeze(1)  # Expand mask to match target/output shape

        # Compute weighted binary cross-entropy loss
        output_symmetrized = self.enforce_symmetry_avg(output) #self.enforce_symmetry_vectorized(output)
        output_symmetrized_preprocessed = preprocess_logits(output_symmetrized)
        loss = self.bce_loss(output_symmetrized_preprocessed, target)
        #loss = self.bce_loss(output_symmetrized, target)
        weighted_loss = loss * mask

        # Apply class weights
        #positive_weight = self.class_weights[1]
        #negative_weight = self.class_weights[0]
        #weights = (target * positive_weight + (1 - target) * negative_weight)
        #weighted_loss *= weights

        # Reduce loss over all dimensions except batch
        total_distance_matrix_loss = weighted_loss.sum(dim=( 2, 3)) / (mask.sum(dim=( 2, 3)).clamp(min=1))
        if(lightning_module):
            train_or_val = "val" if is_val else "train"
            lightning_module.log(train_or_val+"/dist_matrix_loss_g1", torch.sum(torch.flatten(  weighted_loss[:,:,0,:].sum(dim=( 2)) / (mask[:,:,0,:].sum(dim=( 2)).clamp(min=1))  )), on_step=False, on_epoch=True)
            lightning_module.log(train_or_val+"/dist_matrix_loss_g2", torch.sum(torch.flatten(  weighted_loss[:,:,1,:].sum(dim=( 2)) / (mask[:,:,1,:].sum(dim=( 2)).clamp(min=1))  )), on_step=False, on_epoch=True)
            lightning_module.log(train_or_val+"/dist_matrix_loss_g3", torch.sum(torch.flatten(weighted_loss[:, :, 2, :].sum(dim=(2)) / (mask[:, :, 2, :].sum(dim=(2)).clamp(min=1)))), on_step=False, on_epoch=True)
            lightning_module.log(train_or_val+"/dist_matrix_loss_g4", torch.sum(torch.flatten(weighted_loss[:, :, 3, :].sum(dim=(2)) / (mask[:, :, 3, :].sum(dim=(2)).clamp(min=1)))), on_step=False, on_epoch=True)
            #lightning_module.log(train_or_val+"/dist_matrix_loss_g5", torch.sum(torch.flatten(weighted_loss[:, :, 4, :].sum(dim=(2)) / (mask[:, :, 4, :].sum(dim=(2)).clamp(min=1)))), on_step=False, on_epoch=True)
            #lightning_module.log(train_or_val+"/dist_matrix_loss_g6", torch.sum(torch.flatten(weighted_loss[:, :, 5, :].sum(dim=(2)) / (mask[:, :, 5, :].sum(dim=(2)).clamp(min=1)))), on_step=False, on_epoch=True)

            if( global_index is not None):
                sample_to_log = 1234
                idx = (global_index == sample_to_log).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    idx = idx[0]
                    input_matrix0 = target[idx,:,0,:].detach().cpu().unsqueeze(0)
                    input_matrix2 = target[idx, :, 2, :].detach().cpu().unsqueeze(0)
                    output_matrix0 = output_symmetrized[idx, :, 0, :].detach().cpu().unsqueeze(0)
                    output_matrix2 = output_symmetrized[idx, :, 2, :].detach().cpu().unsqueeze(0)
                    # Log the first sample prediction in TensorBoard
                    lightning_module.logger.experiment.add_image(f"input_matrix_0_{sample_to_log}", input_matrix0,
                                                     lightning_module.current_epoch)
                    lightning_module.logger.experiment.add_image(f"output_matrix_0_{sample_to_log}", output_matrix0,
                                                     lightning_module.current_epoch)
                    lightning_module.logger.experiment.add_image(f"input_matrix_2_{sample_to_log}", input_matrix2,
                                                     lightning_module.current_epoch)
                    lightning_module.logger.experiment.add_image(f"output_matrix_2_{sample_to_log}", output_matrix2,
                                                     lightning_module.current_epoch)



        return total_distance_matrix_loss

    def tensor_size(self):
        """
        Return the tensor size for the distance map embedding.
        :return: Tuple (max_atoms, max_dist, max_atoms).
        """
        return (self.constants.MAX_ATOMS, self.constants.MAX_GRAPH_DIST_EXACT, self.constants.MAX_ATOMS)

