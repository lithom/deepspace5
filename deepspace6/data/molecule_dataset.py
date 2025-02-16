import random

import h5py
import pickle
import torch
from rdkit import Chem
from torch.utils.data import Dataset, DataLoader

import numpy as np
import multiprocessing as mp

from deepspace5.datagenerator.dataset_generator import generate_conformers_and_compute_statistics
from deepspace6.configs.ds_constants import DeepSpaceConstants




class HDF5Dataset(Dataset):
    def __init__(self, base_dataset, hdf5_path="dataset.h5", mode="w"):
        """
        Stores a dataset on disk in an HDF5 file.

        Args:
            base_dataset (Dataset): The dataset to be stored.
            hdf5_path (str): Path to the HDF5 file.
            mode (str): File mode, "w" (overwrite), "a" (append), "r" (read).
        """
        self.hdf5_path = hdf5_path
        self.mode = mode

        if mode in ["w", "a"]:
            with h5py.File(hdf5_path, mode) as f:
                for i in range(len(base_dataset)):
                    if i % 100 == 0:
                        print(f"Processing {i} / {len(base_dataset)}")

                    try:
                        element = base_dataset[i]
                        # Handle different data types
                        if isinstance(element, torch.Tensor):
                            element = element.numpy()  # Convert to NumPy for HDF5 storage
                        if isinstance(element, np.ndarray):
                            f.create_dataset(f"data_{i}", data=element)
                        else:
                            # Convert any other Python object to bytes
                            f.create_dataset(f"data_{i}", data=np.void(pickle.dumps(element)))
                    except Exception as e:
                        print(f"Error at index {i}: {e}")

        # Get the dataset length from stored keys
        with h5py.File(hdf5_path, "r") as f:
            self.length = len(f.keys())

    def __getitem__(self, idx):
        """Load the item from HDF5 storage."""
        with h5py.File(self.hdf5_path, "r") as f:
            data = f[f"data_{idx}"][()]
            if isinstance(data, np.void):  # If stored as pickled bytes, deserialize
                data = pickle.loads(data.tobytes())
            elif isinstance(data, np.ndarray):
                data = torch.tensor(data)  # Convert back to PyTorch if needed
            return data

    def __len__(self):
        """Return the number of elements in the dataset."""
        return self.length




class InMemoryDataset(Dataset):
    def __init__(self, base_dataset):
        """
        Wraps a dataset and caches all its data in memory.

        Args:
            base_dataset (Dataset): The dataset to be cached.
        """
        self.base_dataset = base_dataset
        self.cached_data = []
        for i in range(len(base_dataset)):
            if i % 100 == 0:
                print(f"{i}")
            try:
                element_i = base_dataset[i]
                self.cached_data.append(element_i)
            except Exception as e:
                print(f"An error occurred while processing index {i}: {e}")

    def __getitem__(self, idx):
        """
        Fetch the precomputed data from the cache.
        """
        return self.cached_data[idx]

    def __len__(self):
        """
        Return the length of the dataset.
        """
        return len(self.cached_data)


class MoleculeDatasetHelper:
    def __init__(self, atom_embedding_parts, bond_embedding_parts, constants):
        self.atom_embedding_parts = atom_embedding_parts
        self.bond_embedding_parts = bond_embedding_parts
        self.constants = constants

    def prepare_for_loss(self, combined_data, metadata, part_name, batch_dim = True):
        """
        Extract and unflatten data for a specific embedding part using metadata.
        :param combined_data: Combined tensor.
        :param metadata: Metadata dictionary for atom or bond embeddings.
        :param part_name: Name of the embedding part.
        :return: Unflattened tensor for the specified part.
        """
        part_info = metadata[part_name]
        start_idx = part_info["start_idx"][0]
        end_idx = part_info["end_idx"][0]
        tensor_size = [xi[0].item() for xi in part_info["tensor_size"]] # "list unrolling" necessary to get the dimensions

        if batch_dim:
            flattened_part = combined_data[:,:, start_idx:end_idx]
            return self.unflatten_tensor(flattened_part, tensor_size)
        else:
            flattened_part = combined_data[:, start_idx:end_idx]
            return self.unflatten_tensor(flattened_part, tensor_size)

    def canonical_atom_order(self, mol):
        """
        Get the canonical order of atoms for a molecule.
        :param mol: RDKit molecule object.
        :return: List of atom indices.
        """
        num_atoms = mol.GetNumAtoms()
        return list(range(num_atoms)) + [-1] * (self.constants.MAX_ATOMS - num_atoms)


    def canonical_bond_order(self, mol):
        """
        Get the canonical order of bonds for a molecule.
        :param mol: RDKit molecule object.
        :return: List of bond indices.
        """
        bonds = mol.GetBonds()
        sorted_bonds = sorted(bonds, key=lambda bond: (
            min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        ))
        return [bond.GetIdx() for bond in sorted_bonds] + [-1] * (self.constants.MAX_BONDS - len(bonds))

    def flatten_tensor(self, tensor):
        """
        Flatten a tensor to 1D per atom/bond.
        :param tensor: Original tensor.
        :param tensor_size: Shape of the unflattened tensor per atom/bond.
        :return: Flattened tensor.
        """
        return tensor.view(tensor.size()[0], -1)

    def unflatten_tensor(self, tensor, tensor_size, batch = True):
        """
        Unflatten a 1D tensor to its original shape per atom/bond.
        :param tensor: Flattened tensor.
        :param tensor_size: Shape of the unflattened tensor per atom/bond.
        :return: Unflattened tensor.
        """
        if batch:
            return tensor.view( tensor.size()[0], *tensor_size)
        else:
            return tensor.view(-1, *tensor_size)


class MoleculeDataset(Dataset):
    def __init__(self, smiles_list, atom_embedding_parts, bond_embedding_parts, constants, scramble_molecules = False, compute_conformers = False, num_scrambled = 4):
        """
        :param smiles_list: List of SMILES strings.
        :param embedding_parts: List of DSAbstractEmbeddingPart objects.
        :param constants: DeepSpaceConstants object.
        """
        self.helper = MoleculeDatasetHelper(atom_embedding_parts,bond_embedding_parts,constants)

        self.smiles_list_all = smiles_list
        self.atom_embedding_parts = atom_embedding_parts
        self.bond_embedding_parts = bond_embedding_parts
        self.constants = constants

        self.molecules_all = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]

        self.molecules_and_indeces = [ (mol,idx) for idx,mol in enumerate(self.molecules_all) if mol.GetNumAtoms() <=32 and mol.GetNumBonds()<=64 ]
        self.molecules_a = [ mol for mol,idx in self.molecules_and_indeces]


        self.molecules = []

        self.molecules_smiles = {}
        self.molecules_to_hist = {}
        self.molecules_to_3dcoords = {}

        if scramble_molecules:
            # Prepare arguments for each parallel task
            args = [(xi, mi, num_scrambled, compute_conformers) for xi, mi in enumerate(self.molecules_a)]

            # Initialize results dictionaries
            self.all_molecules = []
            self.all_molecules_smiles = {}
            self.all_molecules_to_hist = {}
            self.all_molecules_to_3dcoords = {}

            # Run in parallel using multiprocessing
            with mp.Pool(12) as pool:
                results = pool.map(self.process_molecule, args)

            # Merge results from all processes
            for result in results:
                self.all_molecules.extend(result['molecules'])
                self.all_molecules_smiles.update(result['molecules_smiles'])
                self.all_molecules_to_hist.update(result['molecules_to_hist'])
                self.all_molecules_to_3dcoords.update(result['molecules_to_3dcoords'])

            self.molecules = self.all_molecules
            self.molecules_smiles = self.all_molecules_smiles
            self.molecules_to_hist = self.all_molecules_to_hist
            self.molecules_to_3dcoords = self.all_molecules_to_3dcoords

            if False:
                # scramble..
                for xi, mi in enumerate(self.molecules_a):
                    if xi % 100 == 0:
                        print(f"{xi}")
                    for zi in range(num_scrambled):
                        try:
                            # Get the original atom indices and shuffle
                            atom_indices = list(range(mi.GetNumAtoms()))
                            random.shuffle(atom_indices)
                            # Renumber the atoms in the molecule
                            scrambled_mol = Chem.RenumberAtoms(mi, atom_indices)
                            # self.molecules[xi] = scrambled_mol

                            if compute_conformers:
                                coords_list, distance_stats = generate_conformers_and_compute_statistics(scrambled_mol, 32, num_bins=64, max_dist=32.0)
                                self.molecules_to_hist[scrambled_mol] = distance_stats
                                self.molecules_to_3dcoords[scrambled_mol] = coords_list
                            else:
                                self.molecules_to_hist[scrambled_mol] = []
                                self.molecules_to_3dcoords[scrambled_mol] = []

                            scm_smiles = Chem.MolToSmiles(scrambled_mol)
                            self.molecules.append(scrambled_mol)
                            self.molecules_smiles[scrambled_mol] = scm_smiles
                        except Exception as e:
                            print(f"An error occurred while processing index {xi} / {zi}: {e}")
        else:
            self.molecules = self.molecules_a
            #for xi, mi in enumerate(self.molecules_a):

        #random.shuffle(self.molecules)

    def process_molecule(self, args):
        xi, mi, num_scrambled, compute_conformers = args
        results = {
            'molecules': [],
            'molecules_smiles': {},
            'molecules_to_hist': {},
            'molecules_to_3dcoords': {}
        }
        try:
            for zi in range(num_scrambled):
                # Shuffle and renumber the atoms
                atom_indices = list(range(mi.GetNumAtoms()))
                random.shuffle(atom_indices)
                scrambled_mol = Chem.RenumberAtoms(mi, atom_indices)

                # Compute conformers and histograms if requested
                if compute_conformers:
                    coords_list, distance_stats = generate_conformers_and_compute_statistics(
                        scrambled_mol, 32, num_bins=64, max_dist=32.0
                    )
                    results['molecules_to_hist'][scrambled_mol] = distance_stats
                    results['molecules_to_3dcoords'][scrambled_mol] = coords_list
                else:
                    results['molecules_to_hist'][scrambled_mol] = []
                    results['molecules_to_3dcoords'][scrambled_mol] = []

                scm_smiles = Chem.MolToSmiles(scrambled_mol)
                results['molecules'].append(scrambled_mol)
                results['molecules_smiles'][scrambled_mol] = scm_smiles
        except Exception as e:
            print(f"An error occurred while processing index {xi}: {e}")

        return results


    def _canonical_atom_order(self, mol):
        return self.helper.canonical_atom_order(mol)

    def _canonical_bond_order(self, mol):
        return self.helper.canonical_bond_order(mol)

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, idx):
        mol = self.molecules[idx]
        max_atoms = self.constants.MAX_ATOMS

        if mol is None or mol.GetNumAtoms() > max_atoms:
            return None  # Skip invalid molecules

        # Get canonical orders
        atom_order = (self._canonical_atom_order(mol))
        #atom_order_2 = self._canonical_atom_order(mol)


        canonical_bond_order = self._canonical_bond_order(mol)

        # Prepare atom data
        atom_tensors = []
        atom_masks = []
        atom_metadata = {}
        offset = 0
        for part in self.atom_embedding_parts:
            atom_tensor, atom_mask = part.create_vector_input(mol, atom_order)
            flattened_tensor = self.helper.flatten_tensor(atom_tensor)
            atom_tensors.append(flattened_tensor)
            atom_masks.append(atom_mask)

            tensor_size = part.tensor_size()
            flattened_size = flattened_tensor.size(1)
            atom_metadata[part.__class__.__name__] = {
                "start_idx": offset,
                "end_idx": offset + flattened_size,
                "tensor_size": tensor_size
            }
            offset += flattened_size

        combined_atom_data = torch.cat(atom_tensors, dim=1)
        combined_atom_mask = atom_masks[0]  # Assuming all atom masks are identical

        # Prepare bond data
        bond_tensors = []
        bond_masks = []
        bond_metadata = {}
        offset = 0
        for part in self.bond_embedding_parts:
            bond_tensor, bond_mask = part.create_vector_input(mol, canonical_bond_order)
            flattened_tensor = self.helper.flatten_tensor(bond_tensor)
            bond_tensors.append(flattened_tensor)
            bond_masks.append(bond_mask)

            tensor_size = part.tensor_size()
            flattened_size = flattened_tensor.size(1)
            bond_metadata[part.__class__.__name__] = {
                "start_idx": offset,
                "end_idx": offset + flattened_size,
                "tensor_size": tensor_size
            }
            offset += flattened_size

        combined_bond_data = torch.cat(bond_tensors, dim=1)
        combined_bond_mask = bond_masks[0]  # Assuming all bond masks are identical

        return {
            "global_index": idx,
            "atom_data": combined_atom_data,
            "atom_mask": atom_masks,
            "atom_metadata": atom_metadata,
            "bond_data": combined_bond_data,
            "bond_mask": bond_masks,
            "bond_metadata": bond_metadata,
            "hist": self.molecules_to_hist.get(mol) if self.molecules_to_hist.get(mol) else [],  #self.molecules_to_hist[mol]
            "smiles": Chem.MolToSmiles(mol)
            #"hist": [] if len(self.smiles_to_hist)>0 else self.smiles_to_hist[ self.molecules_smiles[mol] ]
        }



def create_dataset(smiles_list, atom_embedding_parts, bond_embedding_parts, constants, scramble_molecules=True):
    """
    Create a DataLoader for batch processing.
    """
    dataset = MoleculeDataset(smiles_list, atom_embedding_parts, bond_embedding_parts, constants, scramble_molecules=scramble_molecules)
    return dataset, MoleculeDatasetHelper(atom_embedding_parts, bond_embedding_parts, constants)
    #return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)


def create_dataset_with_conformers(smiles_list, atom_embedding_parts, bond_embedding_parts, constants, num_scrambled = 4):
    """
    Create a DataLoader for batch processing.
    """
    dataset = MoleculeDataset(smiles_list, atom_embedding_parts, bond_embedding_parts, constants, scramble_molecules=True, compute_conformers=True, num_scrambled=num_scrambled)
    return dataset, MoleculeDatasetHelper(atom_embedding_parts, bond_embedding_parts, constants)
    #return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

# def collate_fn(batch):
#     """
#     Custom collate function to handle batching and padding.
#     """
#     batch = [b for b in batch if b is not None]
#     if not batch:
#         return None
#
#     inputs = torch.stack([b["inputs"] for b in batch])
#     targets = torch.stack([b["targets"] for b in batch])
#     masks = torch.stack([b["mask"] for b in batch])
#
#     return {"inputs": inputs, "targets": targets, "masks": masks}
