import pickle
from concurrent.futures import ProcessPoolExecutor
from random import random
import numpy as np
from numpy import float16
from rdkit import Chem
from rdkit.Chem import rdmolops, AllChem, rdMolDescriptors
from rdkit.Chem.EnumerateStereoisomers import GetStereoisomerCount




def compute_dist_matrix_and_adjacency_tensor(mol, num_atoms, max_dist=4):
    adjacency_tensor = np.zeros((num_atoms, num_atoms, 4), dtype=int)
    distance_matrix = rdmolops.GetDistanceMatrix(mol)
    distance_matrix_full = np.zeros((num_atoms,num_atoms),dtype=int)
    ni = mol.GetNumAtoms()
    distance_matrix_full[:ni,:ni] = distance_matrix
    for dist in range(0,max_dist):
        for i in range(ni):
            for j in range(ni):
                if i != j:
                    distance = int(distance_matrix[i, j])
                    if distance == dist+1:  # Consider distances up to 3
                        adjacency_tensor[i, j, dist] = 1

    return distance_matrix_full, adjacency_tensor

def dfs_traversal(mol, start_atom, visited_atoms):
    """Perform depth-first search traversal starting from the given atom."""
    visited_atoms.add(start_atom.GetIdx())  # Mark the start atom as visited
    yield start_atom  # Yield the current atom
    for neighbor in start_atom.GetNeighbors():
        if neighbor.GetIdx() not in visited_atoms:
            yield from dfs_traversal(mol, neighbor, visited_atoms)

def random_dfs_order(mol):
    """Generate a random DFS traversal order for the atoms in the molecule."""
    visited_atoms = set()
    start_atom = random.choice(list(mol.GetAtoms()))
    for atom in dfs_traversal(mol, start_atom, visited_atoms):
        yield atom


def compute_atom_properties_01(mol):
    """
    Compute various atom properties for all atoms in a given molecule.

    Args:
    - mol: RDKit molecule object

    Returns:
    - A list of dictionaries, each containing properties for one atom.
    """
    atom_properties = []

    for atom in mol.GetAtoms():
        properties = {}
        properties['element'] = atom.GetSymbol()
        properties['is_in_small_ring'] = atom.IsInRingSize(3) or atom.IsInRingSize(4) or \
                                         atom.IsInRingSize(5) or atom.IsInRingSize(6) or \
                                         atom.IsInRingSize(7) or atom.IsInRingSize(8) or \
                                         atom.IsInRingSize(9) or atom.IsInRingSize(10)
        properties['num_aromatic_bonds'] = sum(
            1 for bond in atom.GetBonds() if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC)
        properties['num_hydrogen_atoms'] = atom.GetTotalNumHs()

        # Initialize bond counts
        properties['num_single_bonds'] = 0
        properties['num_double_bonds'] = 0
        properties['num_triple_bonds'] = 0

        # Count bond types
        for bond in atom.GetBonds():
            if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
                properties['num_single_bonds'] += 1
            elif bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                properties['num_double_bonds'] += 1
            elif bond.GetBondType() == Chem.rdchem.BondType.TRIPLE:
                properties['num_triple_bonds'] += 1

        atom_properties.append(properties)


    ## TODO: add rotatable bonds property..
    #for atom_prop in atom_properties:
    #    atom_prop['num_rotatable_bonds'] = num_rotatable_bonds

    return atom_properties


def generate_conformers_and_compute_statistics(mol, num_atoms, nConformers=32, num_bins=None, max_dist=20.0):
    """
    Generate n conformers for a given molecule and compute statistics for the ensemble.

    Args:
    - mol: RDKit molecule object.
    - n: Number of conformers to generate (default=32).

    Returns:
    - coords_list: A list of matrices, each containing the coordinates for a conformer.
    - distance_stats: A dictionary with keys 'mean' and 'variance', containing matrices of mean distances and variances, respectively.
    """
    # Generate n conformers
    _ = AllChem.EmbedMultipleConfs(mol, numConfs=nConformers)

    num_atoms_in_mol = mol.GetNumAtoms()

    # Prepare list to hold coordinate matrices for each conformer
    coords_list = []

    # Prepare matrices to hold cumulative distances and squared distances for computing variance
    collected_distances = np.zeros((num_atoms,num_atoms,nConformers))
    cum_distances = np.zeros((num_atoms, num_atoms))
    cum_squared_distances = np.zeros((num_atoms, num_atoms))

    for conf_id in range(nConformers):
        conf = mol.GetConformer(conf_id)
        coords_a = np.array([conf.GetAtomPosition(atom_idx) for atom_idx in range(num_atoms_in_mol)])
        coords = np.zeros((num_atoms,3),dtype=float16)
        coords[:num_atoms_in_mol,:] = coords_a
        coords_list.append(coords)
        # Compute pairwise distances for this conformer
        for i in range(num_atoms_in_mol):
            for j in range(i + 1, num_atoms_in_mol):  # Avoid redundant calculations and diagonal
                distance = np.linalg.norm(coords[i] - coords[j])
                cum_distances[i, j] += distance
                cum_squared_distances[i, j] += distance ** 2

                # Since the distance matrix is symmetric
                cum_distances[j, i] = cum_distances[i, j]
                cum_squared_distances[j, i] = cum_squared_distances[i, j]
                collected_distances[j,i,conf_id] = distance

    # Compute mean and variance of distances
    mean_distances = cum_distances / nConformers
    variance_distances = (cum_squared_distances / nConformers) - (mean_distances ** 2)


    # optionally compute histograms
    histogram_data = None
    if( not (num_bins is None) ):
        histogram_data = np.zeros((num_atoms,num_atoms,num_bins),dtype=float16)
        for i in range(num_atoms_in_mol):
            for j in range(i + 1, num_atoms_in_mol):  # Avoid redundant calculations and diagonal
                # Calculate histogram
                hist_i, bin_edges = np.histogram(collected_distances[j,i,:], bins=num_bins, range=(0, max_dist))
                # Normalize histogram counts
                normalized_counts = hist_i / np.sum(hist_i)
                histogram_data[i,j,:] = normalized_counts
                histogram_data[j,i, :] = normalized_counts


    distance_stats = {'mean': mean_distances, 'variance': variance_distances}
    if(not (num_bins is None)):
        distance_stats['histogram'] = histogram_data

    return coords_list, distance_stats


def compute_pharmacophore_features(mol):
    """
    Compute arrays indicating the presence of various pharmacophore features for each atom in a molecule.

    Args:
    - mol: RDKit molecule object.

    Returns:
    - features: A dictionary containing boolean arrays for each pharmacophore feature.
    """
    num_atoms = mol.GetNumAtoms()
    features = {
        'donor': [False] * num_atoms,
        'acceptor': [False] * num_atoms,
        'aromatic': [False] * num_atoms,
        'positive_charge': [False] * num_atoms,
        'negative_charge': [False] * num_atoms
    }

    # Identify hydrogen bond donors and acceptors
    #hbd_info = rdMolDescriptors._CalcNumHBD(mol)
    #hba_info = rdMolDescriptors._CalcNumHBA(mol)
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if atom.GetAtomicNum() == 7 or atom.GetAtomicNum() == 8:  # Nitrogen or oxygen
            if atom.GetTotalNumHs() > 0:  # Has attached hydrogens
                features['donor'][idx] = True
            features['acceptor'][idx] = True
        features['aromatic'][idx] = atom.GetIsAromatic()

        # Charges (simplified model: considering formal charge only)
        charge = atom.GetFormalCharge()
        if charge > 0:
            features['positive_charge'][idx] = True
        elif charge < 0:
            features['negative_charge'][idx] = True

    return features

def remove_isotopes(mol):
    """Remove isotope information from all atoms in the molecule."""
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)  # Set the isotope to the default value (0)
    return mol

def process_smiles(smiles, num_atoms=32, max_smiles_length=64, only_well_defined_stereoisomers = True):
    mol = Chem.MolFromSmiles(smiles)
    if(mol is None):
        return None

    if(mol.GetNumAtoms() > num_atoms):
        return None
    if(only_well_defined_stereoisomers):
        if( GetStereoisomerCount(mol) > 1):
            return None

    # preprocessing:
    mol = Chem.RemoveHs(mol)
    mol = remove_isotopes(mol)

    random_smiles = Chem.MolToRandomSmilesVect(mol,4)
    random_smiles_i = random_smiles[0]
    if(len(random_smiles_i)>max_smiles_length):
        return None
    random_mol = Chem.MolFromSmiles(random_smiles_i)
    map_mol_to_rand = []
    if random_mol.HasSubstructMatch(mol):
        map_mol_to_rand = random_mol.GetSubstructMatch(mol)
    else:
        return None

    dist_matrix, adj_tensor = compute_dist_matrix_and_adjacency_tensor(random_mol,num_atoms)
    atom_properties = compute_atom_properties_01(random_mol)
    coords_list, distance_stats = generate_conformers_and_compute_statistics(random_mol,num_atoms,num_bins=40,max_dist=20.0)
    #features = compute_pharmacophore_features(mol.GetConformer(0))

    Sample = {'smiles_original': smiles,
              'smiles_rand': random_smiles_i,
              'dist_matrix': dist_matrix,
              'adj_tensor': adj_tensor,
              'atom_properties': atom_properties,
              'coords3d': coords_list,
              'distance_state': distance_stats,
              }
    print("mkay")
    return Sample




    #print(random_smiles[0])

#smiles_a = 'COc1ccc2c(c1)[nH]c(n2)[S@@](=O)Cc1ncc(c(c1C)OC)C'
#smiles_a = 'O=C(O)CCc1nc(-c2ccc(OCc3ccc(-c4ccccc4)cc3)c(C(F)(F)F)c2)c[nH]1'
#smiles_a = "[11CH3]Oc1cc(C(O)c2cccs2)cc(O)c1-c1cc(Cl)cc(Cl)c1"
#process_smiles(smiles_a,num_atoms=48,max_smiles_length=96)

def process_wrapper(args):
    try:
        sample_i = None
        sample_i = process_smiles(*args)
    except:
        print("exception")
    return sample_i

def process_smiles_list_parallel(smiles_file, output_file, num_atoms=32, max_smiles_length=64, skip_first_line=False, max_workers=12):
    def read_smiles(file_name):
        with open(file_name, 'r') as file:
            if skip_first_line:
                next(file)  # Skip the first line
            return [line.strip() for line in file if line.strip()]


    # Read all SMILES strings from file
    smiles = read_smiles(smiles_file)
    # Limit the number of SMILES to 1000 for processing
    smiles = smiles[200000:220000]

    # Prepare arguments for processing
    args = [(smile, num_atoms, max_smiles_length) for smile in smiles]

    Samples = []
    # Use ProcessPoolExecutor to parallelize the computation
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(process_wrapper, args):
            if result is not None:
                Samples.append(result)

    print(f"Samples: {len(Samples)}")
    # Save dictionary to a file using pickle
    with open(output_file, 'wb') as f:
        pickle.dump(Samples, f)

def process_smiles_list(smiles_file, output_file, num_atoms=32, max_smiles_length=64, skip_first_line=False):
    Samples = []

    # Open the file in read mode
    cnt_lines = 0
    with open(smiles_file, 'r') as file:
        # Iterate over each line in the file
        first = True
        for line in file:
            cnt_lines += 1
            if(len(Samples) > 1000):
                break
            if(first and skip_first_line):
                first=False
                continue
            line = line.strip()
            try:
                sample_i = None
                sample_i = process_smiles(line,num_atoms,max_smiles_length)
            except:
                print("exception..")
            if(not(sample_i is None)):
                Samples.append(sample_i)
    # Save dictionary to a file using pickle
    with open(output_file, 'wb') as f:
        pickle.dump(Samples, f)


if __name__ == '__main__':
    input_a = "C:\\datasets\\chembl_size90_input_smiles.csv"
    #process_smiles_list(input_a,"dataset_a.pickle",32,64,True)
    process_smiles_list_parallel(input_a,"dataset_a_s32.pickle",32,64,True)