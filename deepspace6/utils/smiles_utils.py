from rdkit import Chem
import random


def filter_smiles(smiles_file, max_number, min_atoms, max_atoms, return_original=True, return_scrambled=0):
    filtered_smiles = []

    with open(smiles_file, 'r') as file:
        smiles_list = [line.strip() for line in file]

    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            if mol.GetNumAtoms() > max_atoms or mol.GetNumAtoms()<min_atoms:
                continue

            if len(Chem.GetMolFrags(mol, asMols=False)) > 1:
                continue

            if return_original:
                filtered_smiles.append(smiles)

            for _ in range(return_scrambled):
                atom_indices = list(range(mol.GetNumAtoms()))
                random.shuffle(atom_indices)
                scrambled_mol = Chem.RenumberAtoms(mol, atom_indices)
                scrambled_smiles = Chem.MolToSmiles(scrambled_mol, canonical=False)
                filtered_smiles.append(scrambled_smiles)

            if len(filtered_smiles) >= max_number:
                break

        except Exception as e:
            print(f"Error processing {smiles}: {e}")

    return filtered_smiles[:max_number]


def create_scrambled(smiles_in, num_scrambled=4):
    scrambled_smiles = []
    mol = Chem.MolFromSmiles(smiles_in)
    atom_indices = list(range(mol.GetNumAtoms()))
    for zi in range(num_scrambled):
        random.shuffle(atom_indices)
        try:
            scrambled_mol = Chem.RenumberAtoms(mol, atom_indices)
            scrambled_smiles_i = Chem.MolToSmiles(scrambled_mol, canonical=False)
            scrambled_smiles.append(scrambled_smiles_i)
        except Exception as e:
            print(f"Error processing {smiles_in}: {e}")

    return scrambled_smiles


def filter_smiles_for_train_and_val(smiles_file, max_number, ratio_val=0.1, min_atoms=5, max_atoms=32, return_original=True, return_scrambled=0):

    with open(smiles_file, 'r') as file:
        smiles_list = [line.strip() for line in file]

    filtered_smiles = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            if mol.GetNumAtoms() > max_atoms or mol.GetNumAtoms()<min_atoms:
                continue

            if len(Chem.GetMolFrags(mol, asMols=False)) > 1:
                continue

            filtered_smiles.append(smiles)

            if len(filtered_smiles) >= max_number:
                break

        except Exception as e:
            print(f"Error processing {smiles}: {e}")


    random.shuffle(filtered_smiles)
    num_train = int( len(filtered_smiles) * (1.0-ratio_val) )
    smiles_train = filtered_smiles[0:num_train]
    smiles_val = filtered_smiles[num_train:]

    if return_original:
        return smiles_train, smiles_val

    else:
        # Apply scrambling to the training SMILES
        scrambled_train_pre = [create_scrambled(sm, return_scrambled) for sm in smiles_train]
        # Apply scrambling to the validation SMILES
        scrambled_val_pre = [create_scrambled(sm, return_scrambled) for sm in smiles_val]
        #flatten the list of lists:
        scrambled_train = [item for sublist in scrambled_train_pre for item in sublist]
        scrambled_val = [item for sublist in scrambled_val_pre for item in sublist]

        random.shuffle(scrambled_val)
        random.shuffle(scrambled_train)
        return scrambled_train, scrambled_val



from sklearn.neighbors import KDTree, BallTree


class SmilesSet:
    def __init__(self, smiles_list):
        """
        Initialize the SmilesSet with a list of SMILES strings.
        Computes binary descriptors (fingerprints) for all valid SMILES.

        Parameters:
        smiles_list (list): List of SMILES strings.
        """
        self.smiles_list = smiles_list
        self.fingerprints = []
        self.valid_smiles = []

        # Generate fingerprints for all SMILES
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256)
                arr = np.zeros((1,), dtype=np.int8)
                DataStructs.ConvertToNumpyArray(fp, arr)
                self.fingerprints.append(arr)
                self.valid_smiles.append(smiles)

        # Build KDTree for fast querying
        self.tree = BallTree(self.fingerprints, metric="hamming")

    def query(self, query_smiles, k=1):
        """
        Query the set to find the closest SMILES based on Hamming distance.

        Parameters:
        query_smiles (str): The SMILES string to query.
        k (int): Number of nearest neighbors to return.

        Returns:
        list: List of tuples containing (SMILES, distance) for the k closest matches.
        """
        # Compute the fingerprint for the query SMILES
        mol = Chem.MolFromSmiles(query_smiles)
        if not mol:
            raise ValueError(f"Invalid SMILES: {query_smiles}")

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=256)
        query_fp = np.zeros((1,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, query_fp)

        # Query the KDTree
        dist, idx = self.tree.query([query_fp], k=k)

        # Map results to SMILES and distances
        results = [(self.valid_smiles[i], dist[0][j]) for j, i in enumerate(idx[0])]
        return results

def evaluate_set_distances(set_a, set_b):
    """
    Evaluate the minimum distance from each SMILES in set B to the closest SMILES in set A.

    Parameters:
    set_a (list): Large set of SMILES strings.
    set_b (list): Smaller set of SMILES strings.

    Returns:
    dict: A dictionary with statistics (min, max, mean, std) of distances.
    """
    # Create SmilesSet for set A
    smiles_set_a = SmilesSet(set_a)

    distances = []
    for smiles in set_b:
        try:
            # Query the closest distance for each SMILES in set B
            closest = smiles_set_a.query(smiles, k=1)
            distances.append(closest[0][1])  # Append the distance
        except ValueError as e:
            print(f"Skipping invalid SMILES in set B: {smiles}")

    # Calculate statistics
    distances = np.array(distances)
    stats = {
        "min": distances.min() if len(distances) > 0 else None,
        "max": distances.max() if len(distances) > 0 else None,
        "mean": distances.mean() if len(distances) > 0 else None,
        "std": distances.std() if len(distances) > 0 else None
    }

    return stats























from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import random





# Function to calculate the Tanimoto similarity between two fingerprints
def tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)


# Function to generate fingerprints for a list of SMILES
def generate_fingerprints(smiles_list):
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:  # Valid SMILES check
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fingerprints.append(fp)
        else:
            fingerprints.append(None)  # Mark invalid SMILES
    return fingerprints


# Function to create separated sets A and B
# Function to create diverse sets A and B
def create_diverse_sets(smiles_list, set_a_size, set_b_size, threshold=0.7):
    fingerprints = generate_fingerprints(smiles_list)

    # Filter out invalid fingerprints
    valid_data = [(smiles, fp) for smiles, fp in zip(smiles_list, fingerprints) if fp is not None]
    random.shuffle(valid_data)  # Shuffle for randomness

    set_a = []
    set_b = []

    for smiles, fp in valid_data:
        # Fill set A first
        if len(set_a) < set_a_size:
            set_a.append((smiles, fp))
        # Add to set B only if diverse relative to set A
        elif len(set_b) < set_b_size:
            if all(tanimoto_similarity(fp, a_fp) < threshold for _, a_fp in set_a):
                set_b.append((smiles, fp))

        # Stop when both sets are filled
        if len(set_a) >= set_a_size and len(set_b) >= set_b_size:
            break

    # Extract SMILES strings from the sets
    set_a_smiles = [smiles for smiles, _ in set_a]
    set_b_smiles = [smiles for smiles, _ in set_b]

    return set_a_smiles, set_b_smiles


def export_smiles_to_file(smiles_list, filename):
    """
    Export a list of SMILES strings to a file, one per line.
    """
    with open(filename, 'w') as file:
        for smiles in smiles_list:
            file.write(smiles + '\n')

if __name__ == "__main__":

    smiles_list = filter_smiles("C:\datasets\chembl_size90_input_smiles.csv", max_number=800000, min_atoms=8, max_atoms=32, return_original=True, return_scrambled=1)
    random.shuffle(smiles_list)
    set_a, set_b = create_diverse_sets(smiles_list, set_a_size=8000, set_b_size=128000, threshold=0.875)
    export_smiles_to_file(set_a,"smilesSet2_A.txt")
    export_smiles_to_file(set_b, "smilesSet2_B.txt")

# # Example usage with dummy data
# sample_smiles = [
#                     "CCO", "CCN", "CCC", "CCCl", "CBr", "CN", "C=O", "COC", "CCF", "CCOCC",
#                     "CCCC", "CC(C)O", "C1CC1", "C1CCC1", "C1CCCCC1", "C1CCOCC1"
#                 ] * 100  # Simulate a large dataset
#
# set_a, set_b = create_diverse_sets(sample_smiles, set_a_size=64, set_b_size=8, threshold=0.7)
#
# # Display results
# import pandas as pd
#
# result_df = pd.DataFrame({
#     "Set A": set_a,
#     "Set B": set_b + [""] * (len(set_a) - len(set_b))  # Pad B to match A's length for display
# })
# print(result_df)


# Example usage:
# smiles_list = filter_smiles('smiles.txt', max_number=100, max_atoms=50, return_original=True, return_scrambled=3)
# print(smiles_list)
