import numpy as np
import multiprocessing
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import AllChem
from typing import Iterable, Any

class CCDCNotAvailableError(Exception):
    pass

def get_full_matrix_from_tril(tril_matrix: Iterable[Any], 
                              n: int) -> np.ndarray:
    # let the desired full distance matrix between 4 samples be
    # [a b c d
    #  e f g h
    #  i j k l
    #  m n o p]
    # where the diagonal a = f = l = p = 0
    # having b = e, c = i, d = m, g = j, h = n, l = o by symmetry
    # scipy squareform works on triu [b c d g h l]
    # while we want to work from tril [e i j m n o]
    
    matrix = np.zeros((n, n))
    i=1
    j=0
    for v in tril_matrix:
        matrix[i, j] = matrix[j, i] = v
        j = j + 1
        if j == i:
            i = i + 1
            j = 0
    return matrix

try:
    from ccdc.io import Molecule
    CCDC_IMPORTED = True
except ImportError:
    class Molecule: pass
    CCDC_IMPORTED = False
    
def rdkit_conf_to_ccdc_mol(rdkit_mol: Mol, 
                            conf_id: int = -1) -> Molecule:
    """Create a ccdc molecule for a given conformation from a rdkit molecule
    Communication via mol block
    
    :param rdkit_mol: RDKit molecule
    :type rdkit_mol: Mol
    :param conf_id: Conformer ID in the RDKit molecule
    :type conf_id: int
    :return: CCDC molecule
    :rtype: Molecule
    
    """
    if not CCDC_IMPORTED:
        raise CCDCNotAvailableError("CCDC is not available")
    molblock = Chem.MolToMolBlock(rdkit_mol, 
                                    confId=conf_id)
    molecule: Molecule = Molecule.from_string(molblock)
    return molecule

def ccdc_mol_to_rdkit_mol(ccdc_mol: Molecule) -> Mol:
    """Transforms a ccdc molecule to an rdkit molecule

    :param ccdc_mol: CCDC molecule
    :type ccdc_mol: Molecule
    :return: RDKit molecule
    :rtype: Mol
    """
    
    if not CCDC_IMPORTED:
        raise CCDCNotAvailableError("CCDC is not available")
    
    # First line is necessary in case the ccdc mol is a DockedLigand
    # because it contains "fake" atoms with atomic_number lower than 1
    ccdc_mol.remove_atoms([atom 
                            for atom in ccdc_mol.atoms 
                            if atom.atomic_number < 1])
    mol2block = ccdc_mol.to_string()
    
    return Chem.MolFromMol2Block(mol2block, 
                                    removeHs=False)
    
def shift_torsion_values(values: list[float],
                        x: float):
    '''
    shift a distribution of degrees such that the current x becomes the -180.
    '''
    assert (x >= -180) and (x <= 180)
    values = np.array(values)
    positive_values = values + 180 # -180 
    positive_shift = x + 180
    shifted_values = positive_values - positive_shift
    new_values = shifted_values % 360
    centred_new_values = new_values - 180
    return centred_new_values


def unshift_torsion_values(values: list[float],
                            x: float):
    '''
    shift a distribution of degrees such that the current x becomes the 0.
    '''
    return shift_torsion_values(values, x=-x)
        
        
def shift_abs_torsion_values(values: list[float],
                            x: float):
    '''
    shift a distribution of degrees such that the current x becomes the 0.
    '''
    return (values - x) % 180
    
    
def unshift_abs_torsion_values(values: list[float],
                                x: float):
    '''
    reverse shift: current 0 of a degree distribution becomes the new x
    '''
    return (values + x) % 180


def preprocess_mols(mol_list):
    new_mol_list = []
    for mol in mol_list:
        mol_was_parsed = mol is not None
        if mol_was_parsed:
            mol_is_not_empty = mol.GetNumAtoms() > 0
            try:
                mol_is_single_fragment = not '.' in Chem.MolToSmiles(mol)
                if mol_is_not_empty and mol_is_single_fragment:
                    new_mol_list.append(mol)
            except Exception as e:
                pass
    return new_mol_list


def add_hs_to_mols(mol_list,
                   n_threads=12):
    mol_list = [Chem.AddHs(mol, addCoords=True) for mol in mol_list]
    
    # See https://github.com/rdkit/rdkit/discussions/4528
    def minimize_thread(mol):
        mp = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
        ff = AllChem.MMFFGetMoleculeForceField(mol, mp)
        for a in mol.GetAtoms():
            if (a.GetAtomicNum() > 1):
                ff.MMFFAddPositionConstraint(a.GetIdx(), 0.0, 1.e4)
        ff.Minimize(maxIts=10000)
        return mol

    pool = multiprocessing.Pool(n_threads)

    min_mols = pool.map(minimize_thread, mol_list)

    return min_mols