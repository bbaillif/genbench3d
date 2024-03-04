import logging
import pickle

from rdkit import Chem
from rdkit.Chem import Mol
from collections import defaultdict
from tqdm import tqdm
from .reference_geometry import ReferenceGeometry
from genbench3d.params import DATA_DIRPATH
from ccdc.io import Molecule
from genbench3d.utils import ccdc_mol_to_rdkit_mol

class CrossDockedGeometry(ReferenceGeometry):
    
    def __init__(self, 
                 root: str = DATA_DIRPATH, 
                 source_name: str ='CrossDocked',
                 validity_method: str='mixtures') -> None:
        super().__init__(root, source_name, validity_method)
        
        
    def get_mol_iterator(self):
        cd_ligands_path = '/home/bb596/hdd/ThreeDGenMolBenchmark/train_ligand_cd.sdf'
        return Chem.SDMolSupplier(cd_ligands_path)
    
    
    def compute_values(self):
        
        logging.info(f'Compiling geometry values for {self.source_name}')
        
        mol_iterator = self.get_mol_iterator()

        random_idxs = range(len(mol_iterator))
        
        all_bond_values = defaultdict(list)
        all_angle_values = defaultdict(list)
        all_torsion_values = defaultdict(list)
        
        # for mol in tqdm(mol_iterator):
        for i in tqdm(random_idxs):
            original_mol = mol_iterator[int(i)]
            if isinstance(original_mol, Molecule):
                try:
                    mol = ccdc_mol_to_rdkit_mol(original_mol)
                except Exception as e:
                    logging.warning('CCDC mol could not be converted to RDKit :' + str(e))
                    mol = None
            else:
                mol = original_mol
            
            if mol is not None:
                assert isinstance(mol, Mol)
                
                mol_bond_values = self.get_mol_bond_lengths(mol)
                for bond_tuple, bond_values in mol_bond_values.items():
                    all_bond_values[bond_tuple].extend(bond_values)
                
                mol_angle_values = self.get_mol_angle_values(mol)
                for angle_tuple, angle_values in mol_angle_values.items():
                    all_angle_values[angle_tuple].extend(angle_values)
                    
                mol_torsion_values = self.get_mol_torsion_values(mol)
                for torsion_tuple, torsion_values in mol_torsion_values.items():
                    all_torsion_values[torsion_tuple].extend(torsion_values)
        
        values = {'bond': all_bond_values,
                  'angle': all_angle_values,
                  'torsion': all_torsion_values}
        
        with open(self.values_filepath, 'wb') as f:
            pickle.dump(values, f)
            
        return values