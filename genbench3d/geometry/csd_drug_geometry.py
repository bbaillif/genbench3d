import logging
import pickle

from .reference_geometry import ReferenceGeometry
from genbench3d.params import DATA_DIRPATH, CSD_DRUG_SUBSET_PATH
from rdkit import Chem
from rdkit.Chem import Mol
from ccdc import io
from ccdc.io import Molecule
from genbench3d.utils import ccdc_mol_to_rdkit_mol
from tqdm import tqdm
from collections import defaultdict


from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class CSDDrugGeometry(ReferenceGeometry):
    
    def __init__(self, 
                 root: str = DATA_DIRPATH, 
                 source_name: str ='CSDDrug',
                 validity_method: str = 'mixtures',
                 subset_path: str = CSD_DRUG_SUBSET_PATH) -> None:
        
        # Download drug subset from : https://ars.els-cdn.com/content/image/1-s2.0-S0022354918308104-mmc2.zip
        self.subset_path = subset_path
        self.subset_csd_ids = []
        with open(self.subset_path, 'r') as f:
            for line in f.readlines():
                self.subset_csd_ids.append(line.strip())
        
        super().__init__(root, source_name, validity_method)
        
        
    def get_mol_iterator(self):
        logging.info('Loading molecules')
        csd_reader = io.MoleculeReader('CSD')
                
        return csd_reader
    
    def compute_values(self):
        
        logging.info(f'Compiling geometry values for {self.source_name}')
        
        mol_iterator = self.get_mol_iterator()
        
        all_bond_values = defaultdict(list)
        all_angle_values = defaultdict(list)
        all_torsion_values = defaultdict(list)
        
        # for mol in tqdm(mol_iterator):
        for csd_id in tqdm(self.subset_csd_ids):
            try:
                original_mol = mol_iterator.molecule(csd_id)
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
                    Chem.SanitizeMol(mol)
                    
                    mol_bond_values = self.get_mol_bond_lengths(mol)
                    for bond_pattern, bond_values in mol_bond_values.items():
                        all_bond_values[bond_pattern].extend(bond_values)
                    
                    mol_angle_values = self.get_mol_angle_values(mol)
                    for angle_pattern, angle_values in mol_angle_values.items():
                        all_angle_values[angle_pattern].extend(angle_values)
                        
                    mol_torsion_values = self.get_mol_torsion_values(mol)
                    for torsion_pattern, torsion_values in mol_torsion_values.items():
                        all_torsion_values[torsion_pattern].extend(torsion_values)
            except RuntimeError as e:
                logging.warning(e)
        
        values = {'bond': all_bond_values,
                  'angle': all_angle_values,
                  'torsion': all_torsion_values}
        
        with open(self.values_filepath, 'wb') as f:
            pickle.dump(values, f)
            
        return values