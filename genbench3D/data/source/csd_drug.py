import logging

from .data_source import DataSource
from rdkit import Chem
from ccdc.io import MoleculeReader
from genbench3D.utils import ccdc_mol_to_rdkit_mol
from rdkit.Chem.MolStandardize import rdMolStandardize

class CSDDrug(DataSource):
    
    def __init__(self,
                 subset_path: str,
                 name: str = 'CSDDrug',
                 ) -> None:
        super().__init__(name)
        self.subset_path = subset_path
        
        self.subset_csd_ids = []
        with open(self.subset_path, 'r') as f:
            for line in f.readlines():
                self.subset_csd_ids.append(line.strip())
    
    def __iter__(self):
        csd_reader = MoleculeReader('CSD')
        for csd_id in self.subset_csd_ids:
            try:
                original_mol = csd_reader.molecule(csd_id)
                mol = ccdc_mol_to_rdkit_mol(original_mol)
                if mol is not None:
                    largest_Fragment = rdMolStandardize.LargestFragmentChooser()
                    mol = largest_Fragment.choose(mol)
                    include = True
                    for bond in mol.GetBonds():
                        if bond.GetBondTypeAsDouble() == 0.0:
                            logging.warning(f'Invalid bond in CSD Drug: {Chem.MolToSmiles(mol)}')
                            include = False
                            break
                    if include:
                        yield mol
            except Exception as e:
                logging.warning(e)
                
    
        