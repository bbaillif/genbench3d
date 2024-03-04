import logging

from .reference_geometry import ReferenceGeometry
from genbench3d.params import DATA_DIRPATH
from rdkit import Chem
from ccdc import io
from genbench3d.utils import ccdc_mol_to_rdkit_mol
from tqdm import tqdm

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

class CSDGeometry(ReferenceGeometry):
    
    def __init__(self, 
                 root: str = DATA_DIRPATH, 
                 source_name: str ='CSD',
                 validity_method: str = 'mixtures') -> None:
        super().__init__(root, source_name, validity_method)
        
        
    def get_mol_iterator(self):
        logging.info('Loading molecules')
        csd_reader = io.MoleculeReader('CSD')
        # print(len(csd_reader))
        # mols = []
        # for ccdc_mol in tqdm(csd_reader):
        #     try:
        #         mol = ccdc_mol_to_rdkit_mol(ccdc_mol)
        #         if mol is not None:
        #             mols.append(mol)
        #     except Exception as e:
        #         print(e)
                
        return csd_reader