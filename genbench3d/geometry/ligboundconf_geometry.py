from .reference_geometry import ReferenceGeometry
from genbench3d.params import DATA_DIRPATH, LIGBOUNDCONF_MINIMIZED_FILEPATH
from rdkit import Chem

class LigBoundConfGeometry(ReferenceGeometry):
    
    def __init__(self, 
                 root: str = DATA_DIRPATH, 
                 source_name: str ='LigBoundConf_minimized') -> None:
        super().__init__(root, source_name)
        
        
    def load_ligands(self):
        minimized_path = LIGBOUNDCONF_MINIMIZED_FILEPATH
        ligands = [mol for mol in Chem.SDMolSupplier(minimized_path) if mol is not None]
        return ligands