from .data_source import DataSource
from rdkit import Chem

class SDFSource(DataSource):
    
    def __init__(self,
                 ligands_path: str,
                 name: str,
                 removeHs: bool = False,
                 ) -> None:
        super().__init__(name)
        self.ligands_path = ligands_path
        self.removeHs = removeHs
    
    def __iter__(self,):
        for mol in Chem.SDMolSupplier(self.ligands_path, removeHs=self.removeHs):
            yield mol