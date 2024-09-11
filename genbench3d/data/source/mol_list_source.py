from .data_source import DataSource
from rdkit import Chem

class MolListSource(DataSource):
    
    def __init__(self,
                 mol_list: str,
                 name: str,
                 ) -> None:
        super().__init__(name)
        self.mol_list = mol_list
    
    def __iter__(self):
        for mol in self.mol_list:
            yield mol