from typing import List, Dict
from rdkit.Chem import Mol
from genbench3d.params import BIO_CONF_DIRNAME, DATA_DIRPATH
from .conf_ensemble_library import ConfEnsembleLibrary

class GeneratedCEL(ConfEnsembleLibrary):
    
    def __init__(self, 
                 cel_name: str = BIO_CONF_DIRNAME, 
                 root: str = DATA_DIRPATH, 
                 load: bool = True) -> None:
        super().__init__(cel_name, 
                         root,
                         load)
        self.n_total_mols = None
        self.n_total_confs = None
        self.n_total_graphs = None
    
    
    @classmethod
    def from_mol_list(cls,
                      mol_list: List[Mol], 
                      cel_name: str = BIO_CONF_DIRNAME,
                      root: str = DATA_DIRPATH,
                      names: List[str] = None,
                      standardize: bool = False,
                      n_total_mols: int = None) -> ConfEnsembleLibrary:
        cel = super().from_mol_list(mol_list, 
                                    cel_name, 
                                    root, 
                                    names, 
                                    standardize)
        
        if n_total_mols is not None:
            cel.n_total_mols = n_total_mols
            # allow the number to be higher, 
            # in case all mols were not parsed or saved during generation
        else:
            cel.n_total_mols = len(mol_list)
        cel.n_total_confs = sum([mol.GetNumConformers() for mol in cel.itermols()])
        cel.n_total_graphs = len(cel)
        return cel
    
    
    @classmethod
    def get_cel_subset(cls,
                       cel: 'ConfEnsembleLibrary',
                       subset_conf_ids: Dict[str, List[int]]):
        new_cel = super().get_cel_subset(cel,
                                         subset_conf_ids)
        new_cel.n_total_confs = sum([mol.GetNumConformers() for mol in cel.itermols()])
        new_cel.n_total_graphs = len(cel)
        return new_cel