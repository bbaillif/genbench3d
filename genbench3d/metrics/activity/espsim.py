import logging
import numpy as np

from genbench3d.conf_ensemble import GeneratedCEL
from ..metric import Metric
from rdkit.Chem import Mol
from rdkit import Chem
from espsim import GetEspSim

class ESPSIM(Metric):
    
    def __init__(self, 
                 native_ligand: Mol,
                 name: str = 'ESPSIM') -> None:
        super().__init__(name)
        self.native_ligand = native_ligand
        self.espsims = None
        
        
    def get(self, 
            cel: GeneratedCEL) -> float:
        all_espsims = []
        self.espsims = {}
        for name, ce in cel.items():
            espsims = []
            mols = ce.to_mol_list()
            for ligand in mols:
                try:
                    ligand = Chem.AddHs(ligand, addCoords=True)
                    espsim = GetEspSim(ligand, self.native_ligand, renormalize=True)
                except Exception as e:
                    logging.warning(f'ESPSIM computation error: {e}')
                    espsim = np.nan
                espsims.append(espsim)
            self.espsims[name] = espsims
            all_espsims.extend(espsims)
            
        return all_espsims