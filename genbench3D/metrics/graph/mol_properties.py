import os
import sys
import logging
import numpy as np

from typing import Dict
from rdkit import Chem
from rdkit.Chem import (Descriptors,
                        Crippen,
                        QED,
                        RDConfig)

from genbench3D.conf_ensemble import GeneratedCEL
from ..metric import Metric

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer
    
MOL_PROPERTY_CALCULATORS = {'MW': Descriptors.MolWt,
                            'logP': Crippen.MolLogP,
                            'SAScore': sascorer.calculateScore,
                            'QED' : QED.qed
                            }
    
class MolProperty(Metric):
    
    def __init__(self, 
                 name: str) -> None:
        assert name in MOL_PROPERTY_CALCULATORS
        super().__init__(name)
        self.calculator = MOL_PROPERTY_CALCULATORS[name]
        self.value: float = None
        self.values: Dict[str, float] = {}

    def get(self, 
            cel: GeneratedCEL) -> float:
        all_values = []
        for name, ce in cel.items():
            mol = ce.mol
            # SAScore is calculated on mol without Hs
            # Does not change for other properties
            mol = Chem.RemoveHs(mol) 
            
            try:
                value = self.calculator(mol)
                all_values.append(value)
                self.values[name] = value
            except:
                # import pdb;pdb.set_trace()
                logging.warning(f'Computing of {self.name} for {name} has failed')
        
        return all_values