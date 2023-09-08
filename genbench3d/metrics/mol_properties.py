import os
import sys
from typing import List
from rdkit.Chem import (Mol,
                        Descriptors,
                        Crippen,
                        QED,
                        RDConfig)

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer
    
class MolPropertiesCalculator():
    
    @staticmethod   
    def get_mw(mol: Mol) -> List:
        return Descriptors.MolWt(mol)
    
    @staticmethod
    def get_logp(mol: Mol) -> List:
        return Crippen.MolLogP(mol)
    
    @staticmethod
    def get_sascore(mol: Mol) -> List:
        try:
            sascore = sascorer.calculateScore(mol)
        except:
            sascore = None
        return sascore
    
    @staticmethod
    def get_qed(mol: Mol) -> List:
        return QED.qed(mol)