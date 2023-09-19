import numpy as np
import logging

from genbench3d.conf_ensemble import GeneratedCEL
from genbench3d.metrics import Metric
from rdkit.Chem.TorsionFingerprints import GetTFDMatrix
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from collections import defaultdict

class Diversity3D(Metric):
    
    def __init__(self,
                 name: str = 'Diversity3D') -> None:
        super().__init__(name)
        self.value = None
    
    def get(self, 
            cel: GeneratedCEL) -> float:
        icds = []
        self.icds = defaultdict(list)
        for name, ce in cel.items():
            
            try:
                n_confs = ce.mol.GetNumConformers()
                
                if n_confs > 1:
                    tfd_matrix = cel.get_tfd_matrix(name)
                    icd = np.mean(tfd_matrix)
                    
                    icds.append(icd)
                    self.icds[name].append(icd)
            except Exception as e:
                logging.warning(f'Diversity 3D exception: {e}')
                
        if len(icds) > 0:
            diversity_3D = np.mean([icd 
                        for icd in icds 
                        if icd != 0])
        else:
            diversity_3D = 0
            
        self.value = diversity_3D
            
        return self.value