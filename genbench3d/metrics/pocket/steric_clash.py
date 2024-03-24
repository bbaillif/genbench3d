from ..metric import Metric
from genbench3d.conf_ensemble import GeneratedCEL
from genbench3d.data.structure import Pocket
from rdkit import Chem
from genbench3d.geometry import GeometryExtractor
from genbench3d.geometry import ClashChecker
from rdkit.Chem import Mol
from collections import defaultdict
from genbench3d.params import CLASH_SAFETY_RATIO, CONSIDER_HYDROGENS

class StericClash(Metric):
    
    def __init__(self, 
                 pocket: Pocket,
                 name: str = 'Steric clash',
                 clash_safety_ratio: float = CLASH_SAFETY_RATIO,
                 consider_hs: bool = CONSIDER_HYDROGENS
                 ) -> None:
        super().__init__(name)
        self.pocket = pocket
        self.geometry_extractor = GeometryExtractor()
        self.clash_checker = ClashChecker(clash_safety_ratio)
        self.clashes = None
        self.valid_pldist_conf_ids = None
        
    
    def get(self,
            cel: GeneratedCEL) -> float:
        
        self.clashes = {}
        self.valid_pldist_conf_ids = defaultdict(list)
        self.n_valid = 0
        all_n_clashes = []
        
        for name, ce in cel.items():
            ce_clashes = {}
            
            conf_ids = [conf.GetId() for conf in ce.mol.GetConformers()]
            for conf_id in conf_ids:
                
                clashes = self.clash_checker.get_pocket_ligand_clashes(pocket=self.pocket,
                                                                       ligand=ce.mol,
                                                                       conf_id=conf_id)
                all_n_clashes.append(len(clashes))
                                    
                if len(clashes) == 0:
                    self.valid_pldist_conf_ids[name].append(conf_id)
                    self.n_valid += 1
                else:
                    ce_clashes[conf_id] = clashes
                
            if len(ce_clashes) > 0:
                self.clashes[name] = ce_clashes
                
        self.value = self.n_valid / cel.n_total_confs
                    
        return all_n_clashes