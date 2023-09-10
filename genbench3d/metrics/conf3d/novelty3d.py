from genbench3d.conf_ensemble import (GeneratedCEL,
                                      ConfEnsembleLibrary)
from genbench3d.metrics import TrainingMetric
from collections import defaultdict
from rdkit.Chem.TorsionFingerprints import GetTFDBetweenMolecules
from genbench3d.params import DEFAULT_TFD_THRESHOLD

class Novelty3D(TrainingMetric):
    
    def __init__(self,
                 tfd_threshold: float = DEFAULT_TFD_THRESHOLD,
                 name: str = 'Novelty3D') -> None:
        super().__init__(name)
        self.tfd_threshold = tfd_threshold
        self.value = None
        self.n_novel = None
        self.novel_conf_ids = None
    
    def get(self, 
            cel: GeneratedCEL,
            training_cel: ConfEnsembleLibrary,
            ) -> float:
        
        self.n_novel = 0
        self.novel_conf_ids = defaultdict(list)
        
        for name, ce in cel.items():
            
            ce = cel[name]
            n_confs = ce.mol.GetNumConformers()
            
            if not name in training_cel:
                self.n_novel += n_confs
            else:
                mol = ce.mol
                        
                training_ce = training_cel[name]
                training_mol = training_ce.mol
                    
                for conf1 in mol.GetConformers():
                    conf_id1 = conf1.GetId()
                    tfds = []
                    is_novel = False
                    for conf2 in training_mol.GetConformers():
                        conf_id2 = conf2.GetId()
                        tfd = GetTFDBetweenMolecules(mol1=mol, 
                                                     mol2=training_mol, 
                                                     confId1=conf_id1, 
                                                     confId2=conf_id2)
                        tfds.append(tfd)
                        is_novel = tfd > self.tfd_threshold
                        if is_novel: 
                            self.novel_conf_ids[name].append(conf_id1)
                            self.n_novel += 1
                            break   
        
        if self.n_novel == 0:
            self.value = 0
        else:
            self.value = self.n_unique / cel.n_total_confs
        
        return self.value