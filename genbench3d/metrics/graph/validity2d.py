from genbench3d.conf_ensemble import GeneratedCEL
from genbench3d.metrics import Metric

class Validity2D(Metric):
    
    def __init__(self,
                 name: str = 'Validity2D') -> None:
        super().__init__(name)
        self.value = None
    
    def get(self, 
            cel: GeneratedCEL) -> float:
        self.value = cel.n_total_confs / cel.n_total_mols
        if self.value > 1:
            import pdb;pdb.set_trace()
        return self.value