from genbench3D.conf_ensemble import GeneratedCEL
from genbench3D.metrics import Metric

class Uniqueness2D(Metric):
    
    def __init__(self,
                 name: str = 'Uniqueness2D') -> None:
        super().__init__(name)
        self.value = None
    
    def get(self, 
            cel: GeneratedCEL) -> float:
        self.value = cel.n_total_graphs / cel.n_total_confs
        return self.value