from ..metric import Metric
from genbench3d.conf_ensemble import GeneratedCEL
from typing import Dict
from collections import Counter

class RingProportion(Metric) :
    
    def __init__(self, 
                 name: str = 'Ring proportion') -> None:
        super().__init__(name)
        
    def get(self,
            cel: GeneratedCEL) -> Dict[int, float]:
        all_ring_sizes = []
        for mol in cel.itermols():
            ring_info = mol.GetRingInfo()
            rings = ring_info.AtomRings()
            ring_sizes = [len(ring) for ring in rings]
            all_ring_sizes.extend(ring_sizes)
            
        self.counter = Counter(all_ring_sizes)
        self.value = {k: v / self.counter.total()
                      for k, v in self.counter.items()}
        return self.value