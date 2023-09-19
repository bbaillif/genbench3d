from abc import ABC, abstractmethod
from genbench3d.conf_ensemble import (GeneratedCEL,
                                      ConfEnsembleLibrary)

class Metric(ABC):
    
    def __init__(self,
                 name: str) -> None:
        self.name = name
        self.value = None
        
    @abstractmethod
    def get(self,
            cel: GeneratedCEL) -> float:
        pass
    
    
class TrainingMetric(Metric, ABC):
    
    @abstractmethod
    def get(self,
            cel: GeneratedCEL,
            training_cel: ConfEnsembleLibrary) -> float:
        pass