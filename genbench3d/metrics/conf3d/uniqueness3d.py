from genbench3d.conf_ensemble import GeneratedCEL
from genbench3d.metrics import Metric
from scipy.cluster.hierarchy import linkage, fcluster
from genbench3d.params import DEFAULT_TFD_THRESHOLD

class Uniqueness3D(Metric):
    
    def __init__(self,
                 tfd_threshold: float = DEFAULT_TFD_THRESHOLD,
                 name: str = 'Uniqueness3D') -> None:
        super().__init__(name)
        self.tfd_threshold = tfd_threshold
        self.value = None
    
    def get(self, 
            cel: GeneratedCEL) -> float:
        self.n_unique = 0
        self.unique_mol = {}
        self.clusters = {}

        for name, ce in cel.items():
            
            n_confs = ce.mol.GetNumConformers()

            if n_confs == 1:
                self.n_unique += 1
            elif n_confs > 1:
                mol = ce.mol
  
                try:
                    tfd_matrix = cel.get_tfd_matrix(name)

                    Z = linkage(tfd_matrix)

                    max_value = self.tfd_threshold
                    T = fcluster(Z, 
                                t=max_value, 
                                criterion='distance')
                    n_clusters = max(T)
                    self.n_unique += n_clusters
                    self.clusters[mol] = T
                    
                    if n_clusters == 1 and mol.GetNumConformers() > 1:
                        self.unique_mol[name] = ce.mol
                except Exception as e:
                    print('Uniqueness 3D exception:', e)
                    
            else:
                print('Conf ensemble without conformers, please check')
                import pdb;pdb.set_trace()
                raise RuntimeError()
                
        if self.n_unique == 0:
            self.value = 1
        else:
            self.value = self.n_unique / cel.n_total_confs
                
        return self.value