import numpy as np
import logging
import time

from genbench3d.conf_ensemble import GeneratedCEL
from ..metric import Metric
from .vina_scorer import VinaScorer
from genbench3d.data.structure import Protein

class VinaScore(Metric):
    
    def __init__(self, 
                 vina_scorer: VinaScorer,
                 reference_score: float = 0,
                 name: str = 'Vina score',
                 ) -> None:
        super().__init__(name)
        self.scores = {}
        self.reference_score = reference_score
        self.vina_scorer = vina_scorer
        
        
    def get(self, 
            cel: GeneratedCEL) -> float:
        all_scores = []
        for name, ce in cel.items():
            ce_scores = []
            mols = ce.to_mol_list()
            for mol in mols:
                try:
                    # start_time = time.time()
                    energies = self.vina_scorer.score_mol(mol)
                    # logging.info(f'Time: {time.time() - start_time}')
                    score = energies[0]
                    relative_score = score - self.reference_score
                    ce_scores.append(relative_score)
                except Exception as e:
                    logging.warning(f'Vina scoring error: {e}')
            self.scores[name] = ce_scores
            all_scores.extend(ce_scores)
        
        return all_scores
            