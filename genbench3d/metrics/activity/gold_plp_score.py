from genbench3d.conf_ensemble import GeneratedCEL
from genbench3d.metrics import Metric
from rdkit import Chem
from rdkit.Chem import Mol
from .gold_scorer import GoldScorer

class GoldPLPScore(Metric):
    
    def __init__(self, 
                 protein_path: str,
                 native_ligand: Mol,
                 name: str = 'Gold PLP score',
                 ) -> None:
        super().__init__(name)
        self.protein_path = protein_path
        self.gold_scorer = GoldScorer(protein_path,
                                      native_ligand)
        self.plp_scores = {}
        
        
    def get(self, 
            cel: GeneratedCEL) -> float:
        mols = cel.to_mol_list()
        all_scores = self.gold_scorer.score_mols(mols)
        
        # all_scores = []
        # for name, ce in cel.items():
        #     mols = ce.to_mol_list()
        #     scores = self.gold_scorer.score_mols(mols)
        #     self.plp_scores[name] = scores
        #     all_scores.extend(scores)
        
        return all_scores