import logging

from genbench3d.conf_ensemble import ConfEnsembleLibrary
from rdkit.Chem import Mol
from .metrics import (VinaScorer,
                      VinaScore,
                      GlideScore,
                      GoldPLPScore,
                      IFPSimilarity,
                      ESPSIM,
                      StericClash,
                      DistanceToNativeCentroid)
from .genbench3d import GenBench3D
from .data.structure import (VinaProtein,
                             GlideProtein,
                             Pocket)
from genbench3d.conf_ensemble import GeneratedCEL
from genbench3d.metrics import Metric

class SBGenBench3D(GenBench3D):
    
    def __init__(self, 
                 vina_protein: VinaProtein,
                 glide_protein: GlideProtein,
                 pocket: Pocket,
                 native_ligand: Mol,
                 root: str = 'genbench3d_results/', 
                 training_mols: list = None, 
                 show_plots: bool = False, 
                 tfd_threshold: float = 0.2) -> None:
        super().__init__(root, 
                         training_mols, 
                         show_plots, 
                         tfd_threshold)
        
        self.vina_protein = vina_protein
        self.glide_protein = glide_protein
        self.pocket = pocket
        self.native_ligand = native_ligand # should have Hs
        
        # Vina score
        self._vina_scorer = VinaScorer.from_ligand(ligand=self.native_ligand,
                                                   vina_protein=vina_protein)
        self.native_ligand_vina_score = self._vina_scorer.score_mol(native_ligand)[0]
        self.vina_score = VinaScore(self._vina_scorer)
        
        # Minimized Vina score
        self.min_vina_score = VinaScore(self._vina_scorer, 
                                        name='Minimized Vina score',
                                        minimized=True)
        self.native_ligand_min_vina_score = self._vina_scorer.score_mol(native_ligand, 
                                                                    minimized=True)[0]
        
        # Glide score
        native_cel = ConfEnsembleLibrary.from_mol_list([self.native_ligand])
        self.glide_score = GlideScore(glide_protein)
        self.native_ligand_glide_score = self.glide_score.get(native_cel)[0]
        
        # Minimized Glide score
        self.min_glide_score = GlideScore(glide_protein, 
                                          mininplace=True, 
                                          name='Minimized Glide score')
        self.native_ligand_min_glide_score = self.min_glide_score.get(native_cel)[0]
        
        self.plp_score = GoldPLPScore(protein_path=self.vina_protein.protein_clean_filepath,
                                      native_ligand=native_ligand)
        self.native_ligand_plp_score = self.plp_score.gold_scorer.score_mols([native_ligand])[0]
        
        # self.ifp_similarity = IFPSimilarity(universe=self.pocket.protein.universe,
        #                                     native_ligand=self.native_ligand)
        # self.espsim = ESPSIM(native_ligand=self.native_ligand)
        self.steric_clash = StericClash(pocket)
        self.distance_to_centroid = DistanceToNativeCentroid(pocket)
        self.sbmetrics: list[Metric] = [self.vina_score,
                                        self.min_vina_score,
                                        self.glide_score,
                                        self.min_glide_score,
                                        self.plp_score,
                                        # self.ifp_similarity,
                                        # self.espsim,
                                        self.steric_clash,
                                        self.distance_to_centroid]
        
        
    def get_results_for_mol_list(self, 
                                 mols: list[Mol], 
                                 n_total_mols: int = None,
                                 valid_only: bool = False,
                                 do_conf_analysis: bool = False) -> dict[str, float]:
        if valid_only:
            assert do_conf_analysis, 'Conformation analysis must be done to activate the valid only mode'

        cel = GeneratedCEL.from_mol_list(mol_list=mols, 
                                         n_total_mols=n_total_mols)
        if do_conf_analysis:
            self.results = super().get_results_for_cel(cel)
        else:
            self.results = {}
        
        if valid_only:
            valid_conf_ids = self.validity3D_csd.valid_conf_ids
            # print(len(cel))
            cel = GeneratedCEL.get_cel_subset(cel, 
                                              subset_conf_ids=valid_conf_ids)
            # print(len(cel))
        
        if len(cel) > 0:
        
            for metric in self.sbmetrics:
                logging.info(f'Evaluating {metric.name}')
                self.results[metric.name] = metric.get(cel)
                
            relative_vina_scores = [score - self.native_ligand_vina_score
                            for score in self.results[self.vina_score.name]]
            self.results['Relative Vina score'] = relative_vina_scores
            
            relative_min_vina_scores = [score - self.native_ligand_min_vina_score
                            for score in self.results[self.min_vina_score.name]]
            self.results['Relative Min Vina score'] = relative_min_vina_scores
            
            relative_glide_scores = [score - self.native_ligand_glide_score
                            for score in self.results[self.glide_score.name]]
            self.results['Relative Glide score'] = relative_glide_scores
            
            relative_min_glide_scores = [score - self.native_ligand_min_glide_score
                            for score in self.results[self.min_glide_score.name]]
            self.results['Relative Min Glide score'] = relative_min_glide_scores
            
            relative_plp_scores = [score - self.native_ligand_plp_score
                            for score in self.results[self.plp_score.name]]
            self.results['Relative PLP score'] = relative_plp_scores
        
        return dict(self.results)