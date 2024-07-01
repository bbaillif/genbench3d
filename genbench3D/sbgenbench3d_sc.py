import logging

from genbench3d.conf_ensemble import ConfEnsembleLibrary
from rdkit.Chem import Mol
from .metrics import (VinaScorer,
                      VinaScore,
                      GlideScore,
                      GoldPLPScore,
                    #   IFPSimilarity,
                    #   ESPSIM,
                      StericClash,
                      DistanceToNativeCentroid)
from .genbench3d import GenBench3D
from .data.structure import (VinaProtein,
                             GlideProtein,
                             Protein,
                             Pocket)
from genbench3d.conf_ensemble import GeneratedCEL
from genbench3d.metrics import Metric
from typing import Any
from genbench3d.params import ADD_MINIMIZED_DOCKING_SCORES, DEFAULT_TFD_THRESHOLD

class SBGenBench3DSC(GenBench3D):
    
    def __init__(self, 
                 pocket: Pocket,
                 native_ligand: Mol,
                 root: str = 'genbench3d_results/', 
                 show_plots: bool = False, 
                 tfd_threshold: float = DEFAULT_TFD_THRESHOLD) -> None:
        super().__init__(root, 
                         show_plots, 
                         tfd_threshold)
        
        self.pocket = pocket
        self.native_ligand = native_ligand # should have Hs
        
        # self.ifp_similarity = IFPSimilarity(universe=self.pocket.protein.universe,
        #                                     native_ligand=self.native_ligand)
        # self.espsim = ESPSIM(native_ligand=self.native_ligand)
        self.steric_clash = StericClash(pocket)
        self.distance_to_centroid = DistanceToNativeCentroid(pocket)
        self.sb_metrics: list[Metric] = [
                                        self.steric_clash,
                                        self.distance_to_centroid
                                        # self.ifp_similarity,
                                        # self.espsim,
                                        ]
        
        
    def setup_vina(self,
                   vina_protein: VinaProtein,
                   add_minimized: bool = ADD_MINIMIZED_DOCKING_SCORES) -> None:
        self.vina_protein = vina_protein
        
        # Vina score
        self._vina_scorer = VinaScorer.from_ligand(ligand=self.native_ligand,
                                                   vina_protein=vina_protein)
        self.native_ligand_vina_score = self._vina_scorer.score_mol(self.native_ligand)[0]
        self.vina_score = VinaScore(self._vina_scorer)
        self.sb_metrics.append(self.vina_score)
        
        if add_minimized:
            self.min_vina_score = VinaScore(self._vina_scorer, 
                                            name='Minimized Vina score',
                                            minimized=True)
            self.native_ligand_min_vina_score = self._vina_scorer.score_mol(self.native_ligand, 
                                                                            minimized=True)[0]
            self.sb_metrics.append(self.min_vina_score)
        
        
    def setup_glide(self,
                    glide_protein: GlideProtein,
                    add_minimized: bool = ADD_MINIMIZED_DOCKING_SCORES) -> None:
        self.glide_protein = glide_protein
        
        # Glide score
        native_cel = ConfEnsembleLibrary.from_mol_list([self.native_ligand])
        self.glide_score = GlideScore(glide_protein)
        self.native_ligand_glide_score = self.glide_score.get(native_cel)[0]
        self.sb_metrics.append(self.glide_score)
        
        if add_minimized:
            # Minimized Glide score
            self.min_glide_score = GlideScore(glide_protein, 
                                            mininplace=True, 
                                            name='Minimized Glide score')
            self.native_ligand_min_glide_score = self.min_glide_score.get(native_cel)[0]
            self.sb_metrics.append(self.min_glide_score)


    def setup_gold_plp(self,
                       protein: Protein) -> None:
        self.plp_score = GoldPLPScore(protein_path=protein.protein_clean_filepath,
                                      native_ligand=self.native_ligand)
        self.native_ligand_plp_score = self.plp_score.gold_scorer.score_mols([self.native_ligand])[0]
        self.sb_metrics.append(self.plp_score)

        
    def get_results_for_mol_list(self, 
                                 mols: list[Mol], 
                                 n_total_mols: int = None,
                                 valid_only: bool = False,
                                 do_conf_analysis: bool = False) -> dict[str, Any]:
        assert len(mols) > 0, 'Empty list was given as input'
        if valid_only:
            assert do_conf_analysis, 'Conformation analysis must be done to activate the valid only mode'

        cel = GeneratedCEL.from_mol_list(mol_list=mols, 
                                         n_total_mols=n_total_mols)
        if do_conf_analysis:
            results = super().get_results_for_cel(cel)
        else:
            self.results = {}
        
        if valid_only:
            valid_conf_ids = self.validity3D_csd.valid_conf_ids
            # print(len(cel))
            cel = GeneratedCEL.get_cel_subset(cel, 
                                              subset_conf_ids=valid_conf_ids)
            # print(len(cel))
        
        if len(cel) > 0:
            results = self.get_sb_results_for_cel(cel)
            
        return results
    
    
    def get_ligand_only_results_for_mol_list(self,
                                             mols: list[Mol], 
                                            n_total_mols: int = None,
                                            ) -> dict[str, Any]:
        cel = GeneratedCEL.from_mol_list(mol_list=mols, 
                                         n_total_mols=n_total_mols)
        return self.get_ligand_only_results_for_cel(cel)
    
    
    def get_ligand_only_results_for_cel(self,
                                        cel: GeneratedCEL):
        self.results = super().get_results_for_cel(cel)
        return dict(self.results)
    
    
    def get_sb_results_for_cel(self,
                               cel: GeneratedCEL):
        assert len(cel) > 0, 'There is no molecule in the CEL'
        for metric in self.sb_metrics:
            logging.info(f'Evaluating {metric.name}')
            self.results[metric.name] = metric.get(cel)
            
        if hasattr(self, 'native_ligand_vina_score'):
            relative_vina_scores = [score - self.native_ligand_vina_score
                            for score in self.results[self.vina_score.name]]
            self.results['Relative Vina score'] = relative_vina_scores
            
        if hasattr(self, 'native_ligand_min_vina_score'):
            relative_min_vina_scores = [score - self.native_ligand_min_vina_score
                            for score in self.results[self.min_vina_score.name]]
            self.results['Relative Min Vina score'] = relative_min_vina_scores
        
        if hasattr(self, 'native_ligand_glide_score'):
            relative_glide_scores = [score - self.native_ligand_glide_score
                            for score in self.results[self.glide_score.name]]
            self.results['Relative Glide score'] = relative_glide_scores
        
        if hasattr(self, 'native_ligand_min_glide_score'):
            relative_min_glide_scores = [score - self.native_ligand_min_glide_score
                            for score in self.results[self.min_glide_score.name]]
            self.results['Relative Min Glide score'] = relative_min_glide_scores
        
        if hasattr(self, 'native_ligand_plp_score'):
            relative_plp_scores = [score - self.native_ligand_plp_score
                            for score in self.results[self.plp_score.name]]
            self.results['Relative PLP score'] = relative_plp_scores
        
        return dict(self.results)