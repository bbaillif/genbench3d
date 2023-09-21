import logging
import copy

from typing import List, Dict
from rdkit import Chem
from rdkit.Chem import Mol
from .metrics import (VinaScorer,
                      VinaScore,
                      GoldPLPScore,
                      IFPSimilarity,
                      ESPSIM,
                      StericClash)
from .genbench3d import GenBench3D
from .data.structure import (VinaProtein,
                             Pocket)
from genbench3d.conf_ensemble import GeneratedCEL
from genbench3d.metrics import Metric
# from ccdc.docking import Docker
# from ccdc.io import MoleculeReader, Entry

class SBGenBench3D(GenBench3D):
    
    def __init__(self, 
                 vina_protein: VinaProtein,
                 pocket: Pocket,
                 native_ligand: Mol,
                 root: str = 'genbench3d_results/', 
                 training_mols: List = None, 
                 show_plots: bool = False, 
                 tfd_threshold: float = 0.2) -> None:
        super().__init__(root, 
                         training_mols, 
                         show_plots, 
                         tfd_threshold)
        
        self.vina_protein = vina_protein
        self.pocket = pocket
        self.native_ligand = Chem.AddHs(native_ligand, addCoords=True)
        
        self._vina_scorer = VinaScorer.from_ligand(ligand=self.native_ligand,
                                                   vina_protein=vina_protein)
        self.native_ligand_vina_score = self._vina_scorer.score_mol(self.native_ligand)[0]
        self.vina_score = VinaScore(self._vina_scorer)
        # self.relative_vina_score = VinaScore(self._vina_scorer,
        #                                      reference_score=self.native_ligand_vina_score,
        #                                      name='Relative Vina score')
        self.plp_score = GoldPLPScore(protein_path=self.vina_protein.protein_clean_filepath,
                                      native_ligand=native_ligand)
        self.native_ligand_plp_score = self.plp_score.gold_scorer.score_mols([native_ligand])[0]
        self.ifp_similarity = IFPSimilarity(universe=self.pocket.protein.universe,
                                            native_ligand=self.native_ligand)
        self.espsim = ESPSIM(native_ligand=self.native_ligand)
        self.steric_clash = StericClash(pocket)
        self.sbmetrics: List[Metric] = [self.vina_score,
                                        self.plp_score,
                                        self.ifp_similarity,
                                        self.espsim,
                                        self.steric_clash]
        
        
    def get_results_for_mol_list(self, 
                                 mols: List[Mol], 
                                 n_total_mols: int = None) -> Dict[str, float]:
        cel = GeneratedCEL.from_mol_list(mol_list=mols, 
                                         n_total_mols=n_total_mols)
        self.results = super().get_results_for_cel(cel)
        
        for metric in self.sbmetrics:
            logging.info(f'Evaluating {metric.name}')
            self.results[metric.name] = metric.get(cel)
            
        relative_vina_scores = [score - self.native_ligand_vina_score
                           for score in self.results[self.vina_score.name]]
        self.results['Relative Vina score'] = relative_vina_scores
        
        relative_plp_scores = [score - self.native_ligand_vina_score
                           for score in self.results[self.plp_score.name]]
        self.results['Relative PLP score'] = relative_plp_scores
        
        # absolute_vina_scores = self.get_vina_score(ligands=mols)
        # self.absolute_vina_scores = np.array(absolute_vina_scores)
        # self.results['Median absolute Vina score'] = np.nanmedian(self.absolute_vina_scores)
        
        # self.relative_vina_scores = self.absolute_vina_scores - self.native_ligand_vina_score
        # median_vina_relative = np.nanmedian(self.relative_vina_scores)
        # self.results['Median Vina score relative to test ligand'] = median_vina_relative

        # self.ifp_sims = self.get_ifp_sims(ligands=mols)
        # median_ifp_sim = np.nanmedian(self.ifp_sims)
        # self.results['Median IFP similarity to test ligand'] = median_ifp_sim
        
        # self.espsims = self.get_espsims(ligands=mols)
        # median_espsim = np.nanmedian(self.espsims)
        # self.results['Median ESPSIM to test ligand'] = median_espsim
        
        
        return dict(self.results)

    
    # def gold_plp_score_ligands(self,
    #                         ligands: List[Mol],
    #                         native_ligand_filepath: str):
    #     self.gold_plp_scores = []
    #     docker = Docker()
    #     settings = docker.settings
    #     settings.output_directory = 'docking_results'
    #     settings.fitness_function = None
    #     settings.rescore_function = 'plp'
        
    #     receptor_filepath = self.original_structure_path.replace('.pdb', '_protein_only_clean.pdb')
    #     settings.add_protein_file(receptor_filepath)

    #     protein = settings.proteins[0]
    #     native_ligand = MoleculeReader(native_ligand_filepath)[0]
    #     settings.binding_site = settings.BindingSiteFromLigand(protein=protein, ligand=native_ligand, distance=10.)
        
    #     for i, gen_mol in enumerate(ligands):
    #         ligand_filename = f'docking_results/test_lig_{i}.sdf'
        
    #         with Chem.SDWriter(ligand_filename) as writer:
    #             writer.write(gen_mol)
    #         ligand = MoleculeReader(ligand_filename)[0]
    #         ligand_entry = Entry.from_molecule(ligand)

    #         ligand_preparation = Docker.LigandPreparation()
    #         ligand_preped = ligand_preparation.prepare(ligand_entry)
    #         ligand_mol2_filename = ligand_filename.replace('sdf', 'mol2')
    #         mol2_string = ligand_preped.to_string(format='mol2')
    #         with open(ligand_mol2_filename, 'w') as writer :
    #             writer.write(mol2_string)
    #         settings.add_ligand_file(ligand_mol2_filename, 1)

    #     docker.dock()
        
    #     scores = []
    #     for docked_ligand in docker.results.ligands:
    #         score = docked_ligand.attributes['Gold.PLP.Fitness']
    #         scores.append(score)
        
    #     self.gold_plp_scores.extend(scores)
    